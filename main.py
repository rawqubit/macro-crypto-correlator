import os
import argparse
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from src.data_collection import fetch_multiple_assets
from src.analysis import calculate_returns, align_and_clean_data, calculate_correlation_matrix, calculate_rolling_correlations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Macro-Crypto Correlator Dashboard Generator")
    parser.add_argument('--primary', type=str, default='BTC-USD', help='Primary asset ticker (e.g., BTC-USD)')
    parser.add_argument('--secondary', type=str, nargs='+', default=['^GSPC', 'GC=F', '^TNX'], help='Secondary asset tickers (e.g., ^GSPC GC=F)')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--window', type=int, default=30, help='Rolling correlation window size in days')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for the dashboard')
    return parser.parse_args()

def main():
    args = parse_args()

    all_assets = [args.primary] + args.secondary
    logger.info(f"Fetching data for {', '.join(all_assets)} from {args.start} to {args.end}...")

    data_df = fetch_multiple_assets(all_assets, args.start, args.end)

    if data_df is None or data_df.empty:
        logger.error("Failed to fetch data. Exiting.")
        return

    logger.info("Calculating daily returns...")
    returns_df = calculate_returns(data_df)

    if returns_df is None or returns_df.empty:
        logger.error("Failed to calculate returns. Exiting.")
        return

    logger.info("Aligning and cleaning data...")
    aligned_data = align_and_clean_data(returns_df)

    if aligned_data is None or aligned_data.empty:
        logger.error("Data alignment failed. Exiting.")
        return

    logger.info("Calculating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(aligned_data)

    logger.info(f"Calculating {args.window}-day rolling correlations against {args.primary}...")
    rolling_corrs = calculate_rolling_correlations(aligned_data, args.primary, window=args.window)

    if rolling_corrs is None or rolling_corrs.empty:
        logger.error("Failed to calculate rolling correlations. Exiting.")
        return

    logger.info("Generating Plotly dashboard...")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"Overall Return Correlation Matrix", f"{args.window}-Day Rolling Correlation against {args.primary}"),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6]
    )

    # Add heatmap for correlation matrix
    heatmap = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        colorbar=dict(title='Correlation', x=1.0)
    )
    fig.add_trace(heatmap, row=1, col=1)

    # Add rolling correlation lines
    for col in rolling_corrs.columns:
        fig.add_trace(
            go.Scatter(x=rolling_corrs.index, y=rolling_corrs[col], mode='lines', name=col),
            row=2, col=1
        )

    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"Macro-Crypto Correlation Analysis ({args.start} to {args.end})",
        showlegend=True,
        template='plotly_dark'
    )
    fig.update_yaxes(title_text="Assets", row=1, col=1)
    fig.update_yaxes(title_text="Correlation Coefficient", range=[-1.1, 1.1], row=2, col=1)

    # Save the dashboard
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, 'dashboard.html')
    fig.write_html(output_path)
    logger.info(f"Dashboard saved successfully to {output_path}")

if __name__ == "__main__":
    main()
