import pandas as pd
import numpy as np

def calculate_returns(data):
    """
    Calculates daily percentage returns for a given price series.
    """
    if data is None or data.empty:
        return None

    # Calculate daily returns
    returns = data.pct_change().dropna()
    return returns

import logging

logger = logging.getLogger(__name__)

def align_and_clean_data(returns_df):
    """
    Drops missing values to align data across all assets.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns for multiple assets.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if returns_df is None or returns_df.empty:
        return None

    # Drop rows where any value is missing (typically weekends for traditional markets)
    aligned_data = returns_df.dropna()
    return aligned_data

def calculate_correlation_matrix(aligned_data):
    """
    Calculates the full correlation matrix for all assets over the entire period.

    Args:
        aligned_data (pd.DataFrame): Dataframe containing asset returns.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    if aligned_data is None or aligned_data.empty:
        return None
    return aligned_data.corr()

def calculate_rolling_correlations(aligned_data, primary_asset, window=30):
    """
    Calculates rolling correlations between a primary asset and all other assets.

    Args:
        aligned_data (pd.DataFrame): Dataframe containing asset returns.
        primary_asset (str): The column name of the primary asset to correlate against.
        window (int): The rolling window size (in days).

    Returns:
        pd.DataFrame: Dataframe containing the rolling correlation series for each other asset.
    """
    if aligned_data is None or aligned_data.empty or primary_asset not in aligned_data.columns:
        return None

    if len(aligned_data) < window:
        logger.warning(f"Not enough data points ({len(aligned_data)}) for a rolling window of {window}.")
        return None

    rolling_corrs = pd.DataFrame(index=aligned_data.index)

    for col in aligned_data.columns:
        if col != primary_asset:
            corr_series = aligned_data[primary_asset].rolling(window=window).corr(aligned_data[col])
            rolling_corrs[f"{primary_asset} vs {col}"] = corr_series

    return rolling_corrs.dropna(how='all')
