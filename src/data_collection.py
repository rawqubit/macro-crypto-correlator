import yfinance as yf
import pandas as pd
import requests_cache
import logging

logger = logging.getLogger(__name__)

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical data for given tickers from Yahoo Finance.

    Args:
        tickers (str or list): The ticker symbol(s) (e.g., 'BTC-USD', ['BTC-USD', '^GSPC']).
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame or pd.Series: A pandas structure containing the prices.
    """
    try:
        # Note: requests_cache is not supported by yfinance curl_cffi implementation currently.
        # We rely on yfinance's built-in session handling.
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data found for {tickers}")
            return None

        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        elif 'Close' in data.columns:
            price_data = data['Close']
        else:
            logger.error(f"Could not find 'Close' or 'Adj Close' in data for {tickers}")
            return None

        return price_data.squeeze() if isinstance(price_data, pd.DataFrame) and price_data.shape[1] == 1 else price_data

    except Exception as e:
        logger.error(f"Error fetching data for {tickers}: {e}")
        return None

def fetch_multiple_assets(assets, start_date, end_date):
    """
    Fetches historical data for a list of assets.

    Args:
        assets (list): A list of ticker symbols.
        start_date (str): The start date.
        end_date (str): The end date.

    Returns:
        pd.DataFrame: DataFrame containing price data for all requested assets.
    """
    data = fetch_data(assets, start_date, end_date)

    if isinstance(data, pd.Series):
        return pd.DataFrame({assets[0]: data})

    return data
