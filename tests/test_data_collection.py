import pytest
import pandas as pd
from src.data_collection import fetch_data, fetch_multiple_assets

def test_fetch_data_single_asset_success(mocker):
    # Mock yfinance download to return a valid DataFrame
    mock_df = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.date_range('2020-01-01', periods=3))
    mocker.patch('yfinance.download', return_value=mock_df)

    result = fetch_data('DUMMY', '2020-01-01', '2020-01-03')

    assert result is not None
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert result.iloc[0] == 100

def test_fetch_data_multiple_assets_success(mocker):
    # Mock yfinance download to return a valid DataFrame with multiple columns
    dates = pd.date_range('2020-01-01', periods=3)
    mock_df = pd.DataFrame({
        ('Close', 'DUMMY1'): [100, 101, 102],
        ('Close', 'DUMMY2'): [200, 201, 202]
    }, index=dates)

    # Simulate yfinance multi-index format when passing multiple tickers
    mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)
    mocker.patch('yfinance.download', return_value=mock_df)

    result = fetch_data(['DUMMY1', 'DUMMY2'], '2020-01-01', '2020-01-03')

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['DUMMY1', 'DUMMY2']
    assert len(result) == 3

def test_fetch_data_empty(mocker):
    # Mock yfinance to return an empty DataFrame
    mock_df = pd.DataFrame()
    mocker.patch('yfinance.download', return_value=mock_df)

    result = fetch_data('DUMMY', '2020-01-01', '2020-01-03')

    assert result is None

def test_fetch_multiple_assets(mocker):
    # Test wrapping function to ensure single asset returns a DF with the column named properly
    dummy_series = pd.Series([1, 2, 3], name='Close')
    mocker.patch('src.data_collection.fetch_data', return_value=dummy_series)

    result = fetch_multiple_assets(['DUMMY1'], '2020-01-01', '2020-01-03')

    assert isinstance(result, pd.DataFrame)
    assert 'DUMMY1' in result.columns
    assert len(result) == 3
