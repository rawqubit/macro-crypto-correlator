import pytest
import pandas as pd
import numpy as np
from src.analysis import calculate_returns, align_and_clean_data, calculate_correlation_matrix, calculate_rolling_correlations

def test_calculate_returns():
    data = pd.Series([100, 110, 104.5])
    returns = calculate_returns(data)

    assert returns is not None
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], 0.10) # (110-100)/100
    assert np.isclose(returns.iloc[1], -0.05) # (104.5-110)/110

def test_calculate_returns_empty():
    assert calculate_returns(None) is None
    assert calculate_returns(pd.Series(dtype=float)) is None

def test_align_and_clean_data():
    dates = pd.date_range('2020-01-01', periods=5)
    crypto = pd.Series([1, 2, np.nan, 4, 5], index=dates)
    macro = pd.Series([10, 20, 30, 40, np.nan], index=dates)

    df = pd.DataFrame({'Crypto': crypto, 'Macro': macro})

    aligned = align_and_clean_data(df)

    assert aligned is not None
    assert len(aligned) == 3 # Should drop index 2 (np.nan in crypto) and index 4 (np.nan in macro)
    assert aligned.index[0] == pd.Timestamp('2020-01-01')
    assert aligned.index[1] == pd.Timestamp('2020-01-02')
    assert aligned.index[2] == pd.Timestamp('2020-01-04')

def test_calculate_correlation_matrix():
    # Constructing perfect positive correlation
    crypto = pd.Series([1, 2, 3, 4, 5])
    macro1 = pd.Series([2, 4, 6, 8, 10])
    macro2 = pd.Series([5, 4, 3, 2, 1]) # Negative correlation

    aligned = pd.DataFrame({'Crypto': crypto, 'Macro1': macro1, 'Macro2': macro2})

    corr_matrix = calculate_correlation_matrix(aligned)

    assert corr_matrix is not None
    assert corr_matrix.shape == (3, 3)
    assert np.isclose(corr_matrix.loc['Crypto', 'Macro1'], 1.0)
    assert np.isclose(corr_matrix.loc['Crypto', 'Macro2'], -1.0)

def test_calculate_rolling_correlations():
    # Constructing perfect positive correlation for a 3-day window
    crypto = pd.Series([1, 2, 3, 4, 5])
    macro1 = pd.Series([2, 4, 6, 8, 10])
    aligned = pd.DataFrame({'Crypto': crypto, 'Macro1': macro1})

    corr_df = calculate_rolling_correlations(aligned, 'Crypto', window=3)

    assert corr_df is not None
    assert 'Crypto vs Macro1' in corr_df.columns
    assert len(corr_df) == 3 # 5 data points - 3 window + 1 = 3
    # Check that the correlation is 1.0 (or very close)
    assert np.isclose(corr_df['Crypto vs Macro1'].iloc[0], 1.0)
    assert np.isclose(corr_df['Crypto vs Macro1'].iloc[1], 1.0)
    assert np.isclose(corr_df['Crypto vs Macro1'].iloc[2], 1.0)

def test_calculate_rolling_correlations_insufficient_data():
    aligned = pd.DataFrame({'Crypto': [1, 2], 'Macro1': [2, 4]})
    corr_df = calculate_rolling_correlations(aligned, 'Crypto', window=3)

    assert corr_df is None
