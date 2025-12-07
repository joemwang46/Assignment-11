import pandas as pd
import numpy as np
from feature_engineering import (
    calculate_returns, add_lagged_features, add_technical_indicators,
    scale_features, binarize_features, add_features
)

def sample_df():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=50, freq="D"),
        "close": np.linspace(100, 150, 50)
    }).set_index("date")


def test_calculate_returns():
    df = calculate_returns(sample_df().copy())
    assert "return" in df.columns
    assert "log_return" in df.columns
    assert df["return"].isna().sum() == 1


def test_add_lagged_features():
    df = calculate_returns(sample_df().copy())
    df = add_lagged_features(df, 3)
    assert "return_1d" in df.columns
    assert "return_3d" in df.columns


def test_add_technical_indicators():
    df = add_technical_indicators(sample_df().copy())
    assert "sma_5" in df.columns
    assert "rsi_14" in df.columns
    assert "macd" in df.columns


def test_binarize_features():
    df = add_features(sample_df().copy())
    assert "return_bin" in df.columns
    assert "macd_bin" in df.columns
    assert "rsi_14_bin" in df.columns


def test_add_features_pipeline():
    df = add_features(sample_df().copy())
    assert "direction" in df.columns
    assert df.isna().sum().sum() == 0
