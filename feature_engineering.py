import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_tickers(file_path: str) -> list:
    df = pd.read_csv(file_path)
    return df['symbol'].tolist()

def load_market_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df

def add_lagged_features(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    for lag in range(1, lags + 1):
        df[f'return_{lag}d'] = df['return'].shift(lag)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:

    def compute_rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_macd(series: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.Series:
        exp1 = series.ewm(span=short_window, adjust=False).mean()
        exp2 = series.ewm(span=long_window, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd - signal
    
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['rsi_14'] = compute_rsi(df['close'], window=14)
    df['macd'] = compute_macd(df['close'])
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()

    cols_to_scale = [
        'return', 'log_return',
        'return_1d', 'return_3d', 'return_5d',
        'sma_5', 'sma_10', 'macd'
    ]

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

def binarize_features(df: pd.DataFrame) -> pd.DataFrame:
    zero_based = ['return', 'log_return', 'macd',
                  'return_1d', 'return_3d', 'return_5d']
    
    for col in zero_based:
        df[col + "_bin"] = (df[col] > 0).astype(int)

    median_based = ['sma_5', 'sma_10']
    for col in median_based:
        threshold = df[col].median()
        df[col + "_bin"] = (df[col] > threshold).astype(int)

    df['rsi_14_bin'] = ((df['rsi_14'] > 70) | (df['rsi_14'] < 30)).astype(int)

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_returns(df)
    df = add_lagged_features(df, 1)
    df = add_lagged_features(df, 3)
    df = add_lagged_features(df, 5)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df = scale_features(df)
    df = binarize_features(df)
    df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df