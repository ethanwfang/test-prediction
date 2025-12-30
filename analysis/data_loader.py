"""
Data loading utilities for Kalshi market analysis.
"""

import glob
from pathlib import Path

import pandas as pd


def find_latest_file(data_dir: str, pattern: str) -> str:
    """Find the most recent file matching pattern in data_dir."""
    files = glob.glob(str(Path(data_dir) / pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
    return max(files)  # Filenames include timestamps, so max = latest


def load_market_info(filepath: str) -> pd.DataFrame:
    """
    Load market info CSV with proper dtypes.

    Args:
        filepath: Path to markets_info CSV file

    Returns:
        DataFrame with parsed timestamps and normalized prices
    """
    df = pd.read_csv(filepath)

    # Parse timestamp columns
    time_cols = ['created_time', 'open_time', 'close_time', 'expiration_time',
                 'expected_expiration_time', 'latest_expiration_time']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Ensure price columns are numeric (already in cents 0-100)
    price_cols = ['last_price', 'yes_bid', 'yes_ask', 'no_bid', 'no_ask']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure volume columns are numeric
    vol_cols = ['volume', 'volume_24h', 'open_interest', 'liquidity']
    for col in vol_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df


def load_candlesticks(filepath: str) -> pd.DataFrame:
    """
    Load candlesticks CSV with proper dtypes.

    Args:
        filepath: Path to candlesticks CSV file

    Returns:
        DataFrame with parsed datetime and OHLCV columns
    """
    df = pd.read_csv(filepath)

    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Ensure OHLCV columns are numeric (in cents 0-100)
    ohlc_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in ohlc_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by market and datetime
    if 'market_ticker' in df.columns and 'datetime' in df.columns:
        df = df.sort_values(['market_ticker', 'datetime'])

    return df


def load_trades(filepath: str) -> pd.DataFrame:
    """
    Load trades CSV with proper dtypes.

    Args:
        filepath: Path to trades CSV file

    Returns:
        DataFrame with parsed timestamps and trade data
    """
    df = pd.read_csv(filepath)

    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    elif 'created_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['created_time'], errors='coerce')

    # Ensure price columns are numeric
    price_cols = ['yes_price', 'no_price', 'price']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure count is numeric
    if 'count' in df.columns:
        df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)

    return df


def load_all_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data files from a directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Tuple of (market_info, candlesticks, trades) DataFrames
    """
    market_info = None
    candlesticks = None
    trades = None

    # Find and load market info
    try:
        filepath = find_latest_file(data_dir, "*markets_info*.csv")
        market_info = load_market_info(filepath)
        print(f"  Loaded {len(market_info)} markets from {Path(filepath).name}")
    except FileNotFoundError:
        print("  No market info file found")
        market_info = pd.DataFrame()

    # Find and load candlesticks
    try:
        filepath = find_latest_file(data_dir, "*candlesticks*.csv")
        candlesticks = load_candlesticks(filepath)
        print(f"  Loaded {len(candlesticks)} candlesticks from {Path(filepath).name}")
    except FileNotFoundError:
        print("  No candlesticks file found")
        candlesticks = pd.DataFrame()

    # Find and load trades
    try:
        filepath = find_latest_file(data_dir, "*trades*.csv")
        trades = load_trades(filepath)
        print(f"  Loaded {len(trades)} trades from {Path(filepath).name}")
    except FileNotFoundError:
        print("  No trades file found")
        trades = pd.DataFrame()

    return market_info, candlesticks, trades
