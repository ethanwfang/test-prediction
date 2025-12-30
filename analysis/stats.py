"""
Statistical analysis functions for Kalshi market data.
"""

import pandas as pd
import numpy as np


def market_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for market info.

    Args:
        df: Market info DataFrame

    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {'total_markets': 0}

    summary = {
        'total_markets': len(df),
        'total_volume': int(df['volume'].sum()) if 'volume' in df.columns else 0,
        'avg_volume': float(df['volume'].mean()) if 'volume' in df.columns else 0,
        'total_open_interest': int(df['open_interest'].sum()) if 'open_interest' in df.columns else 0,
    }

    # Status breakdown
    if 'status' in df.columns:
        summary['by_status'] = df['status'].value_counts().to_dict()

    # Top markets by volume
    if 'volume' in df.columns:
        top = df.nlargest(10, 'volume')[['ticker', 'title', 'volume', 'status']]
        if 'result' in df.columns:
            top = df.nlargest(10, 'volume')[['ticker', 'title', 'volume', 'status', 'result']]
        summary['top_by_volume'] = top.to_dict('records')

    # Date range
    if 'created_time' in df.columns:
        valid_dates = df['created_time'].dropna()
        if not valid_dates.empty:
            summary['earliest_market'] = valid_dates.min().isoformat()
            summary['latest_market'] = valid_dates.max().isoformat()

    return summary


def outcome_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze outcomes for finalized markets.

    Args:
        df: Market info DataFrame

    Returns:
        Dictionary with outcome statistics
    """
    if df.empty or 'status' not in df.columns:
        return {'finalized_count': 0}

    finalized = df[df['status'] == 'finalized']

    if finalized.empty:
        return {'finalized_count': 0}

    analysis = {
        'finalized_count': len(finalized),
        'finalized_pct': len(finalized) / len(df) * 100,
    }

    # Outcome breakdown
    if 'result' in finalized.columns:
        result_counts = finalized['result'].value_counts()
        analysis['resolved_yes'] = int(result_counts.get('yes', 0))
        analysis['resolved_no'] = int(result_counts.get('no', 0))

        # Average last price for winners vs losers
        yes_markets = finalized[finalized['result'] == 'yes']
        no_markets = finalized[finalized['result'] == 'no']

        if 'last_price' in finalized.columns:
            if not yes_markets.empty:
                analysis['avg_price_resolved_yes'] = float(yes_markets['last_price'].mean())
            if not no_markets.empty:
                analysis['avg_price_resolved_no'] = float(no_markets['last_price'].mean())

    # Volume of finalized markets
    if 'volume' in finalized.columns:
        analysis['finalized_volume'] = int(finalized['volume'].sum())
        analysis['avg_finalized_volume'] = float(finalized['volume'].mean())

    return analysis


def price_volatility(candles: pd.DataFrame, ticker: str = None) -> dict:
    """
    Calculate price volatility metrics.

    Args:
        candles: Candlesticks DataFrame
        ticker: Optional specific market ticker

    Returns:
        Dictionary with volatility metrics
    """
    if candles.empty:
        return {}

    if ticker:
        data = candles[candles['market_ticker'] == ticker]
    else:
        data = candles

    if data.empty:
        return {}

    metrics = {}

    if 'high_price' in data.columns and 'low_price' in data.columns:
        # Overall price range
        metrics['price_high'] = float(data['high_price'].max())
        metrics['price_low'] = float(data['low_price'].min())
        metrics['price_range'] = metrics['price_high'] - metrics['price_low']

        # Average daily range
        daily_range = data['high_price'] - data['low_price']
        metrics['avg_daily_range'] = float(daily_range.mean())

    if 'close_price' in data.columns:
        # Price at start and end
        sorted_data = data.sort_values('datetime')
        metrics['first_price'] = float(sorted_data['close_price'].iloc[0])
        metrics['last_price'] = float(sorted_data['close_price'].iloc[-1])
        metrics['price_change'] = metrics['last_price'] - metrics['first_price']

        # Simple volatility (std dev of daily returns)
        if len(sorted_data) > 1:
            returns = sorted_data['close_price'].pct_change().dropna()
            if not returns.empty and not returns.isna().all():
                metrics['volatility'] = float(returns.std())

    if 'volume' in data.columns:
        metrics['total_volume'] = int(data['volume'].sum())
        metrics['avg_daily_volume'] = float(data['volume'].mean())
        metrics['trading_days'] = len(data)

    return metrics


def per_market_stats(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each market.

    Args:
        candles: Candlesticks DataFrame

    Returns:
        DataFrame with per-market statistics
    """
    if candles.empty or 'market_ticker' not in candles.columns:
        return pd.DataFrame()

    stats = []
    for ticker in candles['market_ticker'].unique():
        market_data = candles[candles['market_ticker'] == ticker]
        vol = price_volatility(market_data)
        vol['ticker'] = ticker
        if 'market_title' in market_data.columns:
            vol['title'] = market_data['market_title'].iloc[0]
        stats.append(vol)

    return pd.DataFrame(stats)


def trading_patterns(trades: pd.DataFrame) -> dict:
    """
    Analyze trading patterns from trade data.

    Args:
        trades: Trades DataFrame

    Returns:
        Dictionary with trading pattern statistics
    """
    if trades.empty:
        return {}

    patterns = {
        'total_trades': len(trades),
    }

    if 'count' in trades.columns:
        patterns['total_contracts'] = int(trades['count'].sum())
        patterns['avg_trade_size'] = float(trades['count'].mean())
        patterns['median_trade_size'] = float(trades['count'].median())
        patterns['max_trade_size'] = int(trades['count'].max())

    # Taker side analysis
    if 'taker_side' in trades.columns:
        side_counts = trades['taker_side'].value_counts()
        patterns['yes_taker_trades'] = int(side_counts.get('yes', 0))
        patterns['no_taker_trades'] = int(side_counts.get('no', 0))

        if 'count' in trades.columns:
            yes_volume = trades[trades['taker_side'] == 'yes']['count'].sum()
            no_volume = trades[trades['taker_side'] == 'no']['count'].sum()
            patterns['yes_taker_volume'] = int(yes_volume)
            patterns['no_taker_volume'] = int(no_volume)
            total_vol = yes_volume + no_volume
            if total_vol > 0:
                patterns['yes_taker_pct'] = float(yes_volume / total_vol * 100)

    # Time patterns
    if 'datetime' in trades.columns:
        valid_times = trades['datetime'].dropna()
        if not valid_times.empty:
            patterns['first_trade'] = valid_times.min().isoformat()
            patterns['last_trade'] = valid_times.max().isoformat()

            # Hour of day distribution
            hours = valid_times.dt.hour
            patterns['peak_hour'] = int(hours.mode().iloc[0]) if not hours.empty else None

    return patterns
