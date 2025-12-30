"""
Statistical feature extraction from market data.

Extracts comprehensive statistics from candlestick and trade data
for LLM context generation.
"""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats


def extract_price_journey(candles: pd.DataFrame) -> dict:
    """
    Extract key price journey statistics.

    Args:
        candles: DataFrame with datetime, close_price, high_price, low_price, volume

    Returns:
        Dictionary with price journey stats
    """
    if candles.empty:
        return {}

    df = candles.sort_values('datetime').copy()

    # Basic start/end
    start_row = df.iloc[0]
    end_row = df.iloc[-1]

    # Min/max
    min_idx = df['close_price'].idxmin()
    max_idx = df['close_price'].idxmax()
    min_row = df.loc[min_idx]
    max_row = df.loc[max_idx]

    journey = {
        "start": {
            "date": start_row['datetime'].strftime('%Y-%m-%d') if pd.notna(start_row['datetime']) else None,
            "price": float(start_row['close_price'])
        },
        "end": {
            "date": end_row['datetime'].strftime('%Y-%m-%d') if pd.notna(end_row['datetime']) else None,
            "price": float(end_row['close_price'])
        },
        "min": {
            "date": min_row['datetime'].strftime('%Y-%m-%d') if pd.notna(min_row['datetime']) else None,
            "price": float(min_row['close_price'])
        },
        "max": {
            "date": max_row['datetime'].strftime('%Y-%m-%d') if pd.notna(max_row['datetime']) else None,
            "price": float(max_row['close_price'])
        },
        "total_change": float(end_row['close_price'] - start_row['close_price']),
        "total_change_pct": float((end_row['close_price'] - start_row['close_price']) / max(start_row['close_price'], 1) * 100),
        "total_days": (end_row['datetime'] - start_row['datetime']).days if pd.notna(end_row['datetime']) else 0,
        "price_range": float(max_row['close_price'] - min_row['close_price']),
    }

    # Price at timeline milestones (25%, 50%, 75%)
    n = len(df)
    if n >= 4:
        journey["at_25_pct"] = {
            "date": df.iloc[n // 4]['datetime'].strftime('%Y-%m-%d'),
            "price": float(df.iloc[n // 4]['close_price'])
        }
        journey["at_50_pct"] = {
            "date": df.iloc[n // 2]['datetime'].strftime('%Y-%m-%d'),
            "price": float(df.iloc[n // 2]['close_price'])
        }
        journey["at_75_pct"] = {
            "date": df.iloc[3 * n // 4]['datetime'].strftime('%Y-%m-%d'),
            "price": float(df.iloc[3 * n // 4]['close_price'])
        }

    return journey


def extract_trend_analysis(candles: pd.DataFrame) -> dict:
    """
    Extract trend direction and strength metrics.

    Args:
        candles: DataFrame with datetime, close_price

    Returns:
        Dictionary with trend analysis
    """
    if candles.empty or len(candles) < 3:
        return {}

    df = candles.sort_values('datetime').copy()
    prices = df['close_price'].values

    # Linear regression for overall trend
    x = np.arange(len(prices))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)

    # Determine direction
    if slope > 0.1:
        direction = "bullish"
    elif slope < -0.1:
        direction = "bearish"
    else:
        direction = "sideways"

    # Trend strength: how consistently did it move in one direction?
    daily_changes = np.diff(prices)
    if len(daily_changes) > 0:
        positive_days = np.sum(daily_changes > 0)
        negative_days = np.sum(daily_changes < 0)
        total_days = len(daily_changes)
        consistency = max(positive_days, negative_days) / total_days if total_days > 0 else 0
    else:
        consistency = 0

    trend = {
        "direction": direction,
        "strength": round(consistency, 3),
        "linear_slope": round(slope, 4),
        "r_squared": round(r_value ** 2, 3),
    }

    # Detect trend segments (simplified: split into 3-4 segments)
    segments = _detect_trend_segments(df)
    if segments:
        trend["segments"] = segments

    return trend


def _detect_trend_segments(candles: pd.DataFrame, min_segment_days: int = 14) -> list:
    """
    Detect distinct trend segments in the price data.

    Uses a simple approach: split into chunks and classify each.
    """
    if len(candles) < min_segment_days * 2:
        return []

    df = candles.sort_values('datetime').copy()

    # Split into roughly equal segments (max 5)
    n = len(df)
    num_segments = min(5, max(2, n // min_segment_days))
    segment_size = n // num_segments

    segments = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, n - 1)

        if end_idx <= start_idx:
            continue

        segment_df = df.iloc[start_idx:end_idx + 1]
        start_price = segment_df.iloc[0]['close_price']
        end_price = segment_df.iloc[-1]['close_price']
        change = end_price - start_price

        if change > 5:
            direction = "bullish"
        elif change < -5:
            direction = "bearish"
        else:
            direction = "sideways"

        segments.append({
            "start": segment_df.iloc[0]['datetime'].strftime('%Y-%m-%d'),
            "end": segment_df.iloc[-1]['datetime'].strftime('%Y-%m-%d'),
            "direction": direction,
            "change": round(float(change), 1),
            "start_price": round(float(start_price), 1),
            "end_price": round(float(end_price), 1),
        })

    return segments


def extract_volatility(candles: pd.DataFrame) -> dict:
    """
    Extract volatility metrics.

    Args:
        candles: DataFrame with datetime, close_price, high_price, low_price

    Returns:
        Dictionary with volatility stats
    """
    if candles.empty or len(candles) < 2:
        return {}

    df = candles.sort_values('datetime').copy()

    # Daily returns
    df['return'] = df['close_price'].pct_change()
    returns = df['return'].dropna()

    # Daily range
    if 'high_price' in df.columns and 'low_price' in df.columns:
        df['daily_range'] = df['high_price'] - df['low_price']
        avg_daily_range = float(df['daily_range'].mean())
        max_daily_range = float(df['daily_range'].max())
    else:
        avg_daily_range = None
        max_daily_range = None

    # Daily price changes
    df['daily_change'] = df['close_price'].diff()

    # Find biggest moves
    if not df['daily_change'].dropna().empty:
        max_up_idx = df['daily_change'].idxmax()
        max_down_idx = df['daily_change'].idxmin()
        max_up_row = df.loc[max_up_idx]
        max_down_row = df.loc[max_down_idx]

        max_single_day_up = {
            "date": max_up_row['datetime'].strftime('%Y-%m-%d') if pd.notna(max_up_row['datetime']) else None,
            "change": round(float(max_up_row['daily_change']), 1),
        }
        max_single_day_down = {
            "date": max_down_row['datetime'].strftime('%Y-%m-%d') if pd.notna(max_down_row['datetime']) else None,
            "change": round(float(max_down_row['daily_change']), 1),
        }
    else:
        max_single_day_up = None
        max_single_day_down = None

    # Count big move days (>5% of price range or >5 points)
    big_move_threshold = max(5, df['close_price'].max() * 0.05)
    big_move_days = int((df['daily_change'].abs() > big_move_threshold).sum())

    volatility = {
        "std_dev_daily_returns": round(float(returns.std()), 4) if len(returns) > 0 else None,
        "avg_daily_range": round(avg_daily_range, 2) if avg_daily_range else None,
        "max_daily_range": round(max_daily_range, 2) if max_daily_range else None,
        "max_single_day_up": max_single_day_up,
        "max_single_day_down": max_single_day_down,
        "big_move_days": big_move_days,
        "big_move_threshold": round(big_move_threshold, 1),
    }

    # Annualized volatility (if we have enough data)
    if len(returns) > 20:
        annualized_vol = float(returns.std() * np.sqrt(252))
        volatility["annualized_volatility"] = round(annualized_vol, 3)

    return volatility


def extract_key_moments(candles: pd.DataFrame) -> list:
    """
    Detect key moments from price and volume data.

    These are data-derived events, not external news.

    Args:
        candles: DataFrame with datetime, close_price, volume

    Returns:
        List of key moment dictionaries
    """
    if candles.empty or len(candles) < 5:
        return []

    df = candles.sort_values('datetime').copy()
    moments = []

    # Threshold crossings
    thresholds = [25, 50, 75, 90]
    for threshold in thresholds:
        crossed = df[df['close_price'] >= threshold]
        if not crossed.empty:
            first_cross = crossed.iloc[0]
            moments.append({
                "date": first_cross['datetime'].strftime('%Y-%m-%d'),
                "type": "threshold_cross",
                "detail": f"Crossed {threshold}%",
                "price": float(first_cross['close_price']),
            })

    # Big moves (top 3 by absolute daily change)
    df['daily_change'] = df['close_price'].diff()
    df['abs_change'] = df['daily_change'].abs()

    top_moves = df.nlargest(3, 'abs_change')
    for _, row in top_moves.iterrows():
        if pd.notna(row['daily_change']) and abs(row['daily_change']) > 5:
            direction = "up" if row['daily_change'] > 0 else "down"
            moments.append({
                "date": row['datetime'].strftime('%Y-%m-%d'),
                "type": "big_move",
                "detail": f"{row['daily_change']:+.1f} points ({direction})",
                "price": float(row['close_price']),
            })

    # Volume spikes (if volume data exists)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        avg_volume = df['volume'].mean()
        if avg_volume > 0:
            volume_spikes = df[df['volume'] > avg_volume * 2.5]
            for _, row in volume_spikes.head(3).iterrows():
                multiplier = row['volume'] / avg_volume
                moments.append({
                    "date": row['datetime'].strftime('%Y-%m-%d'),
                    "type": "volume_spike",
                    "detail": f"{multiplier:.1f}x average volume",
                    "volume": int(row['volume']),
                })

    # Sort by date
    moments.sort(key=lambda x: x['date'])

    return moments


def extract_thresholds(candles: pd.DataFrame) -> dict:
    """
    Analyze threshold crossings and time spent above each level.

    Args:
        candles: DataFrame with datetime, close_price

    Returns:
        Dictionary with threshold analysis
    """
    if candles.empty:
        return {}

    df = candles.sort_values('datetime').copy()
    start_date = df.iloc[0]['datetime']

    thresholds = {}

    for level in [25, 50, 75, 90]:
        crossed = df[df['close_price'] >= level]

        if not crossed.empty:
            first_cross = crossed.iloc[0]
            days_from_start = (first_cross['datetime'] - start_date).days

            # Time spent above this level
            days_above = int((df['close_price'] >= level).sum())

            # Did it ever drop back below after crossing?
            first_cross_idx = df.index.get_loc(crossed.index[0])
            after_cross = df.iloc[first_cross_idx:]
            dropped_back = (after_cross['close_price'] < level).any()

            thresholds[f"crossed_{level}"] = {
                "date": first_cross['datetime'].strftime('%Y-%m-%d'),
                "days_from_start": days_from_start,
                "price_at_cross": float(first_cross['close_price']),
            }
            thresholds[f"days_above_{level}"] = days_above
            thresholds[f"dropped_back_below_{level}"] = bool(dropped_back)
        else:
            thresholds[f"crossed_{level}"] = None
            thresholds[f"days_above_{level}"] = 0
            thresholds[f"dropped_back_below_{level}"] = False

    return thresholds


def extract_volume_patterns(candles: pd.DataFrame, trades: pd.DataFrame = None) -> dict:
    """
    Extract volume-related patterns and metrics.

    Args:
        candles: DataFrame with datetime, close_price, volume
        trades: Optional DataFrame with trade-level data

    Returns:
        Dictionary with volume analysis
    """
    if candles.empty:
        return {}

    df = candles.sort_values('datetime').copy()

    patterns = {}

    if 'volume' in df.columns and df['volume'].sum() > 0:
        patterns["total"] = int(df['volume'].sum())
        patterns["avg_daily"] = round(float(df['volume'].mean()), 1)
        patterns["median_daily"] = round(float(df['volume'].median()), 1)

        # Peak day
        peak_idx = df['volume'].idxmax()
        peak_row = df.loc[peak_idx]
        patterns["peak_day"] = {
            "date": peak_row['datetime'].strftime('%Y-%m-%d'),
            "volume": int(peak_row['volume']),
        }

        # Volume trend (first half vs second half)
        n = len(df)
        first_half_vol = df.iloc[:n//2]['volume'].sum()
        second_half_vol = df.iloc[n//2:]['volume'].sum()

        if first_half_vol > 0:
            vol_ratio = second_half_vol / first_half_vol
            if vol_ratio > 1.5:
                patterns["volume_trend"] = "increasing"
            elif vol_ratio < 0.67:
                patterns["volume_trend"] = "decreasing"
            else:
                patterns["volume_trend"] = "stable"

        # VWAP (Volume-Weighted Average Price)
        if df['volume'].sum() > 0:
            vwap = (df['close_price'] * df['volume']).sum() / df['volume'].sum()
            patterns["vwap"] = round(float(vwap), 2)

        # Volume at high prices (>75)
        high_price_vol = df[df['close_price'] > 75]['volume'].sum()
        patterns["volume_at_highs_pct"] = round(float(high_price_vol / df['volume'].sum() * 100), 1)

    # Trade-level analysis if available
    if trades is not None and not trades.empty:
        patterns["total_trades"] = len(trades)
        if 'count' in trades.columns:
            patterns["avg_trade_size"] = round(float(trades['count'].mean()), 1)
            patterns["median_trade_size"] = round(float(trades['count'].median()), 1)
            patterns["max_trade_size"] = int(trades['count'].max())

        if 'taker_side' in trades.columns:
            yes_volume = trades[trades['taker_side'] == 'yes']['count'].sum() if 'count' in trades.columns else 0
            no_volume = trades[trades['taker_side'] == 'no']['count'].sum() if 'count' in trades.columns else 0
            total = yes_volume + no_volume
            if total > 0:
                patterns["yes_taker_pct"] = round(float(yes_volume / total * 100), 1)

    return patterns


def extract_relative_importance(
    market_volume: int,
    all_volumes: list[int]
) -> dict:
    """
    Calculate relative importance of a market within a collection.

    Args:
        market_volume: Volume of this market
        all_volumes: List of volumes for all markets in collection

    Returns:
        Dictionary with relative importance metrics
    """
    if not all_volumes or market_volume is None:
        return {}

    sorted_volumes = sorted(all_volumes, reverse=True)
    total_volume = sum(all_volumes)

    # Rank (1 = highest)
    rank = sorted_volumes.index(market_volume) + 1 if market_volume in sorted_volumes else len(sorted_volumes)

    # Percentile
    percentile = (len([v for v in all_volumes if v < market_volume]) / len(all_volumes)) * 100

    return {
        "volume_rank": rank,
        "volume_percentile": round(percentile, 1),
        "pct_of_total_volume": round(market_volume / total_volume * 100, 2) if total_volume > 0 else 0,
        "total_markets_in_collection": len(all_volumes),
    }
