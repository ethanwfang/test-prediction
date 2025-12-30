"""
Context Builder for LLM-ready market analysis.

Transforms exported market data into structured context optimized
for feeding into an LLM for generating market recaps and summaries.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from feature_extractors import (
    extract_price_journey,
    extract_trend_analysis,
    extract_volatility,
    extract_key_moments,
    extract_thresholds,
    extract_volume_patterns,
    extract_relative_importance,
)


def build_full_context(
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame,
    trades: pd.DataFrame,
    theme: str = "markets"
) -> dict:
    """
    Build complete context for all markets in an export.

    Args:
        market_info: Market metadata DataFrame
        candlesticks: OHLCV DataFrame
        trades: Trades DataFrame
        theme: Search keyword/theme used for this export

    Returns:
        Complete context dictionary ready for LLM
    """
    context = {
        "metadata": {
            "theme": theme,
            "generated_at": datetime.now().isoformat(),
            "generator": "kalshi-context-builder",
        },
        "collection": build_collection_context(market_info, candlesticks, trades),
        "market_groups": build_market_groups_context(market_info, candlesticks),
        "markets": [],
    }

    # Build per-market context
    all_volumes = market_info['volume'].tolist() if 'volume' in market_info.columns else []

    for _, market in market_info.iterrows():
        ticker = market['ticker']
        market_candles = candlesticks[candlesticks['market_ticker'] == ticker] if not candlesticks.empty else pd.DataFrame()
        market_trades = trades[trades['market_ticker'] == ticker] if not trades.empty else pd.DataFrame()

        market_context = build_market_context(
            market=market,
            candles=market_candles,
            trades=market_trades,
            all_volumes=all_volumes
        )
        context["markets"].append(market_context)

    # Sort markets by volume (most important first)
    context["markets"].sort(key=lambda x: x.get("volume", {}).get("total", 0), reverse=True)

    return context


def build_collection_context(
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame,
    trades: pd.DataFrame
) -> dict:
    """
    Build context about the entire collection of markets.

    Args:
        market_info: Market metadata DataFrame
        candlesticks: OHLCV DataFrame
        trades: Trades DataFrame

    Returns:
        Collection-level context dictionary
    """
    collection = {
        "total_markets": len(market_info),
    }

    # Volume stats
    if 'volume' in market_info.columns:
        volumes = market_info['volume'].dropna()
        collection["total_volume"] = int(volumes.sum())

        # Volume concentration
        if len(volumes) > 0:
            sorted_vols = volumes.sort_values(ascending=False)
            top_5_vol = sorted_vols.head(5).sum()
            collection["top_5_volume"] = int(top_5_vol)
            collection["top_5_pct_of_total"] = round(top_5_vol / volumes.sum() * 100, 1) if volumes.sum() > 0 else 0

            # Gini coefficient for volume concentration
            collection["volume_gini"] = round(_calculate_gini(volumes.values), 3)

    # Outcome distribution
    if 'status' in market_info.columns:
        status_counts = market_info['status'].value_counts().to_dict()
        collection["status_distribution"] = status_counts

    if 'result' in market_info.columns:
        finalized = market_info[market_info['status'] == 'finalized']
        if not finalized.empty:
            result_counts = finalized['result'].value_counts().to_dict()
            collection["outcome_distribution"] = {
                "yes": int(result_counts.get('yes', 0)),
                "no": int(result_counts.get('no', 0)),
            }

    # Date range
    if 'created_time' in market_info.columns:
        valid_dates = market_info['created_time'].dropna()
        if not valid_dates.empty:
            collection["earliest_market"] = valid_dates.min().strftime('%Y-%m-%d')
            collection["latest_market"] = valid_dates.max().strftime('%Y-%m-%d')

    # Event ticker groupings
    if 'event_ticker' in market_info.columns:
        groups = market_info.groupby('event_ticker').agg({
            'ticker': 'count',
            'volume': 'sum' if 'volume' in market_info.columns else 'count'
        }).reset_index()
        groups.columns = ['event_ticker', 'market_count', 'total_volume']
        groups = groups.sort_values('total_volume', ascending=False)

        collection["event_groups"] = [
            {
                "event_ticker": row['event_ticker'],
                "market_count": int(row['market_count']),
                "total_volume": int(row['total_volume']),
            }
            for _, row in groups.head(10).iterrows()
        ]

    # Trade stats
    if not trades.empty:
        collection["total_trades"] = len(trades)
        if 'count' in trades.columns:
            collection["total_contracts_traded"] = int(trades['count'].sum())

    return collection


def build_market_context(
    market: pd.Series,
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    all_volumes: list[int]
) -> dict:
    """
    Build comprehensive context for a single market.

    Args:
        market: Series with market metadata
        candles: DataFrame with candlestick data for this market
        trades: DataFrame with trade data for this market
        all_volumes: List of all market volumes for relative comparison

    Returns:
        Market context dictionary
    """
    context = {
        "ticker": market.get('ticker'),
        "title": market.get('title'),
        "status": market.get('status'),
    }

    # Outcome info
    result = market.get('result')
    if pd.notna(result) and result:
        context["outcome"] = str(result).upper()
    if pd.notna(market.get('last_price')):
        context["final_price"] = float(market['last_price'])

    # Event/series info
    event_ticker = market.get('event_ticker')
    if pd.notna(event_ticker) and event_ticker:
        context["event_ticker"] = event_ticker
    subtitle = market.get('subtitle')
    if pd.notna(subtitle) and subtitle:
        context["label"] = subtitle

    # Extract features from candlesticks
    if not candles.empty:
        candles = candles.sort_values('datetime').copy()

        context["price_journey"] = extract_price_journey(candles)
        context["trend"] = extract_trend_analysis(candles)
        context["volatility"] = extract_volatility(candles)
        context["key_moments"] = extract_key_moments(candles)
        context["thresholds"] = extract_thresholds(candles)
        context["volume"] = extract_volume_patterns(candles, trades)
    else:
        context["volume"] = {}

    # Volume from market info (may be more accurate than candlesticks)
    if pd.notna(market.get('volume')):
        context["volume"]["total_reported"] = int(market['volume'])

    # Relative importance
    market_vol = int(market.get('volume', 0)) if pd.notna(market.get('volume')) else 0
    context["relative_importance"] = extract_relative_importance(market_vol, all_volumes)

    return context


def build_market_groups_context(
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame
) -> list[dict]:
    """
    Build context for related market groups (same event_ticker).

    Includes correlation analysis and lead change detection.

    Args:
        market_info: Market metadata DataFrame
        candlesticks: OHLCV DataFrame

    Returns:
        List of market group context dictionaries
    """
    if 'event_ticker' not in market_info.columns:
        return []

    groups = []

    # Group by event_ticker
    for event_ticker, group_df in market_info.groupby('event_ticker'):
        if len(group_df) < 2:
            continue

        tickers = group_df['ticker'].tolist()

        group_context = {
            "event_ticker": event_ticker,
            "market_count": len(tickers),
        }

        # Extract question theme from title
        if 'title' in group_df.columns and not group_df['title'].empty:
            first_title = str(group_df['title'].iloc[0])
            # Try to get the common question part
            for sep in [' between ', ' above ', ' below ', ' be ']:
                if sep in first_title.lower():
                    group_context["question"] = first_title.split(sep)[0].strip()
                    break
            else:
                group_context["question"] = first_title[:100]

        # Volume stats
        if 'volume' in group_df.columns:
            group_context["total_volume"] = int(group_df['volume'].sum())

        # Find winner
        winner = None
        if 'result' in group_df.columns:
            yes_markets = group_df[group_df['result'] == 'yes']
            if not yes_markets.empty:
                winner = yes_markets.iloc[0]['ticker']
                group_context["winner"] = winner
                winner_label = yes_markets.iloc[0].get('subtitle')
                if pd.isna(winner_label) or not winner_label:
                    winner_label = winner
                group_context["winner_label"] = winner_label

        # Outcome distribution within group
        if 'result' in group_df.columns:
            result_counts = group_df['result'].value_counts()
            group_context["outcomes"] = {
                "yes": int(result_counts.get('yes', 0)),
                "no": int(result_counts.get('no', 0)),
            }

        # Build market list for this group
        group_context["markets"] = []
        for _, m in group_df.iterrows():
            label = m.get('subtitle')
            if pd.isna(label) or not label:
                label = m['ticker']
            market_entry = {
                "ticker": m['ticker'],
                "label": label,
                "volume": int(m['volume']) if pd.notna(m.get('volume')) else 0,
            }
            result = m.get('result')
            if pd.notna(result) and result:
                market_entry["outcome"] = str(result).upper()
            if pd.notna(m.get('last_price')):
                market_entry["final_price"] = float(m['last_price'])
            group_context["markets"].append(market_entry)

        # Sort markets in group by volume
        group_context["markets"].sort(key=lambda x: x.get('volume', 0), reverse=True)

        # Correlation and lead change analysis
        if not candlesticks.empty:
            group_candles = candlesticks[candlesticks['market_ticker'].isin(tickers)]

            if not group_candles.empty:
                # Correlation matrix
                correlations = _calculate_group_correlations(group_candles, tickers)
                if correlations:
                    group_context["correlations"] = correlations

                # Lead changes
                lead_changes = _detect_lead_changes(group_candles, tickers)
                if lead_changes:
                    group_context["lead_changes"] = lead_changes

                # When did the eventual winner take the lead (if resolved)
                if winner:
                    winner_lead_info = _analyze_winner_lead(group_candles, tickers, winner)
                    if winner_lead_info:
                        group_context["winner_analysis"] = winner_lead_info

        groups.append(group_context)

    # Sort by total volume
    groups.sort(key=lambda x: x.get('total_volume', 0), reverse=True)

    return groups


def _calculate_group_correlations(
    candles: pd.DataFrame,
    tickers: list[str]
) -> dict:
    """
    Calculate price correlations between markets in a group.

    Args:
        candles: Candlestick data for all markets in group
        tickers: List of ticker symbols

    Returns:
        Dictionary with correlation info
    """
    if len(tickers) < 2:
        return {}

    # Pivot to get prices by date
    pivot = candles.pivot_table(
        index='datetime',
        columns='market_ticker',
        values='close_price',
        aggfunc='last'
    )

    # Only include tickers that have data
    available = [t for t in tickers if t in pivot.columns]
    if len(available) < 2:
        return {}

    pivot = pivot[available].dropna()

    if len(pivot) < 10:  # Need enough data points
        return {}

    # Calculate correlation matrix
    corr_matrix = pivot.corr()

    # Find most negatively correlated pairs (expected for mutually exclusive outcomes)
    correlations = {
        "note": "Mutually exclusive markets should be negatively correlated",
    }

    # Get correlation pairs
    pairs = []
    for i, t1 in enumerate(available):
        for t2 in available[i+1:]:
            corr = corr_matrix.loc[t1, t2]
            pairs.append({
                "market_1": t1,
                "market_2": t2,
                "correlation": round(float(corr), 3)
            })

    # Sort by correlation (most negative first - these are the competing pairs)
    pairs.sort(key=lambda x: x['correlation'])

    correlations["pairs"] = pairs[:5]  # Top 5 most correlated/anti-correlated

    # Average correlation
    all_corrs = [p['correlation'] for p in pairs]
    if all_corrs:
        correlations["average_correlation"] = round(sum(all_corrs) / len(all_corrs), 3)

    return correlations


def _detect_lead_changes(
    candles: pd.DataFrame,
    tickers: list[str]
) -> dict:
    """
    Detect when the "favorite" (highest price) changed.

    Args:
        candles: Candlestick data for all markets in group
        tickers: List of ticker symbols

    Returns:
        Dictionary with lead change info
    """
    # Pivot to get prices by date
    pivot = candles.pivot_table(
        index='datetime',
        columns='market_ticker',
        values='close_price',
        aggfunc='last'
    ).sort_index()

    # Only include tickers that have data
    available = [t for t in tickers if t in pivot.columns]
    if len(available) < 2:
        return {}

    pivot = pivot[available].ffill()

    # Find leader at each point
    leaders = pivot.idxmax(axis=1)

    # Detect changes
    lead_changes = []
    prev_leader = None

    for date, leader in leaders.items():
        if leader != prev_leader and prev_leader is not None:
            lead_changes.append({
                "date": date.strftime('%Y-%m-%d'),
                "new_leader": leader,
                "previous_leader": prev_leader,
                "new_leader_price": round(float(pivot.loc[date, leader]), 1),
            })
        prev_leader = leader

    result = {
        "total_lead_changes": len(lead_changes),
    }

    if lead_changes:
        result["changes"] = lead_changes[-10:]  # Last 10 changes
        result["first_leader"] = leaders.iloc[0] if len(leaders) > 0 else None
        result["final_leader"] = leaders.iloc[-1] if len(leaders) > 0 else None

    return result


def _analyze_winner_lead(
    candles: pd.DataFrame,
    tickers: list[str],
    winner: str
) -> dict:
    """
    Analyze when the eventual winner took and held the lead.

    Args:
        candles: Candlestick data for all markets in group
        tickers: List of ticker symbols
        winner: Ticker of the winning market

    Returns:
        Dictionary with winner lead analysis
    """
    if winner not in tickers:
        return {}

    # Pivot to get prices by date
    pivot = candles.pivot_table(
        index='datetime',
        columns='market_ticker',
        values='close_price',
        aggfunc='last'
    ).sort_index()

    available = [t for t in tickers if t in pivot.columns]
    if winner not in available or len(available) < 2:
        return {}

    pivot = pivot[available].ffill()

    # Find leader at each point
    leaders = pivot.idxmax(axis=1)

    # When did winner first take the lead?
    winner_leads = leaders[leaders == winner]
    if winner_leads.empty:
        return {"winner_never_led": True}

    first_lead_date = winner_leads.index[0]

    # When did winner take *permanent* lead (never lost it again)?
    permanent_lead_date = None
    for i in range(len(leaders) - 1, -1, -1):
        if leaders.iloc[i] != winner:
            if i < len(leaders) - 1:
                permanent_lead_date = leaders.index[i + 1]
            break
    else:
        # Winner led from the beginning
        permanent_lead_date = leaders.index[0]

    # Calculate time at lead
    days_in_lead = int((leaders == winner).sum())
    total_days = len(leaders)

    # Resolution date (last date in data)
    resolution_date = pivot.index[-1]

    result = {
        "first_took_lead": first_lead_date.strftime('%Y-%m-%d'),
        "permanent_lead_from": permanent_lead_date.strftime('%Y-%m-%d') if permanent_lead_date else None,
        "days_in_lead": days_in_lead,
        "pct_time_in_lead": round(days_in_lead / total_days * 100, 1) if total_days > 0 else 0,
    }

    # Days between permanent lead and resolution
    if permanent_lead_date:
        days_known_early = (resolution_date - permanent_lead_date).days
        result["days_known_before_resolution"] = days_known_early

    return result


def _calculate_gini(values: np.ndarray) -> float:
    """Calculate Gini coefficient for measuring concentration."""
    if len(values) == 0:
        return 0

    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * values))) / (n * np.sum(values)) - (n + 1) / n

    return max(0, min(1, gini))


def _clean_for_json(obj):
    """
    Recursively clean an object for JSON serialization.
    Removes NaN, None values and converts numpy types.
    """
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj
                if item is not None and not (isinstance(item, float) and np.isnan(item))]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def export_context_json(context: dict, output_path: str) -> str:
    """
    Export context to JSON file.

    Args:
        context: Context dictionary
        output_path: Path to save JSON file

    Returns:
        Path to saved file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Clean NaN and None values
    cleaned = _clean_for_json(context)

    with open(output_path, 'w') as f:
        json.dump(cleaned, f, indent=2, default=str)

    return output_path


def export_context_compact(context: dict) -> str:
    """
    Export context as compact text for token-efficient LLM input.

    Args:
        context: Context dictionary

    Returns:
        Compact text representation
    """
    lines = []

    # Header
    lines.append(f"# Market Analysis: {context['metadata']['theme']}")
    lines.append(f"Generated: {context['metadata']['generated_at']}")
    lines.append("")

    # Collection summary
    coll = context['collection']
    lines.append("## Collection Summary")
    lines.append(f"- Markets: {coll['total_markets']}")
    lines.append(f"- Total Volume: {coll.get('total_volume', 'N/A'):,}")

    if 'outcome_distribution' in coll:
        outcomes = coll['outcome_distribution']
        lines.append(f"- Outcomes: {outcomes.get('yes', 0)} YES, {outcomes.get('no', 0)} NO")

    lines.append("")

    # Market groups
    if context.get('market_groups'):
        lines.append("## Related Market Groups")
        for group in context['market_groups'][:5]:  # Top 5
            lines.append(f"\n### {group.get('question', group['event_ticker'])}")
            lines.append(f"Markets: {group['market_count']} | Volume: {group.get('total_volume', 0):,}")

            if group.get('winner'):
                lines.append(f"Winner: {group.get('winner_label', group['winner'])}")

            if group.get('winner_analysis'):
                wa = group['winner_analysis']
                if wa.get('days_known_before_resolution'):
                    lines.append(f"Known {wa['days_known_before_resolution']} days before resolution")

            if group.get('lead_changes'):
                lc = group['lead_changes']
                lines.append(f"Lead changes: {lc['total_lead_changes']}")

        lines.append("")

    # Top markets
    lines.append("## Top Markets by Volume")
    for market in context['markets'][:10]:
        lines.append(f"\n### {market['ticker']}")
        lines.append(f"Title: {market.get('title', 'N/A')}")

        if market.get('outcome'):
            lines.append(f"Outcome: {market['outcome']}")

        if market.get('price_journey'):
            pj = market['price_journey']
            lines.append(f"Price: {pj.get('start', {}).get('price', '?')} -> {pj.get('end', {}).get('price', '?')}")

        if market.get('trend'):
            lines.append(f"Trend: {market['trend'].get('direction', 'N/A')}")

        if market.get('thresholds'):
            th = market['thresholds']
            if th.get('crossed_50'):
                lines.append(f"Crossed 50%: {th['crossed_50']['date']}")
            if th.get('crossed_90'):
                lines.append(f"Crossed 90%: {th['crossed_90']['date']}")

    return "\n".join(lines)


# CLI function
def generate_context_from_data(
    data_dir: str,
    theme: str = "markets",
    output_path: str = None
) -> dict:
    """
    Generate context from exported CSV files.

    Args:
        data_dir: Directory containing CSV files
        theme: Search keyword/theme for this export
        output_path: Optional path to save JSON output

    Returns:
        Context dictionary
    """
    from data_loader import load_all_data

    # Load data
    market_info, candlesticks, trades = load_all_data(data_dir)

    if market_info.empty:
        raise ValueError(f"No market data found in {data_dir}")

    # Build context
    context = build_full_context(market_info, candlesticks, trades, theme)

    # Save if output path provided
    if output_path:
        export_context_json(context, output_path)
        print(f"Context saved to: {output_path}")

    return context
