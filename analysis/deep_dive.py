"""
Deep dive analysis for related market groups.

Auto-detects related markets (e.g., different rate brackets for the same event),
generates probability distribution charts, and computes convergence metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional

from probability_distribution import (
    plot_probability_distribution,
    plot_convergence_analysis
)


def find_related_groups(
    market_info: pd.DataFrame,
    min_markets: int = 3
) -> list[dict]:
    """
    Auto-detect related market groups by event_ticker.

    Markets with the same event_ticker are typically mutually exclusive outcomes
    (e.g., different Fed rate brackets, different inflation ranges).

    Args:
        market_info: Market info DataFrame with 'event_ticker' column
        min_markets: Minimum markets to form a group

    Returns:
        List of group dictionaries:
        [
            {
                'event_ticker': 'TERMINALRATE-23DEC31',
                'title': 'Fed Terminal Rate 2023',
                'markets': ['TICKER-1', 'TICKER-2', ...],
                'winner': 'TICKER-THAT-RESOLVED-YES' or None,
                'total_volume': 123456,
            },
            ...
        ]
    """
    if market_info.empty or 'event_ticker' not in market_info.columns:
        return []

    groups = []

    # Group by event_ticker
    for event_ticker, group_df in market_info.groupby('event_ticker'):
        if len(group_df) < min_markets:
            continue

        # Get market tickers sorted by subtitle (often contains the bracket range)
        markets = group_df.sort_values('subtitle')['ticker'].tolist()

        # Find winner (market that resolved YES)
        winner = None
        if 'result' in group_df.columns:
            yes_markets = group_df[group_df['result'] == 'yes']
            if not yes_markets.empty:
                winner = yes_markets.iloc[0]['ticker']

        # Get a descriptive title
        title = event_ticker
        if 'title' in group_df.columns and not group_df['title'].empty:
            # Try to extract common prefix from titles
            first_title = str(group_df['title'].iloc[0])
            # Remove specific bracket info if present
            for sep in [' between ', ' above ', ' below ', ' be ']:
                if sep in first_title.lower():
                    title = first_title.split(sep)[0].strip()
                    break

        # Calculate total volume
        total_volume = 0
        if 'volume' in group_df.columns:
            total_volume = int(group_df['volume'].sum())

        groups.append({
            'event_ticker': event_ticker,
            'title': title,
            'markets': markets,
            'winner': winner,
            'total_volume': total_volume,
            'market_count': len(markets),
        })

    # Sort by total volume (most traded groups first)
    groups.sort(key=lambda x: x['total_volume'], reverse=True)

    return groups


def get_market_labels(
    market_info: pd.DataFrame,
    tickers: list[str],
    winner: str = None
) -> list[str]:
    """
    Generate friendly labels for markets (e.g., '5.50-5.99%' instead of full ticker).

    Args:
        market_info: Market info DataFrame
        tickers: List of market tickers
        winner: Ticker of the winning market (to mark with star)

    Returns:
        List of labels corresponding to tickers
    """
    import re
    labels = []

    for ticker in tickers:
        market = market_info[market_info['ticker'] == ticker]
        if market.empty:
            label = ticker
        else:
            market = market.iloc[0]
            label = None

            # Try to extract from title first (most reliable)
            if 'title' in market and pd.notna(market['title']):
                title = str(market['title'])

                # Look for "between X% and Y%" pattern
                match = re.search(r'between\s+(\d+\.?\d*%?)\s+and\s+(\d+\.?\d*%?)', title, re.IGNORECASE)
                if match:
                    label = f"{match.group(1)}-{match.group(2)}"
                else:
                    # Look for "above X%" pattern
                    match = re.search(r'above\s+(\d+\.?\d*%?)', title, re.IGNORECASE)
                    if match:
                        label = f">{match.group(1)}"
                    else:
                        # Look for "below X%" pattern
                        match = re.search(r'below\s+(\d+\.?\d*%?)', title, re.IGNORECASE)
                        if match:
                            label = f"<{match.group(1)}"
                        else:
                            # Look for any percentage range pattern
                            match = re.search(r'(\d+\.?\d*%?\s*[-–to]+\s*\d+\.?\d*%?)', title)
                            if match:
                                label = match.group(1).replace('–', '-')

            # Fall back to subtitle if no label found
            if not label and 'subtitle' in market and pd.notna(market['subtitle']):
                subtitle = str(market['subtitle'])
                # Only use if it's not just ":: Highest upper bound" or similar
                if not subtitle.startswith('::') and len(subtitle) < 30:
                    label = subtitle

            # Last resort - use ticker
            if not label:
                # Try to extract from ticker (e.g., KXACPI-2025-2.8 -> 2.8)
                match = re.search(r'-(\d+\.?\d*)$', ticker)
                if match:
                    label = match.group(1)
                else:
                    label = ticker

            # Mark winner
            if ticker == winner:
                label = f"{label} (WINNER)"

        labels.append(label)

    return labels


def compute_convergence_metrics(
    candlesticks: pd.DataFrame,
    ticker: str
) -> dict:
    """
    Compute when and how a market crossed key probability thresholds.

    Args:
        candlesticks: Candlesticks DataFrame
        ticker: Market ticker to analyze

    Returns:
        Dictionary with convergence metrics:
        {
            'first_date': '2022-12-27',
            'last_date': '2024-01-07',
            'first_price': 8,
            'last_price': 99,
            'crossed_50_date': '2023-03-07' or None,
            'crossed_50_price': 50,
            'crossed_90_date': '2023-07-19' or None,
            'crossed_90_price': 90,
            'days_to_50': 70 or None,
            'days_to_90': 204 or None,
            'days_at_90_plus': 172 or None,
            'max_price': 99,
            'min_price': 3,
            'biggest_daily_move': 15,
            'biggest_move_date': '2023-06-15',
        }
    """
    df = candlesticks[candlesticks['market_ticker'] == ticker].copy()

    if df.empty:
        return {}

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    metrics = {}

    # Basic info
    metrics['first_date'] = df['datetime'].iloc[0].strftime('%Y-%m-%d')
    metrics['last_date'] = df['datetime'].iloc[-1].strftime('%Y-%m-%d')
    metrics['first_price'] = float(df['close_price'].iloc[0])
    metrics['last_price'] = float(df['close_price'].iloc[-1])
    metrics['max_price'] = float(df['close_price'].max())
    metrics['min_price'] = float(df['close_price'].min())
    metrics['total_days'] = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days

    # Find 50% crossing
    crossed_50 = df[df['close_price'] >= 50]
    if not crossed_50.empty:
        first_50 = crossed_50.iloc[0]
        metrics['crossed_50_date'] = first_50['datetime'].strftime('%Y-%m-%d')
        metrics['crossed_50_price'] = float(first_50['close_price'])
        metrics['days_to_50'] = (first_50['datetime'] - df['datetime'].iloc[0]).days
    else:
        metrics['crossed_50_date'] = None
        metrics['days_to_50'] = None

    # Find 90% crossing
    crossed_90 = df[df['close_price'] >= 90]
    if not crossed_90.empty:
        first_90 = crossed_90.iloc[0]
        metrics['crossed_90_date'] = first_90['datetime'].strftime('%Y-%m-%d')
        metrics['crossed_90_price'] = float(first_90['close_price'])
        metrics['days_to_90'] = (first_90['datetime'] - df['datetime'].iloc[0]).days
        metrics['days_at_90_plus'] = (df['datetime'].iloc[-1] - first_90['datetime']).days
    else:
        metrics['crossed_90_date'] = None
        metrics['days_to_90'] = None
        metrics['days_at_90_plus'] = None

    # Biggest daily move
    if len(df) > 1:
        df['daily_change'] = df['close_price'].diff().abs()
        max_change_idx = df['daily_change'].idxmax()
        if pd.notna(max_change_idx):
            max_change_row = df.loc[max_change_idx]
            metrics['biggest_daily_move'] = float(max_change_row['daily_change'])
            metrics['biggest_move_date'] = max_change_row['datetime'].strftime('%Y-%m-%d')

    return metrics


def generate_deep_dive(
    group: dict,
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame,
    events: list[tuple] = None
) -> dict:
    """
    Generate complete deep dive analysis for a market group.

    Args:
        group: Group dictionary from find_related_groups()
        market_info: Market info DataFrame
        candlesticks: Candlesticks DataFrame
        events: Optional list of (date_str, label) event tuples

    Returns:
        Dictionary with all deep dive content:
        {
            'event_ticker': 'TERMINALRATE-23DEC31',
            'title': 'Fed Terminal Rate 2023',
            'markets': [...],
            'winner': 'TICKER-B5.745',
            'probability_chart': 'base64...',
            'convergence_chart': 'base64...' or None,
            'convergence_metrics': {...} or None,
            'narrative': 'The market converged to...',
        }
    """
    result = {
        'event_ticker': group['event_ticker'],
        'title': group['title'],
        'markets': group['markets'],
        'winner': group['winner'],
        'total_volume': group['total_volume'],
        'market_count': group['market_count'],
    }

    # Get labels for markets
    labels = get_market_labels(market_info, group['markets'], group['winner'])

    # Generate probability distribution chart
    prob_chart = plot_probability_distribution(
        candlesticks,
        group['markets'],
        labels=labels,
        title=f"{group['title']}: Probability Distribution",
        events=events,
        return_base64=True
    )
    result['probability_chart'] = prob_chart

    # Generate convergence analysis for winner (if exists)
    if group['winner']:
        winner_info = market_info[market_info['ticker'] == group['winner']]
        outcome = 'yes'  # Winner always resolved YES

        # Find the winner label
        winner_idx = group['markets'].index(group['winner']) if group['winner'] in group['markets'] else 0
        winner_label = labels[winner_idx] if winner_idx < len(labels) else group['winner']

        conv_chart = plot_convergence_analysis(
            candlesticks,
            group['winner'],
            title=f"{group['title']}: {winner_label} Convergence",
            outcome=outcome,
            events=events,
            return_base64=True
        )
        result['convergence_chart'] = conv_chart

        # Compute metrics
        metrics = compute_convergence_metrics(candlesticks, group['winner'])
        result['convergence_metrics'] = metrics

        # Generate narrative
        result['narrative'] = generate_narrative(group, metrics, winner_label)
    else:
        result['convergence_chart'] = None
        result['convergence_metrics'] = None
        result['narrative'] = f"This market group has not yet resolved. {group['market_count']} markets are tracking different outcome brackets."

    return result


def generate_narrative(group: dict, metrics: dict, winner_label: str) -> str:
    """
    Generate a human-readable narrative about the market convergence.

    Args:
        group: Group dictionary
        metrics: Convergence metrics dictionary
        winner_label: Friendly label for the winning market

    Returns:
        Narrative string
    """
    parts = []

    # Opening
    parts.append(f"The {group['title']} market group contained {group['market_count']} competing brackets.")

    # Winner info
    parts.append(f"The winning outcome was **{winner_label}**.")

    # Price journey
    if 'first_price' in metrics and 'last_price' in metrics:
        parts.append(
            f"The winning market started at {metrics['first_price']:.0f} cents "
            f"and closed at {metrics['last_price']:.0f} cents."
        )

    # 50% crossing
    if metrics.get('crossed_50_date'):
        parts.append(
            f"It became the market favorite (crossed 50%) on {metrics['crossed_50_date']}, "
            f"{metrics['days_to_50']} days after opening."
        )

    # 90% crossing
    if metrics.get('crossed_90_date'):
        parts.append(
            f"High confidence (90%+) was reached on {metrics['crossed_90_date']}."
        )
        if metrics.get('days_at_90_plus'):
            parts.append(
                f"The market remained above 90% for {metrics['days_at_90_plus']} days before resolution."
            )

    # Early knowledge insight
    if metrics.get('days_at_90_plus') and metrics['days_at_90_plus'] > 30:
        months = metrics['days_at_90_plus'] // 30
        parts.append(
            f"**The market effectively knew the answer ~{months} month(s) before official resolution.**"
        )

    # Volume
    if group.get('total_volume'):
        parts.append(f"Total trading volume across all brackets: {group['total_volume']:,} contracts.")

    return " ".join(parts)


def find_events_for_group(event_ticker: str) -> list[tuple]:
    """
    Return known events for specific market groups.

    This could be extended to load from a config file or API.

    Args:
        event_ticker: The event ticker to find events for

    Returns:
        List of (date_str, label) tuples
    """
    # Known events for Fed rate markets
    if 'TERMINALRATE-23' in event_ticker:
        return [
            ('2023-02-01', 'Feb +25bp'),
            ('2023-03-22', 'Mar +25bp'),
            ('2023-05-03', 'May +25bp'),
            ('2023-06-14', 'Jun PAUSE'),
            ('2023-07-26', 'Jul +25bp (FINAL)'),
            ('2023-09-20', 'Sep HOLD'),
            ('2023-11-01', 'Nov HOLD'),
            ('2023-12-13', 'Dec HOLD'),
        ]

    # Add more event sets as needed
    return []


def generate_all_deep_dives(
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame,
    max_groups: int = 5,
    min_markets: int = 3
) -> list[dict]:
    """
    Generate deep dives for all detected market groups.

    Args:
        market_info: Market info DataFrame
        candlesticks: Candlesticks DataFrame
        max_groups: Maximum number of groups to analyze
        min_markets: Minimum markets per group

    Returns:
        List of deep dive dictionaries
    """
    groups = find_related_groups(market_info, min_markets=min_markets)

    # Limit to top groups by volume
    groups = groups[:max_groups]

    deep_dives = []
    for group in groups:
        # Get known events for this group
        events = find_events_for_group(group['event_ticker'])

        dive = generate_deep_dive(group, market_info, candlesticks, events)
        deep_dives.append(dive)

    return deep_dives
