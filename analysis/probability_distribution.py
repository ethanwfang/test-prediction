"""
Generate probability distribution chart for related markets.
Shows how probability mass shifts between competing outcomes over time.
"""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return b64


def _save_or_return(fig, output_path: str = None, return_base64: bool = False) -> str:
    """Save figure to file, return as base64, or show interactively."""
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return output_path
    elif return_base64:
        b64 = fig_to_base64(fig)
        plt.close(fig)
        return b64
    else:
        plt.show()
        return None


def plot_probability_distribution(
    candlesticks: pd.DataFrame,
    market_tickers: list[str],
    labels: list[str] = None,
    title: str = "Probability Distribution Over Time",
    events: list[tuple] = None,
    output_path: str = None,
    return_base64: bool = False
) -> str:
    """
    Create stacked area chart showing probability distribution across related markets.

    Args:
        candlesticks: DataFrame with datetime, market_ticker, close_price
        market_tickers: List of tickers to include (should be mutually exclusive outcomes)
        labels: Optional friendly labels for each ticker
        title: Chart title
        events: List of (date_str, label) tuples to mark on chart
        output_path: Where to save the chart
        return_base64: Return base64 string instead of saving

    Returns:
        Path to saved chart, base64 string, or None
    """
    df = candlesticks[candlesticks['market_ticker'].isin(market_tickers)].copy()
    if df.empty:
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])

    # Pivot to get prices by date for each market
    pivot = df.pivot_table(
        index='datetime',
        columns='market_ticker',
        values='close_price',
        aggfunc='last'
    ).sort_index()

    # Reorder columns to match input order (only include columns that exist)
    available_tickers = [t for t in market_tickers if t in pivot.columns]
    if not available_tickers:
        return None
    pivot = pivot[available_tickers]

    # Forward fill and backward fill to handle NaN values
    pivot = pivot.ffill().bfill()

    # Fill any remaining NaNs with 0
    pivot = pivot.fillna(0)

    # Normalize to 100% (in case they don't sum perfectly)
    row_sums = pivot.sum(axis=1)
    # Avoid division by zero
    row_sums = row_sums.replace(0, 1)
    pivot_normalized = pivot.div(row_sums, axis=0) * 100

    # Update labels to only include available tickers
    if labels:
        label_map = dict(zip(market_tickers, labels))
        pivot_normalized.columns = [label_map.get(c, c) for c in pivot_normalized.columns]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette - from cool to warm
    colors = ['#1e3a5f', '#2563eb', '#16a34a', '#eab308', '#ea580c', '#dc2626', '#991b1b']
    colors = colors[:len(pivot_normalized.columns)]

    # Stacked area chart - convert to numpy arrays to avoid shape issues
    y_data = np.array([pivot_normalized[col].values for col in pivot_normalized.columns])
    ax.stackplot(
        pivot_normalized.index,
        y_data,
        labels=pivot_normalized.columns,
        colors=colors,
        alpha=0.85
    )

    # Add event markers
    if events:
        for date_str, label in events:
            event_date = pd.to_datetime(date_str)
            if pivot_normalized.index.min() <= event_date <= pivot_normalized.index.max():
                ax.axvline(x=event_date, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
                ax.text(event_date, 102, label, rotation=45, ha='left', va='bottom',
                       fontsize=8, color='#374151', fontweight='bold')

    # Formatting
    ax.set_xlim(pivot_normalized.index.min(), pivot_normalized.index.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Legend
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_leg[::-1], loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=9)

    # Grid
    ax.grid(axis='y', alpha=0.3, color='white')
    ax.set_facecolor('#f8fafc')

    plt.tight_layout()

    return _save_or_return(fig, output_path, return_base64)


def plot_convergence_analysis(
    candlesticks: pd.DataFrame,
    market_ticker: str,
    title: str = None,
    outcome: str = None,
    events: list[tuple] = None,
    output_path: str = None,
    return_base64: bool = False
) -> str:
    """
    Create detailed convergence analysis chart for a single market.

    Shows:
    - Price line with confidence bands
    - Volume bars
    - Key threshold crossings (50%, 90%)
    - Events marked

    Returns:
        Path to saved chart, base64 string, or None
    """
    df = candlesticks[candlesticks['market_ticker'] == market_ticker].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    if df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)

    # Price chart
    ax1.fill_between(df['datetime'], df['low_price'], df['high_price'],
                     alpha=0.3, color='#2563eb', label='Daily range')
    ax1.plot(df['datetime'], df['close_price'], color='#2563eb', linewidth=2, label='Close price')

    # Threshold lines
    ax1.axhline(y=50, color='#6b7280', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=90, color='#16a34a', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(df['datetime'].iloc[0], 51, '50% - Market favorite', fontsize=8, color='#6b7280')
    ax1.text(df['datetime'].iloc[0], 91, '90% - High confidence', fontsize=8, color='#16a34a')

    # Mark threshold crossings
    crossed_50 = df[df['close_price'] >= 50].head(1)
    crossed_90 = df[df['close_price'] >= 90].head(1)

    if not crossed_50.empty:
        ax1.scatter(crossed_50['datetime'], crossed_50['close_price'],
                   s=100, color='#6b7280', zorder=5, marker='o')
        ax1.annotate(f"Crossed 50%\n{crossed_50['datetime'].iloc[0].strftime('%b %d')}",
                    (crossed_50['datetime'].iloc[0], crossed_50['close_price'].iloc[0]),
                    xytext=(10, -20), textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='#6b7280'))

    if not crossed_90.empty:
        ax1.scatter(crossed_90['datetime'], crossed_90['close_price'],
                   s=100, color='#16a34a', zorder=5, marker='o')
        ax1.annotate(f"Crossed 90%\n{crossed_90['datetime'].iloc[0].strftime('%b %d')}",
                    (crossed_90['datetime'].iloc[0], crossed_90['close_price'].iloc[0]),
                    xytext=(10, 20), textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='#16a34a'))

    # Event markers
    if events:
        for date_str, label in events:
            event_date = pd.to_datetime(date_str)
            if df['datetime'].min() <= event_date <= df['datetime'].max():
                ax1.axvline(x=event_date, color='#dc2626', linestyle='-', alpha=0.3, linewidth=2)
                ax1.text(event_date, 5, label, rotation=90, ha='right', va='bottom',
                        fontsize=7, color='#dc2626', alpha=0.8)

    # Outcome marker
    if outcome:
        outcome_y = 95 if outcome.lower() == 'yes' else 5
        outcome_color = '#16a34a' if outcome.lower() == 'yes' else '#dc2626'
        ax1.axhline(y=outcome_y, color=outcome_color, linestyle='-', alpha=0.3, linewidth=3)
        ax1.text(df['datetime'].iloc[-1], outcome_y, f' → {outcome.upper()}',
                color=outcome_color, fontweight='bold', va='center', fontsize=10)

    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Price (cents = implied probability %)', fontsize=11)
    ax1.set_title(title or market_ticker, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Volume chart
    ax2.bar(df['datetime'], df['volume'], color='#6b7280', alpha=0.7, width=2)
    ax2.set_ylabel('Volume', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return _save_or_return(fig, output_path, return_base64)


if __name__ == '__main__':
    # Demo with Fed rate data
    df = pd.read_csv('src/data/markets_candlesticks_20251224_004134.csv')

    # Fed rate brackets (from lowest to highest)
    fed_tickers = [
        'TERMINALRATE-23DEC31-B4.870',  # 4.75-4.99%
        'TERMINALRATE-23DEC31-B5.120',  # 5.00-5.24%
        'TERMINALRATE-23DEC31-B5.370',  # 5.25-5.49%
        'TERMINALRATE-23DEC31-B5.745',  # 5.50-5.99% (WINNER)
        'TERMINALRATE-23DEC31-B6.120',  # 6.00-6.24%
    ]

    fed_labels = [
        '4.75-4.99%',
        '5.00-5.24%',
        '5.25-5.49%',
        '5.50-5.99% ★',  # Winner
        '6.00-6.24%',
    ]

    # Key FOMC events
    fomc_events = [
        ('2023-02-01', 'Feb +25bp'),
        ('2023-03-22', 'Mar +25bp'),
        ('2023-05-03', 'May +25bp'),
        ('2023-06-14', 'Jun PAUSE'),
        ('2023-07-26', 'Jul +25bp (FINAL)'),
        ('2023-09-20', 'Sep HOLD'),
        ('2023-11-01', 'Nov HOLD'),
        ('2023-12-13', 'Dec HOLD'),
    ]

    print("Generating probability distribution chart...")
    plot_probability_distribution(
        df,
        fed_tickers,
        fed_labels,
        title="2023 Fed Terminal Rate Predictions: Probability Distribution",
        events=fomc_events,
        output_path='outputs/charts/fed_probability_distribution.png'
    )
    print("  Saved: outputs/charts/fed_probability_distribution.png")

    print("\nGenerating convergence analysis chart...")
    plot_convergence_analysis(
        df,
        'TERMINALRATE-23DEC31-B5.745',
        title='2023 Fed Rate 5.50-5.99%: Convergence to YES',
        outcome='yes',
        events=fomc_events,
        output_path='outputs/charts/fed_convergence_analysis.png'
    )
    print("  Saved: outputs/charts/fed_convergence_analysis.png")
