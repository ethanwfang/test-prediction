"""
Chart generation for Kalshi market analysis.

All chart functions can return either:
- A file path (when output_path is provided)
- A base64 string (when return_base64=True)
- None (when showing interactively)
"""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return b64


def _save_or_return(fig, output_path: str = None, return_base64: bool = False) -> str:
    """
    Save figure to file, return as base64, or show interactively.

    Args:
        fig: Matplotlib figure
        output_path: Path to save file (takes precedence)
        return_base64: Return base64 string if True

    Returns:
        File path, base64 string, or None
    """
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


def plot_price_history(
    candles: pd.DataFrame,
    ticker: str,
    title: str = None,
    outcome: str = None,
    output_path: str = None,
    return_base64: bool = False
) -> str:
    """
    Plot price history for a single market.

    Args:
        candles: Candlesticks DataFrame
        ticker: Market ticker to plot
        title: Optional title override
        outcome: Market outcome ('yes', 'no', or None)
        output_path: Path to save the chart
        return_base64: Return base64 string instead of saving

    Returns:
        Path to saved chart, base64 string, or None
    """
    market_data = candles[candles['market_ticker'] == ticker].copy()
    if market_data.empty:
        return None

    market_data = market_data.sort_values('datetime')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot close price
    ax1.plot(market_data['datetime'], market_data['close_price'],
             color='#2563eb', linewidth=2, label='Price (cents)')
    ax1.fill_between(market_data['datetime'], market_data['low_price'],
                     market_data['high_price'], alpha=0.2, color='#2563eb')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (cents)', color='#2563eb')
    ax1.tick_params(axis='y', labelcolor='#2563eb')
    ax1.set_ylim(0, 100)

    # Add outcome marker
    if outcome:
        outcome_color = '#16a34a' if outcome == 'yes' else '#dc2626'
        outcome_y = 95 if outcome == 'yes' else 5
        ax1.axhline(y=outcome_y, color=outcome_color, linestyle='--', alpha=0.5)
        ax1.text(market_data['datetime'].iloc[-1], outcome_y,
                 f' Resolved: {outcome.upper()}', color=outcome_color,
                 fontweight='bold', va='center')

    # Plot volume on secondary axis
    if 'volume' in market_data.columns and market_data['volume'].sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(market_data['datetime'], market_data['volume'],
                alpha=0.3, color='#6b7280', width=1, label='Volume')
        ax2.set_ylabel('Volume', color='#6b7280')
        ax2.tick_params(axis='y', labelcolor='#6b7280')

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Title
    chart_title = title or ticker
    if len(chart_title) > 60:
        chart_title = chart_title[:57] + '...'
    plt.title(chart_title, fontweight='bold', pad=10)

    plt.tight_layout()

    return _save_or_return(fig, output_path, return_base64)


def plot_volume_ranking(
    market_info: pd.DataFrame,
    top_n: int = 10,
    output_path: str = None,
    return_base64: bool = False
) -> str:
    """
    Create horizontal bar chart of top markets by volume.

    Args:
        market_info: Market info DataFrame
        top_n: Number of top markets to show
        output_path: Path to save the chart
        return_base64: Return base64 string instead of saving

    Returns:
        Path to saved chart, base64 string, or None
    """
    if market_info.empty or 'volume' not in market_info.columns:
        return None

    top = market_info.nlargest(top_n, 'volume').copy()
    top = top.sort_values('volume', ascending=True)  # For horizontal bar

    # Truncate titles
    top['short_title'] = top['title'].apply(
        lambda x: x[:40] + '...' if len(str(x)) > 40 else x
    )

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))

    # Color by outcome if available
    colors = []
    for _, row in top.iterrows():
        if 'result' in row and row['result'] == 'yes':
            colors.append('#16a34a')
        elif 'result' in row and row['result'] == 'no':
            colors.append('#dc2626')
        else:
            colors.append('#2563eb')

    bars = ax.barh(top['short_title'], top['volume'], color=colors)

    # Add value labels
    for bar, vol in zip(bars, top['volume']):
        ax.text(bar.get_width() + max(top['volume']) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{vol:,.0f}', va='center', fontsize=9)

    ax.set_xlabel('Volume (contracts)')
    ax.set_title(f'Top {top_n} Markets by Volume', fontweight='bold')
    ax.set_xlim(0, max(top['volume']) * 1.15)

    plt.tight_layout()

    return _save_or_return(fig, output_path, return_base64)


def plot_outcome_distribution(
    market_info: pd.DataFrame,
    output_path: str = None,
    return_base64: bool = False
) -> str:
    """
    Create pie chart of market outcomes.

    Args:
        market_info: Market info DataFrame
        output_path: Path to save the chart
        return_base64: Return base64 string instead of saving

    Returns:
        Path to saved chart, base64 string, or None
    """
    if market_info.empty or 'result' not in market_info.columns:
        return None

    finalized = market_info[market_info['status'] == 'finalized']
    if finalized.empty:
        return None

    counts = finalized['result'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'yes': '#16a34a', 'no': '#dc2626'}
    pie_colors = [colors.get(x, '#6b7280') for x in counts.index]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[f'{k.upper()}\n({v} markets)' for k, v in counts.items()],
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(counts)
    )

    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax.set_title('Market Outcome Distribution', fontweight='bold', fontsize=14)

    plt.tight_layout()

    return _save_or_return(fig, output_path, return_base64)


def plot_calibration(
    market_info: pd.DataFrame,
    output_path: str = None,
    return_base64: bool = False
) -> str:
    """
    Plot calibration: final price vs actual outcome.

    Shows how well prices predicted outcomes.

    Args:
        market_info: Market info DataFrame
        output_path: Path to save the chart
        return_base64: Return base64 string instead of saving

    Returns:
        Path to saved chart, base64 string, or None
    """
    if market_info.empty:
        return None

    finalized = market_info[market_info['status'] == 'finalized'].copy()
    if finalized.empty or 'result' not in finalized.columns or 'last_price' not in finalized.columns:
        return None

    # Convert result to binary
    finalized['outcome'] = (finalized['result'] == 'yes').astype(int)

    # Bin prices and calculate actual outcome rates
    finalized['price_bin'] = pd.cut(finalized['last_price'],
                                     bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     labels=['0-10', '10-20', '20-30', '30-40', '40-50',
                                            '50-60', '60-70', '70-80', '80-90', '90-100'])

    calibration = finalized.groupby('price_bin').agg({
        'outcome': ['mean', 'count']
    }).reset_index()
    calibration.columns = ['price_bin', 'actual_rate', 'count']
    calibration = calibration[calibration['count'] > 0]

    if calibration.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Ideal calibration line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect calibration')

    # Actual calibration points
    bin_centers = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    x_vals = [bin_centers[i] for i, _ in enumerate(calibration['price_bin'])]
    y_vals = calibration['actual_rate'] * 100

    ax.scatter(x_vals, y_vals, s=calibration['count'] * 10,
               c='#2563eb', alpha=0.7, edgecolors='white', linewidth=1)

    # Add count labels
    for x, y, count in zip(x_vals, y_vals, calibration['count']):
        ax.annotate(f'n={count}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('Final Price (cents = implied probability %)')
    ax.set_ylabel('Actual YES outcome rate (%)')
    ax.set_title('Market Calibration: Price vs Actual Outcome', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()

    return _save_or_return(fig, output_path, return_base64)


def generate_all_charts(
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame,
    output_dir: str,
    top_n: int = 10
) -> list[str]:
    """
    Generate all charts and save to output directory.

    Args:
        market_info: Market info DataFrame
        candlesticks: Candlesticks DataFrame
        output_dir: Directory to save charts
        top_n: Number of top markets to chart individually

    Returns:
        List of paths to generated charts
    """
    charts_dir = Path(output_dir) / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Volume ranking
    path = plot_volume_ranking(market_info, top_n=top_n,
                               output_path=str(charts_dir / 'volume_ranking.png'))
    if path:
        generated.append(path)
        print(f"  - Volume ranking chart saved")

    # Outcome distribution
    path = plot_outcome_distribution(market_info,
                                     output_path=str(charts_dir / 'outcomes.png'))
    if path:
        generated.append(path)
        print(f"  - Outcome distribution chart saved")

    # Calibration
    path = plot_calibration(market_info,
                           output_path=str(charts_dir / 'calibration.png'))
    if path:
        generated.append(path)
        print(f"  - Calibration chart saved")

    # Individual price charts for top markets
    if not market_info.empty and not candlesticks.empty:
        top_markets = market_info.nlargest(top_n, 'volume')
        for _, market in top_markets.iterrows():
            ticker = market['ticker']
            title = market.get('title', ticker)
            outcome = market.get('result', None)

            path = plot_price_history(
                candlesticks, ticker, title, outcome,
                output_path=str(charts_dir / f'{ticker}.png')
            )
            if path:
                generated.append(path)

        print(f"  - Price charts for top {len(top_markets)} markets saved")

    return generated


def generate_charts_base64(
    market_info: pd.DataFrame,
    candlesticks: pd.DataFrame,
    top_n: int = 5
) -> dict:
    """
    Generate all standard charts as base64 strings for HTML embedding.

    Args:
        market_info: Market info DataFrame
        candlesticks: Candlesticks DataFrame
        top_n: Number of top markets to chart individually

    Returns:
        Dictionary with chart names as keys and base64 strings as values
    """
    charts = {}

    # Volume ranking
    b64 = plot_volume_ranking(market_info, top_n=top_n, return_base64=True)
    if b64:
        charts['volume_ranking'] = b64

    # Outcome distribution
    b64 = plot_outcome_distribution(market_info, return_base64=True)
    if b64:
        charts['outcomes'] = b64

    # Calibration
    b64 = plot_calibration(market_info, return_base64=True)
    if b64:
        charts['calibration'] = b64

    # Individual price charts for top markets
    if not market_info.empty and not candlesticks.empty:
        top_markets = market_info.nlargest(top_n, 'volume')
        for _, market in top_markets.iterrows():
            ticker = market['ticker']
            title = market.get('title', ticker)
            outcome = market.get('result', None)

            b64 = plot_price_history(
                candlesticks, ticker, title, outcome,
                return_base64=True
            )
            if b64:
                charts[f'price_{ticker}'] = b64

    return charts
