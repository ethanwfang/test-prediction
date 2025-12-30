#!/usr/bin/env python3
"""
Kalshi Market Analysis CLI

Generates comprehensive analysis reports from Kalshi market data.

Usage:
    # Generate self-contained HTML report (default)
    python analyze.py --data-dir ./src/data --output ./report.html

    # Generate markdown report (legacy)
    python analyze.py --data-dir ./src/data --format md --output-dir ./outputs

    # Skip deep dive analysis
    python analyze.py --data-dir ./src/data --output ./report.html --no-deep-dive
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data
from charts import generate_all_charts
from report import generate_html_report, generate_markdown_report
from stats import market_summary, outcome_analysis, trading_patterns
from deep_dive import generate_all_deep_dives


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Kalshi prediction market data and generate reports.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --data-dir ./src/data --output ./report.html
  python analyze.py --data-dir ./src/data --format md --output-dir ./outputs
  python analyze.py --data-dir ./src/data --output ./report.html --no-deep-dive
        """
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing CSV data files'
    )
    parser.add_argument(
        '--output',
        help='Output file path for HTML report (e.g., ./report.html)'
    )
    parser.add_argument(
        '--output-dir',
        default='./outputs',
        help='Directory to save charts and markdown reports (default: ./outputs)'
    )
    parser.add_argument(
        '--format',
        choices=['html', 'md'],
        default='html',
        help='Output format: html (self-contained) or md (markdown + PNGs)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top markets to analyze in detail (default: 5)'
    )
    parser.add_argument(
        '--no-deep-dive',
        action='store_true',
        help='Skip auto deep-dive analysis of related market groups'
    )
    parser.add_argument(
        '--max-groups',
        type=int,
        default=5,
        help='Maximum number of market groups for deep dive (default: 5)'
    )
    parser.add_argument(
        '--min-group-size',
        type=int,
        default=3,
        help='Minimum markets per group for deep dive (default: 3)'
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine output path
    if args.format == 'html':
        if args.output:
            output_path = Path(args.output)
        else:
            report_date = datetime.now().strftime("%Y%m%d")
            output_path = Path(args.output_dir) / f'report_{report_date}.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Kalshi Market Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    market_info, candlesticks, trades = load_all_data(str(data_dir))

    if market_info.empty:
        print("Error: No market data found.")
        sys.exit(1)

    # Display summary stats
    print("\nCalculating statistics...")
    summary = market_summary(market_info)
    outcomes = outcome_analysis(market_info)
    patterns = trading_patterns(trades)

    print(f"\n  Markets: {summary.get('total_markets', 0):,}")
    print(f"  Total Volume: {summary.get('total_volume', 0):,} contracts")

    if 'by_status' in summary:
        status_str = ", ".join(f"{v} {k}" for k, v in summary['by_status'].items())
        print(f"  Status: {status_str}")

    if outcomes.get('finalized_count', 0) > 0:
        print(f"  Outcomes: {outcomes.get('resolved_yes', 0)} YES, {outcomes.get('resolved_no', 0)} NO")

    # Generate deep dives (for HTML format)
    deep_dives = []
    if args.format == 'html' and not args.no_deep_dive:
        print("\nFinding related market groups...")
        deep_dives = generate_all_deep_dives(
            market_info,
            candlesticks,
            max_groups=args.max_groups,
            min_markets=args.min_group_size
        )
        if deep_dives:
            print(f"  Found {len(deep_dives)} groups for deep dive analysis:")
            for dive in deep_dives:
                winner_str = f", winner: {dive['winner']}" if dive['winner'] else ""
                print(f"    - {dive['event_ticker']}: {dive['market_count']} markets{winner_str}")
        else:
            print("  No related market groups found (need 3+ markets with same event_ticker)")

    # Generate report based on format
    if args.format == 'html':
        print("\nGenerating HTML report...")
        report_path = generate_html_report(
            market_info,
            candlesticks,
            trades,
            deep_dives=deep_dives,
            output_path=str(output_path),
            top_n=args.top_n
        )
        print(f"  Report saved: {report_path}")

        # Calculate file size
        size_bytes = output_path.stat().st_size
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.1f} KB"
        print(f"  File size: {size_str}")

    else:
        # Legacy markdown format
        print("\nGenerating charts...")
        charts = generate_all_charts(
            market_info, candlesticks,
            str(args.output_dir), top_n=args.top_n
        )
        print(f"  Generated {len(charts)} charts")

        print("\nGenerating markdown report...")
        report_path = generate_markdown_report(
            market_info, candlesticks, trades,
            str(args.output_dir), top_n=args.top_n
        )
        print(f"  Report saved: {report_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)

    if 'top_by_volume' in summary and summary['top_by_volume']:
        top = summary['top_by_volume'][0]
        print(f"\n  Top market: {top['ticker']}")
        print(f"    Volume: {top['volume']:,} contracts")
        if 'result' in top and top['result']:
            print(f"    Outcome: {top['result'].upper()}")

    if args.format == 'html':
        print(f"\n  Output: {output_path.absolute()}")
    else:
        print(f"\n  Output directory: {Path(args.output_dir).absolute()}")
        print(f"  Report: {report_path}")

    if deep_dives:
        print(f"\n  Deep dives: {len(deep_dives)} market groups analyzed")
        for dive in deep_dives:
            metrics = dive.get('convergence_metrics') or {}
            if metrics.get('days_at_90_plus'):
                days = metrics['days_at_90_plus']
                print(f"    - {dive['title']}: converged {days} days before resolution")

    print("")


if __name__ == '__main__':
    main()
