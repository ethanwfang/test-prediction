"""
Interactive CLI for fetching Kalshi prediction market data.
"""

import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from kalshi_client import KalshiClient
from market_fetcher import MarketFetcher
from data_exporter import DataExporter

# Load model once at startup (lazy loading)
_model = None


def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        print("Loading semantic model (first time only)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def search_semantic(items: list[dict], query: str, top_k: int = 20, threshold: float = 0.25) -> list[dict]:
    """
    Search items using semantic similarity with transformer embeddings.

    Works for both series and markets.

    Args:
        items: List of dictionaries with 'title' and 'ticker' keys
        query: Search query
        top_k: Number of top matches to return
        threshold: Minimum similarity score (0-1) to include

    Returns:
        List of matching items sorted by semantic similarity
    """
    if not items or not query:
        return []

    model = get_embedding_model()

    # Build texts from titles and tickers
    texts = [f"{s.get('title', '')} {s.get('ticker', '')}" for s in items]

    # Encode query and all titles
    query_embedding = model.encode([query], convert_to_numpy=True)
    title_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Compute cosine similarities
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    title_norms = title_embeddings / np.linalg.norm(title_embeddings, axis=1, keepdims=True)

    similarities = np.dot(title_norms, query_norm.T).flatten()

    # Get top results above threshold
    sorted_indices = similarities.argsort()[::-1]

    results = []
    for idx in sorted_indices:
        score = similarities[idx]
        if score >= threshold and len(results) < top_k:
            item = items[idx].copy()
            item['_score'] = float(score)
            results.append(item)

    return results


def search_series_semantic(series_list: list[dict], query: str, top_k: int = 10, threshold: float = 0.3) -> list[dict]:
    """
    Search series using semantic similarity with transformer embeddings.

    Args:
        series_list: List of series dictionaries
        query: Search query
        top_k: Number of top matches to return
        threshold: Minimum similarity score (0-1) to include

    Returns:
        List of matching series sorted by semantic similarity
    """
    if not series_list or not query:
        return []

    model = get_embedding_model()

    # Build texts from series titles and tickers
    texts = [f"{s.get('title', '')} {s.get('ticker', '')}" for s in series_list]

    # Encode query and all titles
    query_embedding = model.encode([query], convert_to_numpy=True)
    title_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Compute cosine similarities
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    title_norms = title_embeddings / np.linalg.norm(title_embeddings, axis=1, keepdims=True)

    # Cosine similarity
    similarities = np.dot(title_norms, query_norm.T).flatten()

    # Get top results above threshold
    sorted_indices = similarities.argsort()[::-1]

    results = []
    for idx in sorted_indices:
        score = similarities[idx]
        if score >= threshold and len(results) < top_k:
            series = series_list[idx].copy()
            series['_score'] = float(score)
            results.append(series)

    return results


def print_header():
    """Print the application header."""
    print("\n" + "=" * 50)
    print("  Kalshi Market Data Fetcher")
    print("=" * 50 + "\n")


def get_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    value = input(prompt).strip()
    return value if value else default


def select_option(prompt: str, options: list[str]) -> int:
    """
    Display options and get user selection.

    Returns:
        0-based index of selected option
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")

    while True:
        try:
            choice = int(input("\nChoice: "))
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}")


def display_markets(markets: list[dict]):
    """Display a list of markets."""
    if not markets:
        print("\nNo markets found.")
        return

    print(f"\nFound {len(markets)} market(s):\n")
    for i, market in enumerate(markets, 1):
        ticker = market.get("ticker", "N/A")
        title = market.get("title", "N/A")
        status = market.get("status", "N/A")
        yes_price = market.get("yes_price", "N/A")
        if yes_price != "N/A":
            yes_price = f"{yes_price}c"
        print(f"  {i}. [{ticker}] {title}")
        print(f"      Status: {status} | Yes Price: {yes_price}")


def search_series_by_keywords(series_list: list[dict], query: str, top_k: int = 10) -> list[dict]:
    """
    Search series using substring matching.

    Matches if any query word appears in the title or ticker (partial match allowed).

    Args:
        series_list: List of series dictionaries
        query: Search query (keywords)
        top_k: Number of top matches to return

    Returns:
        List of matching series sorted by match quality
    """
    if not series_list or not query:
        return []

    query_words = query.lower().split()
    results = []

    for series in series_list:
        title = series.get("title", "").lower()
        ticker = series.get("ticker", "").lower()
        search_text = f"{title} {ticker}"

        # Count how many query words match (substring match)
        matches = 0
        for word in query_words:
            if word in search_text:
                matches += 1
            # Also check if search_text contains word as start of a word
            # e.g., "fed" matches "federal"
            elif any(w.startswith(word) or word.startswith(w) for w in search_text.split()):
                matches += 0.5

        if matches > 0:
            series_copy = series.copy()
            series_copy['_score'] = matches / len(query_words)
            results.append(series_copy)

    # Sort by score descending
    results.sort(key=lambda x: x['_score'], reverse=True)

    return results[:top_k]


def browse_by_category(fetcher) -> list[dict]:
    """Browse markets by category and series."""
    print("\nFetching categories...")
    try:
        series_list = fetcher.get_series_list()
    except Exception as e:
        print(f"Error fetching series: {e}")
        return []

    # Group series by category
    categories = {}
    for series in series_list:
        cat = series.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(series)

    if not categories:
        print("No categories found.")
        return []

    # Display categories
    cat_list = sorted(categories.keys())
    print(f"\nFound {len(cat_list)} categories:\n")
    for i, cat in enumerate(cat_list, 1):
        print(f"  [{i}] {cat} ({len(categories[cat])} series)")

    # Select category
    cat_input = get_input("\nEnter category number")
    if not cat_input:
        return []

    try:
        cat_idx = int(cat_input) - 1
        if not (0 <= cat_idx < len(cat_list)):
            print("Invalid selection.")
            return []
    except ValueError:
        print("Invalid selection.")
        return []

    selected_cat = cat_list[cat_idx]
    series_in_cat = categories[selected_cat]

    # Choose how to find series
    print(f"\n'{selected_cat}' has {len(series_in_cat)} series.")
    mode_idx = select_option(
        "How would you like to find series?",
        ["Search by keywords (semantic matching)", "Browse all series", "Get all markets in category"]
    )

    if mode_idx == 2:
        # Fetch all markets in category
        return fetch_all_markets_in_category(fetcher, series_in_cat)

    if mode_idx == 0:
        # Search by keywords - fetch markets from ALL matching series
        query = get_input("\nEnter keywords to search series")
        if not query:
            return []

        print(f"\nSearching for '{query}'...")
        matching_series = search_series_semantic(series_in_cat, query, top_k=20, threshold=0.25)

        if not matching_series:
            print("No matching series found.")
            return []

        print(f"\nFound {len(matching_series)} matching series (threshold: 25%)")
        print(f"\nFetching markets from all matching series...\n")

        # Fetch markets from ALL matching series
        all_markets = []
        for i, series in enumerate(matching_series, 1):
            series_ticker = series.get("ticker")
            title = series.get("title", "N/A")
            score = series.get("_score", 0)

            try:
                markets_found = 0
                for status in ["open", "closed", "settled"]:
                    status_markets = fetcher.get_markets_by_series(series_ticker, status=status)
                    if status_markets:
                        all_markets.extend(status_markets)
                        markets_found += len(status_markets)
                        break  # Found markets, move to next series
                    time.sleep(0.2)  # Rate limit

                print(f"  [{i}/{len(matching_series)}] {title} ({series_ticker}) - {score:.0%} match - {markets_found} markets")

            except Exception as e:
                if "429" in str(e):
                    print(f"  [{i}/{len(matching_series)}] {title} - rate limited, waiting...")
                    time.sleep(2)
                else:
                    print(f"  [{i}/{len(matching_series)}] {title} - error: {e}")

            time.sleep(0.3)  # Rate limit between series

        print(f"\nFound {len(all_markets)} total markets from {len(matching_series)} series.")
        return all_markets

    else:
        # Browse all series - keep existing behavior (select one series)
        print(f"\nSeries in '{selected_cat}':\n")
        for i, series in enumerate(series_in_cat, 1):
            ticker = series.get("ticker", "N/A")
            title = series.get("title", "N/A")
            print(f"  [{i}] {title} ({ticker})")

        # Select series
        series_input = get_input("\nEnter series number")
        if not series_input:
            return []

        try:
            series_idx = int(series_input) - 1
            if not (0 <= series_idx < len(series_in_cat)):
                print("Invalid selection.")
                return []
        except ValueError:
            print("Invalid selection.")
            return []

        selected_series = series_in_cat[series_idx]
        series_ticker = selected_series.get("ticker")

        # Fetch markets for this series (try all statuses)
        print(f"\nFetching markets for {series_ticker}...")
        try:
            markets = []
            for status in ["unopened", "open", "closed", "settled"]:
                status_markets = fetcher.get_markets_by_series(series_ticker, status=status)
                if status_markets:
                    print(f"  Found {len(status_markets)} {status} markets")
                markets.extend(status_markets)
                if len(markets) >= 50:
                    break
                time.sleep(0.3)
            markets = markets[:50]

            if not markets:
                print(f"  No markets found for series {series_ticker} in any status.")
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

        return markets


def fetch_all_markets_in_category(fetcher, series_list: list[dict]) -> list[dict]:
    """Fetch all markets from all series in a category."""
    all_markets = []
    total_series = len(series_list)
    consecutive_errors = 0

    print(f"\nFetching markets from {total_series} series...")
    print("(This may take a few minutes due to API rate limits)\n")

    for i, series in enumerate(series_list, 1):
        series_ticker = series.get("ticker")
        if not series_ticker:
            continue

        # Retry logic with backoff
        for attempt in range(3):
            try:
                # Fetch open markets first, then closed if needed
                for status in ["open", "closed", "settled"]:
                    markets = fetcher.get_markets_by_series(series_ticker, status=status)
                    all_markets.extend(markets)
                    if markets:  # Found markets, move to next series
                        break
                    time.sleep(0.5)  # Delay between status checks

                consecutive_errors = 0
                break  # Success, exit retry loop

            except Exception as e:
                if "429" in str(e):
                    consecutive_errors += 1
                    wait_time = 2 ** attempt * 2  # 2, 4, 8 seconds
                    if attempt < 2:
                        print(f"  Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Skip this series after 3 attempts
                        pass
                else:
                    print(f"  Error fetching {series_ticker}: {e}")
                    break

        # Progress indicator
        if i % 20 == 0 or i == total_series:
            print(f"  Processed {i}/{total_series} series... ({len(all_markets)} markets found)")

        # Base delay between series
        time.sleep(0.5)

        # Extra delay if we're hitting rate limits
        if consecutive_errors >= 2:
            print("  Slowing down due to rate limits...")
            time.sleep(3)
            consecutive_errors = 0

    print(f"\nFound {len(all_markets)} total markets in this category.")
    return all_markets


def export_multiple_markets(fetcher, exporter, markets: list, data_types: list, keyword: str):
    """
    Export data from multiple markets to aggregated CSVs.

    Args:
        fetcher: MarketFetcher instance
        exporter: DataExporter instance
        markets: List of market dictionaries
        data_types: List of data types to export ('info', 'candlesticks', 'trades')
        keyword: Keyword used for filename prefix
    """
    from datetime import datetime
    import pandas as pd

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported_files = []
    keyword_clean = keyword.replace(" ", "_").lower() if keyword else "markets"

    print(f"\nExporting data from {len(markets)} markets...\n")

    # Market info - one row per market
    if 'info' in data_types:
        print("Exporting market info...")
        df = pd.DataFrame(markets)
        filepath = f"data/{keyword_clean}_markets_info_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        exported_files.append((filepath, len(markets), "markets"))
        print(f"  Saved {len(markets)} markets to {filepath}")

    # Candlesticks - combine all
    if 'candlesticks' in data_types:
        print("\nFetching candlesticks for each market...")
        all_candlesticks = []

        for i, market in enumerate(markets, 1):
            ticker = market.get("ticker")
            series_ticker = market.get("series_ticker")

            if not series_ticker:
                print(f"  [{i}/{len(markets)}] {ticker} - skipping (no series ticker)")
                continue

            try:
                candles = fetcher.get_candlesticks(
                    ticker=ticker,
                    series_ticker=series_ticker,
                    period_interval=1440,  # Daily
                )
                if candles:
                    # Add market ticker to each row
                    for c in candles:
                        c['market_ticker'] = ticker
                        c['market_title'] = market.get('title', '')
                    all_candlesticks.extend(candles)
                    print(f"  [{i}/{len(markets)}] {ticker} - {len(candles)} candlesticks")
                else:
                    print(f"  [{i}/{len(markets)}] {ticker} - no data")
            except Exception as e:
                if "429" in str(e):
                    print(f"  [{i}/{len(markets)}] {ticker} - rate limited, waiting...")
                    time.sleep(2)
                else:
                    print(f"  [{i}/{len(markets)}] {ticker} - error: {e}")

            time.sleep(0.3)  # Rate limit

        if all_candlesticks:
            df = pd.DataFrame(all_candlesticks)
            if 'end_period_ts' in df.columns:
                df['datetime'] = pd.to_datetime(df['end_period_ts'], unit='s')
            filepath = f"data/{keyword_clean}_candlesticks_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            exported_files.append((filepath, len(all_candlesticks), "candlesticks"))
            print(f"\n  Saved {len(all_candlesticks)} candlesticks to {filepath}")

    # Trades - combine all
    if 'trades' in data_types:
        print("\nFetching trades for each market...")
        all_trades = []

        for i, market in enumerate(markets, 1):
            ticker = market.get("ticker")

            try:
                trades = fetcher.get_all_trades(ticker=ticker, max_trades=1000)
                if trades:
                    # Add market info to each trade
                    for t in trades:
                        t['market_ticker'] = ticker
                        t['market_title'] = market.get('title', '')
                    all_trades.extend(trades)
                    print(f"  [{i}/{len(markets)}] {ticker} - {len(trades)} trades")
                else:
                    print(f"  [{i}/{len(markets)}] {ticker} - no trades")
            except Exception as e:
                if "429" in str(e):
                    print(f"  [{i}/{len(markets)}] {ticker} - rate limited, waiting...")
                    time.sleep(2)
                else:
                    print(f"  [{i}/{len(markets)}] {ticker} - error: {e}")

            time.sleep(0.3)  # Rate limit

        if all_trades:
            df = pd.DataFrame(all_trades)
            if 'created_time' in df.columns:
                df['datetime'] = pd.to_datetime(df['created_time'])
            filepath = f"data/{keyword_clean}_trades_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            exported_files.append((filepath, len(all_trades), "trades"))
            print(f"\n  Saved {len(all_trades)} trades to {filepath}")

    # Summary
    if exported_files:
        print("\n" + "=" * 50)
        print("Export complete! Files created:")
        for filepath, count, dtype in exported_files:
            print(f"  - {filepath} ({count} {dtype})")
        print("=" * 50)

    return exported_files


def main():
    """Main interactive loop."""
    print_header()

    # Initialize client (no authentication needed for public endpoints)
    print("Connecting to Kalshi API...")
    try:
        client = KalshiClient()
        fetcher = MarketFetcher(client)
        exporter = DataExporter()
        print("Connected successfully!\n")
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)

    # Main loop
    while True:
        try:
            # Main menu
            mode_idx = select_option(
                "What would you like to do?",
                ["Browse markets by category", "Quit"]
            )

            if mode_idx == 1:
                print("\nGoodbye!")
                break

            # Browse by category
            markets = browse_by_category(fetcher)

            display_markets(markets)

            if not markets:
                continue

            # Count active markets
            active_markets = [m for m in markets if m.get('status') == 'active']
            print(f"\n{len(active_markets)} of {len(markets)} markets are active.")

            # Export mode selection
            export_mode = select_option(
                "How would you like to export?",
                ["Export all active markets", "Select specific markets", "Select single market"]
            )

            # Determine which markets to export
            selected_markets = []
            keyword = "markets"  # Default keyword for filename

            if export_mode == 0:
                # All active markets
                selected_markets = active_markets
                if not selected_markets:
                    print("No active markets found.")
                    continue

            elif export_mode == 1:
                # Select specific markets
                market_input = get_input("\nEnter market numbers (comma-separated, e.g., 1,3,5)")
                if not market_input:
                    continue

                try:
                    indices = [int(x.strip()) - 1 for x in market_input.split(",")]
                    for idx in indices:
                        if 0 <= idx < len(markets):
                            selected_markets.append(markets[idx])
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas.")
                    continue

                if not selected_markets:
                    print("No valid markets selected.")
                    continue

            elif export_mode == 2:
                # Single market (existing behavior)
                market_input = get_input("\nEnter market number or ticker")
                if not market_input:
                    continue

                selected_market = None
                try:
                    idx = int(market_input) - 1
                    if 0 <= idx < len(markets):
                        selected_market = markets[idx]
                except ValueError:
                    for m in markets:
                        if m.get("ticker", "").upper() == market_input.upper():
                            selected_market = m
                            break

                if not selected_market:
                    print("Invalid selection. Please try again.")
                    continue

                selected_markets = [selected_market]

            print(f"\nSelected {len(selected_markets)} market(s) for export.")

            # Data type selection
            data_type_idx = select_option(
                "What data to export?",
                ["Market info/metadata", "Candlesticks (OHLC)", "Trade history", "All data types"]
            )

            data_types = []
            if data_type_idx == 0:
                data_types = ['info']
            elif data_type_idx == 1:
                data_types = ['candlesticks']
            elif data_type_idx == 2:
                data_types = ['trades']
            elif data_type_idx == 3:
                data_types = ['info', 'candlesticks', 'trades']

            # Export
            export_multiple_markets(fetcher, exporter, selected_markets, data_types, keyword)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
