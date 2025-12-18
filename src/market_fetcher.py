"""
Market data fetching functionality for Kalshi API.
"""

import time
from typing import Optional

from kalshi_client import KalshiClient


class MarketFetcher:
    """Fetch market data from Kalshi."""

    def __init__(self, client: KalshiClient):
        """
        Initialize the market fetcher.

        Args:
            client: Authenticated KalshiClient instance
        """
        self.client = client

    def search_markets(
        self,
        query: str = None,
        status: str = "open",
        max_results: int = 20,
    ) -> list[dict]:
        """
        Search for markets matching criteria.

        Fetches markets in batches and filters client-side since the API
        doesn't support text search.

        Args:
            query: Search term to filter markets (searches in title/ticker)
            status: Market status filter (open, closed, settled)
            max_results: Maximum number of matching markets to return

        Returns:
            List of market dictionaries
        """
        matching_markets = []
        cursor = None
        pages_fetched = 0
        max_pages = 10  # Limit pagination to avoid rate limits

        while pages_fetched < max_pages and len(matching_markets) < max_results:
            params = {
                "limit": 1000,
                "status": status,
            }
            if cursor:
                params["cursor"] = cursor

            response = self.client.get("/markets", params=params)
            markets = response.get("markets", [])
            cursor = response.get("cursor")

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                for m in markets:
                    if (query_lower in m.get("title", "").lower()
                        or query_lower in m.get("ticker", "").lower()
                        or query_lower in m.get("subtitle", "").lower()):
                        matching_markets.append(m)
                        if len(matching_markets) >= max_results:
                            break
            else:
                matching_markets.extend(markets)

            pages_fetched += 1

            # Stop if no more results
            if not cursor or not markets:
                break

            # Rate limit: wait between requests
            time.sleep(0.3)

        return matching_markets[:max_results]

    def get_market(self, ticker: str) -> dict:
        """
        Get details for a specific market.

        Args:
            ticker: The market ticker

        Returns:
            Market details dictionary
        """
        response = self.client.get(f"/markets/{ticker}")
        return response.get("market", {})

    def get_candlesticks(
        self,
        ticker: str,
        series_ticker: str,
        period_interval: int = 1440,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[dict]:
        """
        Get candlestick (OHLC) data for a market.

        Args:
            ticker: The market ticker
            series_ticker: The series ticker the market belongs to
            period_interval: Candlestick period in minutes (1, 60, or 1440)
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)

        Returns:
            List of candlestick dictionaries with OHLC data
        """
        params = {
            "period_interval": period_interval,
        }

        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts

        endpoint = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        response = self.client.get(endpoint, params=params)

        return response.get("candlesticks", [])

    def get_trades(
        self,
        ticker: str = None,
        limit: int = 1000,
        cursor: str = None,
    ) -> tuple[list[dict], Optional[str]]:
        """
        Get trade history for markets.

        Args:
            ticker: Optional market ticker to filter trades
            limit: Maximum number of trades to return (1-1000)
            cursor: Pagination cursor for fetching more results

        Returns:
            Tuple of (trades list, next cursor or None)
        """
        params = {
            "limit": min(limit, 1000),
        }

        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor

        response = self.client.get("/markets/trades", params=params)

        trades = response.get("trades", [])
        next_cursor = response.get("cursor")

        return trades, next_cursor

    def get_all_trades(self, ticker: str, max_trades: int = 10000) -> list[dict]:
        """
        Get all trades for a market, handling pagination.

        Args:
            ticker: Market ticker
            max_trades: Maximum total trades to fetch

        Returns:
            List of all trade dictionaries
        """
        all_trades = []
        cursor = None

        while len(all_trades) < max_trades:
            trades, cursor = self.get_trades(ticker=ticker, cursor=cursor)
            all_trades.extend(trades)

            if not cursor or not trades:
                break

        return all_trades[:max_trades]

    def get_events(self, limit: int = 100) -> list[dict]:
        """
        Get list of events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        params = {"limit": limit}
        response = self.client.get("/events", params=params)
        return response.get("events", [])

    def get_event(self, event_ticker: str) -> dict:
        """
        Get details for a specific event.

        Args:
            event_ticker: The event ticker

        Returns:
            Event details dictionary
        """
        response = self.client.get(f"/events/{event_ticker}")
        return response.get("event", {})

    def get_series_list(self) -> list[dict]:
        """
        Get list of all series with their categories.

        Returns:
            List of series dictionaries
        """
        response = self.client.get("/series")
        return response.get("series", [])

    def get_series(self, series_ticker: str) -> dict:
        """
        Get details for a specific series.

        Args:
            series_ticker: The series ticker

        Returns:
            Series details dictionary
        """
        response = self.client.get(f"/series/{series_ticker}")
        return response.get("series", {})

    def get_markets_by_series(self, series_ticker: str, status: str = "open") -> list[dict]:
        """
        Get all markets for a specific series.

        Args:
            series_ticker: The series ticker to filter by
            status: Market status filter (open, closed, settled)

        Returns:
            List of market dictionaries
        """
        params = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": 1000,
        }
        response = self.client.get("/markets", params=params)
        return response.get("markets", [])
