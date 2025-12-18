"""
Kalshi API client for public market data (no authentication required).
"""

import requests


class KalshiClient:
    """Client for accessing public Kalshi API endpoints."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        """Initialize the Kalshi client for public endpoints."""
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _request(self, method: str, endpoint: str, params: dict = None):
        """
        Make a request to the Kalshi API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/markets")
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.BASE_URL}{endpoint}"

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, params: dict = None):
        """Make a GET request."""
        return self._request("GET", endpoint, params=params)
