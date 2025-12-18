"""
Data export functionality for prediction market data.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


class DataExporter:
    """Export market data to CSV files."""

    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data exporter.

        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, ticker: str, data_type: str, suffix: str = "") -> str:
        """
        Generate a filename for the export.

        Args:
            ticker: Market ticker
            data_type: Type of data (candlesticks, trades)
            suffix: Optional suffix (e.g., time period)

        Returns:
            Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [ticker, data_type]
        if suffix:
            parts.append(suffix)
        parts.append(timestamp)
        return "_".join(parts) + ".csv"

    def export_candlesticks(
        self,
        candlesticks: list[dict],
        ticker: str,
        period: int,
    ) -> str:
        """
        Export candlestick data to CSV.

        Args:
            candlesticks: List of candlestick dictionaries
            ticker: Market ticker
            period: Candlestick period in minutes

        Returns:
            Path to the exported CSV file
        """
        if not candlesticks:
            raise ValueError("No candlestick data to export")

        period_names = {1: "1min", 60: "1hr", 1440: "1day"}
        period_suffix = period_names.get(period, f"{period}min")

        df = pd.DataFrame(candlesticks)

        # Convert timestamp to datetime if present
        if "end_period_ts" in df.columns:
            df["datetime"] = pd.to_datetime(df["end_period_ts"], unit="s")

        # Reorder columns for clarity
        preferred_order = [
            "datetime",
            "end_period_ts",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "price",
            "volume",
            "open_interest",
        ]
        existing_cols = [c for c in preferred_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in preferred_order]
        df = df[existing_cols + other_cols]

        filename = self._generate_filename(ticker, "candlesticks", period_suffix)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)

        return str(filepath)

    def export_trades(self, trades: list[dict], ticker: str) -> str:
        """
        Export trade data to CSV.

        Args:
            trades: List of trade dictionaries
            ticker: Market ticker

        Returns:
            Path to the exported CSV file
        """
        if not trades:
            raise ValueError("No trade data to export")

        df = pd.DataFrame(trades)

        # Convert timestamp to datetime if present
        if "created_time" in df.columns:
            df["datetime"] = pd.to_datetime(df["created_time"])

        # Reorder columns for clarity
        preferred_order = [
            "datetime",
            "created_time",
            "ticker",
            "trade_id",
            "price",
            "count",
            "taker_side",
        ]
        existing_cols = [c for c in preferred_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in preferred_order]
        df = df[existing_cols + other_cols]

        filename = self._generate_filename(ticker, "trades")
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)

        return str(filepath)

    def export_market_summary(self, market: dict, ticker: str) -> str:
        """
        Export market summary/metadata to CSV.

        Args:
            market: Market details dictionary
            ticker: Market ticker

        Returns:
            Path to the exported CSV file
        """
        df = pd.DataFrame([market])

        filename = self._generate_filename(ticker, "summary")
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)

        return str(filepath)
