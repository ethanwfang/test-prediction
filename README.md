# Prediction Market Data Fetcher

Fetch historical data from Kalshi prediction markets and export to CSV for analysis. Uses public API endpoints - no authentication required.

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `requests` - API calls
- `pandas` - CSV export
- `sentence-transformers` - Semantic search (downloads ~80MB model on first use)

## Usage

```bash
cd src
python main.py
```

## Features

### 1. Search by Keyword
Search across all Kalshi markets by keyword.

### 2. Browse by Category
Browse markets organized by category (Economics, Politics, Sports, etc.):
- **Semantic search** - Find series using natural language (e.g., "Federal Reserve" finds "Fed funds rate")
- **Browse all series** - See all series in a category
- **Get all markets** - Fetch every market in a category

### 3. Bulk Export
Export data from multiple markets at once:
- **Export all active markets** - Automatically export all markets with "active" status
- **Select specific markets** - Choose which markets to export (comma-separated)
- **Select single market** - Export one market

### 4. Data Types
Choose what data to export:
- **Market info/metadata** - Ticker, title, status, prices, volume
- **Candlesticks (OHLC)** - Historical price data (daily)
- **Trade history** - Individual trades with timestamps
- **All data types** - Export everything

## Example Session

```
==================================================
  Kalshi Market Data Fetcher
==================================================

Connecting to Kalshi API...
Connected successfully!

How would you like to find markets?
  [1] Search by keyword
  [2] Browse by category
  [3] Quit

Choice: 2

Found 15 categories:

  [1] Climate and Weather (23 series)
  [2] Company News (45 series)
  [3] Economics (357 series)
  ...

Enter category number: 3

'Economics' has 357 series.
How would you like to find series?
  [1] Search by keywords (semantic matching)
  [2] Browse all series
  [3] Get all markets in category

Choice: 1

Enter keywords to search series: federal reserve

Searching for 'federal reserve'...

Found 10 matching series (threshold: 25%)

Fetching markets from all matching series...
  [1/10] Fed meeting (FEDDECISION) - 55% match - 45 markets
  [2/10] Fed funds rate (FED) - 41% match - 23 markets
  ...

Found 156 total markets from 10 series.

45 of 156 markets are active.

How would you like to export?
  [1] Export all active markets
  [2] Select specific markets
  [3] Select single market

Choice: 1

What data to export?
  [1] Market info/metadata
  [2] Candlesticks (OHLC)
  [3] Trade history
  [4] All data types

Choice: 4

Exporting data from 45 markets...

==================================================
Export complete! Files created:
  - data/markets_markets_info_20250118_143022.csv (45 markets)
  - data/markets_candlesticks_20250118_143022.csv (1,234 candlesticks)
  - data/markets_trades_20250118_143022.csv (5,678 trades)
==================================================
```

## Output Files

CSV files are saved to the `data/` directory.

### Market Info CSV
| ticker | title | status | yes_price | volume | series_ticker |
|--------|-------|--------|-----------|--------|---------------|
| FED-29JAN-425 | Fed rate above 4.25%? | active | 45 | 1234 | FED |

### Candlesticks CSV
| market_ticker | market_title | datetime | open | high | low | close | volume |
|---------------|--------------|----------|------|------|-----|-------|--------|
| FED-29JAN-425 | Fed rate... | 2025-01-15 | 0.42 | 0.45 | 0.40 | 0.44 | 500 |

### Trades CSV
| market_ticker | market_title | datetime | price | count | taker_side |
|---------------|--------------|----------|-------|-------|------------|
| FED-29JAN-425 | Fed rate... | 2025-01-15 10:30 | 0.44 | 10 | yes |

## Project Structure

```
prediction-market/
├── src/
│   ├── kalshi_client.py   # API client (public endpoints)
│   ├── market_fetcher.py  # Market search and data retrieval
│   ├── data_exporter.py   # CSV export with pandas
│   └── main.py            # Interactive CLI
├── data/                  # CSV output directory
├── requirements.txt
└── README.md
```
