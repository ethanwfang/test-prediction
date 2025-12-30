# Prediction Market Data Fetcher & Analyzer

Fetch historical data from Kalshi prediction markets, export to CSV, and generate comprehensive analysis reports. Uses public API endpoints - no authentication required.

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `requests` - API calls
- `pandas` - Data processing and CSV export
- `sentence-transformers` - Semantic search (downloads ~80MB model on first use)
- `matplotlib` - Chart generation
- `seaborn` - Statistical visualizations

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

### 5. Analyze Exported Data
Generate comprehensive analysis reports from exported CSV files:
- **Self-contained HTML reports** - All charts embedded as base64, ready for offline viewing
- **Auto deep-dive analysis** - Detects related market groups and analyzes convergence patterns
- **Probability distribution charts** - Shows how probability shifted between competing outcomes
- **Convergence analysis** - Identifies when markets "knew" the answer before resolution
- **Market calibration** - Visualizes how well prices predicted actual outcomes

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

## Standalone Analysis CLI

You can also run analysis directly without the interactive menu:

```bash
# Generate HTML report (recommended)
python analysis/analyze.py --data-dir ./data --output ./report.html

# Generate markdown report with separate chart files
python analysis/analyze.py --data-dir ./data --format md --output-dir ./outputs

# Skip deep-dive analysis for faster generation
python analysis/analyze.py --data-dir ./data --output ./report.html --no-deep-dive
```

### Analysis Output

The HTML report includes:
- **Executive Summary** - Market counts, volumes, outcome statistics
- **Top Markets Table** - Highest volume markets with outcomes
- **Outcome Analysis** - YES/NO distribution with calibration chart
- **Deep Dive Sections** - Auto-detected related market groups with:
  - Probability distribution over time (stacked area chart)
  - Convergence analysis (when did markets cross 50%/90%?)
  - Auto-generated narrative summary
- **Price History Charts** - Individual market price movements
- **Key Insights** - Auto-generated observations

Example insight from analysis:
> "Fed Terminal Rate 2023 markets converged to the correct answer ~5 months before official resolution."

## Project Structure

```
prediction-market/
├── src/
│   ├── kalshi_client.py   # API client (public endpoints)
│   ├── market_fetcher.py  # Market search and data retrieval
│   ├── data_exporter.py   # CSV export with pandas
│   └── main.py            # Interactive CLI
├── analysis/
│   ├── analyze.py         # Standalone analysis CLI
│   ├── data_loader.py     # Load CSVs with proper dtypes
│   ├── stats.py           # Statistical calculations
│   ├── charts.py          # Chart generation (base64 support)
│   ├── report.py          # HTML/Markdown report generation
│   ├── deep_dive.py       # Auto group detection & convergence
│   └── probability_distribution.py  # Stacked area charts
├── data/                  # CSV output directory
├── outputs/               # Generated reports
├── requirements.txt
└── README.md
```
