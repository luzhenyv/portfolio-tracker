# Manual Valuation (CLI)

This guide documents `scripts/manual_valuation.py` and provides a quick command cookbook plus deeper reference.

## Overview

`scripts/manual_valuation.py` is a CLI tool to manually fetch and fix valuation data in your portfolio tracker. It helps resolve missing or stale valuation data that may cause errors in the UI.

## Problem It Solves

The portfolio tracker has **two types of valuation**:

1. **Watchlist Valuation Metrics** (Forward P/E, PEG, EV/EBITDA, growth rates)
   - Stored in `valuation_metrics` table
   - Used by the watchlist page to show BUY/WAIT/AVOID signals
   - Can be missing if data was never fetched or API failed

2. **Portfolio Market Value** (NAV calculation)
   - Computed from positions × latest prices
   - Can fail if price data is missing or has NULL close values
   - Used in Overview and Positions pages

## Quick Reference Commands

### Quick Fixes

```bash
# Fix empty watchlist (no valuation data showing)
python scripts/manual_valuation.py --mode multiples

# Fix portfolio market value errors (missing prices)
python scripts/manual_valuation.py --mode prices
python scripts/manual_valuation.py --mode market

# Do everything at once (recommended after restore)
python scripts/manual_valuation.py --mode both
```

### Specific Tickers

```bash
# Update single ticker's valuation metrics
python scripts/manual_valuation.py --mode multiples --symbols AAPL

# Update multiple tickers
python scripts/manual_valuation.py --mode multiples --symbols AAPL,MSFT,GOOGL

# Fetch prices for specific tickers
python scripts/manual_valuation.py --mode prices --symbols TSLA,NVDA
```

### Historical Analysis

```bash
# Get market value as of a specific date
python scripts/manual_valuation.py --mode market --as-of 2026-01-20

# Compare market value across dates
python scripts/manual_valuation.py --mode market --as-of 2026-01-01
python scripts/manual_valuation.py --mode market --as-of 2026-01-15
python scripts/manual_valuation.py --mode market  # current
```

### Manual Price Overrides

```bash
# Override prices manually (when API fails or data is wrong)
python scripts/manual_valuation.py --mode market --override-price AAPL=192.10,TSLA=235.50

# Combine with as-of date
python scripts/manual_valuation.py --mode market --as-of 2026-01-20 --override-price AAPL=190.00
```

### Troubleshooting Checks (DB)

```bash
# Check what's in the database
sqlite3 db/portfolio.db "SELECT ticker FROM assets;"
sqlite3 db/portfolio.db "SELECT asset_id, COUNT(*) FROM prices_daily GROUP BY asset_id;"
sqlite3 db/portfolio.db "SELECT a.ticker, v.pe_forward, v.peg FROM valuation_metrics v JOIN assets a ON a.id = v.asset_id;"
```

### Advanced Options

```bash
# Use different database file
python scripts/manual_valuation.py --mode both --db-path /path/to/custom.db

# Force refetch (future enhancement)
python scripts/manual_valuation.py --mode multiples --force

# Get help
python scripts/manual_valuation.py --help
```

## Detailed Usage

### 1. Fetch Valuation Metrics for All Assets

```bash
python scripts/manual_valuation.py --mode multiples
```

This fetches:
- Forward P/E ratio
- PEG ratio
- EV/EBITDA
- Revenue growth %
- EPS growth %

### 2. Fetch Metrics for Specific Tickers Only

```bash
python scripts/manual_valuation.py --mode multiples --symbols AAPL,MSFT,GOOGL
```

### 3. Compute Current Portfolio Market Value

```bash
python scripts/manual_valuation.py --mode market
```

Computes market value using **latest available prices** from the database.

### 4. Compute Market Value as of Specific Date

```bash
python scripts/manual_valuation.py --mode market --as-of 2026-01-20
```

Uses prices from a specific historical date (must exist in `prices_daily` table).

### 5. Manual Price Override

```bash
python scripts/manual_valuation.py --mode market --override-price AAPL=192.10,TSLA=235.50
```

### 6. Fetch Missing Price Data

```bash
python scripts/manual_valuation.py --mode prices
```

This fetches latest price data for all assets (last 90 days if no prices exist).

### 7. Do Everything at Once

```bash
python scripts/manual_valuation.py --mode both
```

Fetches both valuation metrics AND computes market value.

## Common Workflows

### Fix Empty Watchlist Valuation

**Symptom:** Watchlist page shows no data or "No valuation data available"

**Solution:**
```bash
python scripts/manual_valuation.py --mode multiples
```

Then refresh the Streamlit UI.

### Fix Portfolio Market Value Errors

**Symptom:** Overview page crashes or shows $0.00 / NULL errors

**Solution:**
```bash
python scripts/manual_valuation.py --mode prices
python scripts/manual_valuation.py --mode market
```

### Update Single Asset

```bash
python scripts/manual_valuation.py --mode multiples --symbols AAPL
```

### Historical Performance Check

```bash
python scripts/manual_valuation.py --mode market --as-of 2026-01-01
```

## Troubleshooting

### "No assets to process"
- Check that you have assets in the database
- Verify ticker symbols are correct (case-insensitive)
- Run: `sqlite3 db/portfolio.db "SELECT ticker FROM assets;"`

### "No price available"
- Run `--mode prices` first to fetch missing price data
- Check if ticker is valid on Yahoo Finance
- Use `--override-price` for manual entry

### "Ticker XXXX not found in database"
- Add the asset first via the UI or `asset_service.py`
- Check spelling of ticker symbol

### Rate Limiting / Slow Performance
The script includes automatic rate limiting (0.5s between requests) to respect Yahoo Finance API limits. For many assets, this is normal and prevents errors.

## Integration With Existing Tools

### vs `data/fetch_prices.py`
- `fetch_prices.py`: Fetches **price history** (OHLC data)
- `manual_valuation.py --mode prices`: Wrapper around the same logic, with better UI

### vs Running `fetch_prices.py` Directly
Running `python data/fetch_prices.py` only fetches **prices**, not **valuation metrics**. The `manual_valuation.py` script handles both.

### vs Daily Update Job
`jobs/daily_update.py` is designed for automated daily runs. `manual_valuation.py` is for:
- One-off manual fixes
- Debugging missing data
- Custom date ranges
- Specific asset updates

## Technical Details

### What It Does Internally

**Multiples Mode:**
1. Queries assets from database
2. Calls `yfinance.Ticker(symbol).info` for each
3. Extracts valuation metrics (Forward P/E, PEG, etc.)
4. Upserts into `valuation_metrics` table
5. Handles missing data (stores as NULL, never synthetic values)

**Market Mode:**
1. Loads all positions from `position_state` table
2. Joins with latest prices from `prices_daily` table
3. Multiplies shares × close price
4. Sums to get total market value
5. Reports missing prices for debugging

**Prices Mode:**
1. Determines last stored price date per asset
2. Fetches new data from day after last date
3. Bulk inserts into `prices_daily` table
4. Idempotent (safe to run multiple times)

### Data Sources
- **Yahoo Finance API** (via `yfinance` Python library)
- Free, no API key required
- Subject to rate limiting (handled automatically)

### Database Tables Modified
- `valuation_metrics` (multiples mode)
- `prices_daily` (prices mode)
- No tables modified in market mode (read-only computation)

## Exit Codes

- `0`: Success
- `1`: Invalid arguments or critical error

## Logging

All operations are logged with timestamps and severity levels:
- `INFO`: Normal operations
- `WARNING`: Missing data, skipped items
- `ERROR`: Failed operations

Logs are printed to stdout in real-time.
