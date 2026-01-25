# 📊 Personal Portfolio Tracker System

A **local-first, read-only investment review system** built with Python, SQLAlchemy, SQLite, and Streamlit.

This project is designed for **long-term investors** who want:
- Clear portfolio analytics
- Risk-aware decision signals
- Valuation context (not price prediction)
- A calm, executive-style review dashboard

> Philosophy: *Facts → Signals → Decisions*  
> No overfitting. No speculation. No emotional UI.

---

## ✨ Key Features

### 📈 Market Data
- End-of-day (EOD) price fetcher from **Yahoo Finance**
- Cached locally in **SQLite** with SQLAlchemy ORM
- Idempotent daily execution (safe to run repeatedly)

### 💼 Portfolio Analytics
- Cost-based portfolio construction
- Market value, P&L, allocation weights
- Supports long and short positions
- **Two cost tracking methods**:
  - `long_avg_cost` — Tax basis (weighted average of purchase prices)
  - `net_invested_avg_cost` — Cash still at risk for P&L display
- Simplified trading: Buy and Sell operations handle everything
  - **Buy**: Covers short positions first, then adds to long
  - **Sell**: Reduces long positions first, then creates shorts

#### Net Invested Avg Cost Formula

```
Net Invested Avg Cost = (Total Cash Out - Total Cash In) / Remaining Shares
```

**Example (TSLA):**
| Action | Shares | Price | Cash Flow |
|--------|--------|-------|-----------|
| BUY    | 100    | $150  | -$15,000  |
| SELL   | 20     | $250  | +$5,000   |
| SELL   | 50     | $170  | +$8,500   |

- **Remaining Shares**: 30
- **Net Invested**: $15,000 - $5,000 - $8,500 = **$1,500**
- **Net Invested Avg Cost**: $1,500 / 30 = **$50**

This reflects your actual cash at risk, not the original purchase price.

### ⚠️ Risk Metrics
- Historical volatility (annualized)
- Maximum drawdown
- Portfolio-level aggregation
- Correlation matrix for diversification analysis

### 🧠 Decision Engine
- Rule-based, explainable decisions:
  - `HOLD` - No action needed
  - `REDUCE` - Consider reducing position size
  - `REVIEW` - Warrants further analysis
- Every action has **explicit reasons**

### 👀 Valuation (Watchlist)
- Auto-fetched valuation multiples:
  - Forward P/E, PEG, EV/EBITDA
  - Revenue & EPS growth
- Band-based valuation signals:
  - `BUY` / `WAIT` / `AVOID`

### 🖥️ Review Dashboard
- Built with **Streamlit**
- Read-only, executive-style UI
- Three views: Overview, Positions, Watchlist

---

## 🚀 Deployment (Docker & PostgreSQL)

The project can be deployed using Docker and PostgreSQL for a more robust setup.

### ⚡ Quick Start

1. **Ensure Docker and Docker Compose are installed.**
2. **Run the following command:**
   ```bash
   docker compose up --build
   ```
3. **Access the dashboard** at `http://localhost:8501`.

### ⚙️ Database Configuration

By default, the Docker setup uses PostgreSQL with the following credentials (defined in `docker-compose.yml`):
- **User**: `portfolio_user`
- **Password**: `portfolio_password`
- **Database**: `portfolio_tracker`

To use an external database, set the `PORTFOLIO_DB_URL` environment variable:
```bash
PORTFOLIO_DB_URL=postgresql+psycopg://user:password@host:port/dbname
```

---

## 🧱 Project Structure

```text
portfolio-tracker/
├── config.py               # Centralized configuration
├── main.py                 # CLI entry point
├── pyproject.toml          # Project metadata & dependencies
│
├── analytics/              # Analytics modules
│   ├── __init__.py
│   ├── portfolio.py        # Portfolio metrics
│   ├── risk.py             # Risk calculations
│   ├── valuation.py        # Valuation analysis
│   └── performance.py      # Return calculations
│
├── data/                   # Data fetching
│   ├── __init__.py
│   └── fetch_prices.py     # Yahoo Finance integration
│
├── db/                     # Database layer (SQLAlchemy)
│   ├── __init__.py
│   ├── models.py           # ORM models
│   ├── repositories.py     # Data access layer
│   ├── session.py          # Session management
│   ├── init_db.py          # DB initialization
│   └── schema.sql          # Legacy SQL schema
│
├── decision/               # Decision engine
│   ├── __init__.py
│   └── engine.py           # Rule-based decisions
│
├── jobs/                   # Scheduled jobs
│   ├── __init__.py
│   └── daily_update.py     # Daily data refresh
│
└── ui/                     # Streamlit dashboard
    ├── __init__.py
    └── app.py              # Dashboard application
```

## 🛠️ Maintenance & Backups

The system provides a dedicated backup tool for data safety and auditability.

```bash
# Full SQLite binary backup (safe while app is running)
python scripts/backup_portfolio.py --format sqlite --out backups/my_portfolio.db

# Export core ledgers to CSV for audit
python scripts/backup_portfolio.py --format csv --out backups/export_jan_25/
```

- **SQLite Format**: Best for disaster recovery. Creates a bit-for-bit copy.
- **CSV Format**: Best for manual inspection or importing into Excel. Exports Assets, Trades, and Cash Transactions.

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
# Using uv
uv sync
```

### 2️⃣ Initialize Database

```bash
# Initialize empty database
python main.py init

# Or with sample data for testing
python db/init_db.py --sample-data
```

### 3️⃣ Trading Operations

```bash
# Buy shares (opens or adds to long position)
python main.py buy AAPL --shares 100 --price 150.00 --date 2024-01-15

# Sell shares (reduces long position or creates short)
python main.py sell AAPL --shares 50 --price 170.00 --date 2024-06-15

# Example: Selling more than you own creates a short
python main.py sell AAPL --shares 150 --price 180.00
# This closes 100 long shares and opens 50 short shares

# Buying when short covers the short position first
python main.py buy AAPL --shares 60 --price 175.00
# This covers 50 short shares and opens 10 long shares

# View trade history
python main.py trades --limit 10

# View realized P&L
python main.py pnl --since 2024-01-01
```

### 4️⃣ Fetch Market Data

```bash
# Run full daily update (prices + valuations)
python main.py update

# Or run from jobs module
python -m jobs.daily_update
```

### 5️⃣ Launch Dashboard

```bash
python main.py dashboard

# Or directly with Streamlit
streamlit run ui/app.py
```

### 6️⃣ View Summary

```bash
python main.py summary
```

---

## 🗄️ Database Design (SQLAlchemy ORM)

### Core Models

- **Asset**: Tracked securities (stocks, future: ETFs, crypto)
- **PriceDaily**: End-of-day price data
- **Position**: Holdings with cost basis (supports multiple lots)
- **ValuationMetric**: Auto-fetched valuation multiples
- **WatchlistTarget**: Target prices for watchlist items
- **InvestmentThesis**: Investment rationale documentation

### Repository Pattern

Clean data access through repository classes:
- `AssetRepository`: Asset CRUD operations
- `PriceRepository`: Price data management
- `PositionRepository`: Position management
- `ValuationRepository`: Valuation data

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite |
| Data Processing | pandas, NumPy |
| Market Data | yfinance |
| Dashboard | Streamlit |

---

## 📐 Configuration

All thresholds are configurable in `config.py`:

```python
# Concentration thresholds
concentration_warning_pct: 0.30   # 30%
concentration_danger_pct: 0.40   # 40%
concentration_extreme_pct: 0.60  # 60%

# Valuation bands
pe_cheap_threshold: 15.0
pe_fair_threshold: 25.0
peg_cheap_threshold: 1.0
peg_fair_threshold: 1.5
```

---

## 🔒 Design Philosophy

### What This Project **Is**

* A **decision support system**
* A **portfolio review tool**
* A **risk-aware analytics platform**

### What This Project **Is NOT**

* ❌ A trading bot
* ❌ A price prediction engine
* ❌ A real-time system
* ❌ A backtesting playground

> The goal is **better decisions**, not more activity.

---

## 📈 Future Enhancements

- [ ] Decision history & audit trail
- [ ] Quarterly review snapshots
- [ ] Market-value weighted portfolio view
- [ ] Options overlay (advanced)
- [ ] ETF and crypto support

---

## ⚠️ Disclaimer

This project is for **educational and personal use only**.
It does not constitute financial advice.

All investment decisions are the responsibility of the user.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

