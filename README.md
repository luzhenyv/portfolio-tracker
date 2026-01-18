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
- Supports multiple buy lots per position

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

### 3️⃣ Add Assets and Positions

```bash
# Add an asset to track
python main.py add-asset AAPL --name "Apple Inc." --status OWNED

# Add a position (buy lot)
python main.py add-position AAPL --shares 100 --price 150.00 --date 2024-01-15

# List tracked assets
python main.py list
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

