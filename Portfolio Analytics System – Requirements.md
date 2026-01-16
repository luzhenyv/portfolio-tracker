# 📋 Portfolio Analytics System – Requirements Document

## 1. Project Overview

### 1.1 Purpose

This project aims to build a **local-first personal investment analytics system** to support long-term portfolio review and decision-making.

The system focuses on:

* Portfolio transparency
* Risk awareness
* Valuation context
* Calm, explainable decision signals

The system is **read-only by design** and does **not** execute trades.

### 1.2 Target Users

* Individual long-term investors
* Technical users comfortable running local Python tools
* Investors who prioritize discipline, risk control, and fundamentals

### 1.3 Investment Philosophy Constraints

* Long-term holding (quarters or longer)
* Stocks only (Phase 1)
* No speculation or short-term trading logic
* Options explicitly excluded in Phase 1

---

## 2. System Scope

### 2.1 In Scope (Phase 1)

* US-listed stocks
* End-of-day (EOD) price data
* Portfolio analytics
* Risk metrics
* Valuation screening
* Read-only dashboard

### 2.2 Out of Scope (Phase 1)

* Real-time data
* Trade execution
* Backtesting engine
* Alerts / notifications
* Options, ETFs, crypto

---

## 3. Functional Requirements

### 3.1 Market Data Ingestion

**FR-1: Price Fetcher**

* Fetch daily EOD prices from Yahoo Finance
* Data fields: `date`, `ticker`, `close_price`
* Idempotent execution (safe to run daily)
* Persist data into SQLite

**FR-2: Data Quality Rules**

* Missing data must not crash the system
* No forward-filling or price interpolation
* Store raw fetched values only

---

### 3.2 Portfolio Management

**FR-3: Position Tracking**

* Store positions manually in SQLite
* Fields:

  * ticker
  * shares
  * buy_price

**FR-4: Portfolio Metrics**

* Market value per position
* Unrealized P&L
* Portfolio allocation weight (%)
* Total portfolio value

---

### 3.3 Risk Analytics

**FR-5: Asset-Level Risk Metrics**

* Historical volatility (annualized)
* Maximum drawdown
* Return series derived from EOD prices

**FR-6: Portfolio-Level Risk Metrics**

* Weighted portfolio volatility
* Risk contribution by position

**FR-7: Risk Constraints**

* Risk metrics must be explainable
* No probabilistic forecasting or VaR models

---

### 3.4 Valuation Metrics (Auto-Fetched)

**FR-8: Valuation Data Fetching**

* Fetch valuation metrics automatically via Yahoo Finance
* Metrics include (when available):

  * Forward P/E
  * PEG ratio
  * EV / EBITDA
  * Revenue growth
  * Earnings growth

**FR-9: Missing Data Handling**

* Missing valuation fields must be stored as `NULL`
* No synthetic or estimated values allowed

---

### 3.5 Decision Engine

**FR-10: Decision Signals**

* Generate qualitative signals only:

  * HOLD
  * REDUCE
  * REVIEW
  * BUY / WAIT / AVOID (watchlist)

**FR-11: Rule-Based Logic**

* Decisions must be deterministic
* Rules based on:

  * Allocation weight
  * Risk contribution
  * Drawdown severity
  * Valuation bands

**FR-12: Explainability**

* Every decision must include textual reasons
* No black-box models allowed

---

### 3.6 Dashboard (UI)

**FR-13: Dashboard Framework**

* Built using Streamlit
* Local execution only

**FR-14: UI Style**

* Read-only
* No sliders or tunable parameters
* Executive-style, calm layout

**FR-15: Dashboard Pages**

* Portfolio Overview
* Positions Detail
* Watchlist / Valuation View

---

## 4. Non-Functional Requirements

### 4.1 Deployment

* Runs locally on macOS (Mac mini)
* No cloud dependency
* Single-user system

### 4.2 Storage

* SQLite as the only database
* Database file must be portable and backup-friendly

### 4.3 Performance

* Daily execution acceptable
* No real-time performance constraints

### 4.4 Reliability

* System must not fail due to missing data
* Partial data should still allow dashboard rendering

---

## 5. Technical Stack

* Python 3.10+
* SQLite
* pandas / NumPy
* yfinance
* Streamlit

---

## 6. Development Guidelines

### 6.1 Code Quality

* Clear module separation
* Deterministic outputs
* Emphasis on readability over optimization

### 6.2 Extensibility

* Design must allow future extension to:

  * ETFs
  * Crypto
  * Options overlays

---

## 7. Success Criteria

The project is considered successful if:

* Portfolio state can be reviewed in under 5 minutes
* Risk concentration is immediately visible
* Valuation context supports buy/wait decisions
* Outputs are trusted and explainable

---

## 8. Explicit Non-Goals

* Alpha generation
* Market timing
* Automated trading
* Prediction accuracy competitions

---

## 9. Disclaimer

This system is a **decision-support tool only**.
All investment decisions remain the responsibility of the user.
