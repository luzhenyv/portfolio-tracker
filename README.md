# 📊 Personal Portfolio Tracker System

A **local-first, read-only investment review system** built with Python, SQLite, and Streamlit.

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
- Cached locally in **SQLite**
- Deterministic and reproducible

### 💼 Portfolio Analytics
- Cost-based portfolio construction
- Market value, P&L, allocation weights
- Supports long-term holding review

### ⚠️ Risk Metrics
- Historical volatility
- Maximum drawdown
- Portfolio-level aggregation
- Based on historical returns (not forecasts)

### 🧠 Decision Engine
- Rule-based, explainable decisions:
  - `HOLD`
  - `REDUCE`
  - `REVIEW`
- Decisions are driven by:
  - Allocation weight
  - Risk contribution
  - Drawdown behavior
- Every action has **explicit reasons**

### 👀 Valuation (Watchlist)
- Auto-fetched valuation multiples:
  - Forward P/E
  - PEG
  - EV / EBITDA
  - Growth metrics
- Band-based valuation signals:
  - BUY / WAIT / AVOID
- Designed for *screening*, not timing

### 🖥️ Review Dashboard
- Built with **Streamlit**
- Read-only, executive-style UI
- Optimized for daily / weekly / quarterly review
- No sliders, no parameter tuning

---

## 🧱 Project Structure

```text
.
├── data/
│   └── portfolio.db          # SQLite database
│
├── fetcher/
│   └── yahoo_price.py        # Yahoo Finance price fetcher
│
├── analytics/
│   ├── portfolio.py          # Portfolio aggregation
│   ├── risk.py               # Risk metrics
│   └── valuation.py          # Valuation logic
│
├── decision/
│   └── engine.py             # Rule-based decision engine
│
├── ui/
│   └── app.py                # Streamlit dashboard
│
├── requirements.txt
└── README.md
````

---

## 🗄️ Database Design (SQLite)

This project uses **SQLite** for simplicity, transparency, and portability.

### Core Tables

* `prices`

  * ticker
  * date
  * close

* `positions`

  * ticker
  * shares
  * buy_price

* `valuations`

  * ticker
  * pe_forward
  * peg
  * ev_ebitda
  * growth_metrics

> SQLite is intentional:
> ✔ Easy backup
> ✔ No infra dependency
> ✔ Perfect for personal / research systems

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Fetch Market Data

```bash
python fetcher/yahoo_price.py
```

This will populate / update the local SQLite database with EOD prices.

### 3️⃣ Launch Dashboard

```bash
streamlit run ui/app.py
```

---

## 🧠 Design Philosophy

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

## 📌 Intended Workflow

1. Fetch prices (daily or weekly)
2. Open dashboard
3. Review:

   * Portfolio health
   * Risk exposure
   * Valuation context
4. Make **human decisions** outside the system

---

## 🔒 Read-Only by Design

* No UI controls to change logic
* No in-dashboard optimization
* All rules live in code
* Encourages discipline and consistency

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **SQLite**
* **pandas**
* **NumPy**
* **yfinance**
* **Streamlit**

---

## 📈 Future Enhancements (Optional)

* Decision history & audit trail
* Quarterly review snapshots
* Market-value weighted portfolio view
* Alert summaries (email / message)
* Options overlay (advanced)

---

## ⚠️ Disclaimer

This project is for **educational and personal use only**.
It does not constitute financial advice.

All investment decisions are the responsibility of the user.

---

## 🧑‍💻 Author

Built for long-term, fundamentals-driven investing
with an emphasis on clarity, risk control, and calm execution.

