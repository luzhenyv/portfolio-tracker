import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path("db/portfolio.db")


def load_price_history():
    query = """
    SELECT
        a.ticker,
        p.date,
        p.adjusted_close
    FROM prices_daily p
    JOIN assets a ON p.asset_id = a.id
    WHERE a.status = 'OWNED'
    ORDER BY p.date
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn, parse_dates=["date"])
    conn.close()
    return df


def compute_returns(df):
    pivot = df.pivot(index="date", columns="ticker", values="adjusted_close")
    returns = np.log(pivot / pivot.shift(1)).dropna()
    return returns


def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)


def max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def portfolio_weights():
    query = """
    SELECT
        a.ticker,
        SUM(p.shares * p.buy_price) AS cost
    FROM positions p
    JOIN assets a ON p.asset_id = a.id
    WHERE a.status = 'OWNED'
    GROUP BY a.ticker
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()

    df["weight"] = df["cost"] / df["cost"].sum()
    return df.set_index("ticker")["weight"]

def portfolio_weights_cost_based():
    """
    Cost-based portfolio weights.
    Reflects original capital allocation decisions.
    """
    query = """
    SELECT
        a.ticker,
        SUM(p.shares * p.buy_price) AS invested_cost
    FROM positions p
    JOIN assets a ON p.asset_id = a.id
    WHERE a.status = 'OWNED'
    GROUP BY a.ticker
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()

    df["weight"] = df["invested_cost"] / df["invested_cost"].sum()
    return df.set_index("ticker")["weight"]


def compute_portfolio_returns(returns, weights):
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(aligned_weights)
    return portfolio_returns


def compute_risk_metrics():
    prices = load_price_history()
    returns = compute_returns(prices)

    weights = portfolio_weights()

    asset_vol = annualized_volatility(returns)
    asset_mdd = returns.apply(max_drawdown)

    portfolio_ret = compute_portfolio_returns(returns, weights)
    portfolio_vol = portfolio_ret.std() * np.sqrt(252)
    portfolio_mdd = max_drawdown(portfolio_ret)

    corr = returns.corr()

    return {
        "asset_volatility": asset_vol,
        "asset_max_drawdown": asset_mdd,
        "portfolio_volatility": portfolio_vol,
        "portfolio_max_drawdown": portfolio_mdd,
        "correlation": corr
    }


if __name__ == "__main__":
    metrics = compute_risk_metrics()

    print("\n📉 Asset Volatility (Annualized)")
    print(metrics["asset_volatility"])

    print("\n📉 Asset Max Drawdown")
    print(metrics["asset_max_drawdown"])

    print("\n📊 Portfolio Volatility")
    print(f"{metrics['portfolio_volatility']:.2%}")

    print("\n📊 Portfolio Max Drawdown")
    print(f"{metrics['portfolio_max_drawdown']:.2%}")

    print("\n🔗 Correlation Matrix")
    print(metrics["correlation"])
