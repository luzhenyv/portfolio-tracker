import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("db/portfolio.db")


def load_positions():
    query = """
    SELECT
        a.ticker,
        p.shares,
        p.buy_price,
        p.buy_date,
        a.id AS asset_id
    FROM positions p
    JOIN assets a ON p.asset_id = a.id
    WHERE a.status = 'OWNED'
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_latest_prices():
    query = """
    SELECT
        asset_id,
        date,
        close
    FROM prices_daily
    WHERE (asset_id, date) IN (
        SELECT asset_id, MAX(date)
        FROM prices_daily
        GROUP BY asset_id
    )
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def compute_portfolio():
    positions = load_positions()
    prices = load_latest_prices()

    if positions.empty:
        raise ValueError("No positions found.")

    df = positions.merge(prices, on="asset_id")

    df["cost"] = df["shares"] * df["buy_price"]
    df["market_value"] = df["shares"] * df["close"]
    df["pnl"] = df["market_value"] - df["cost"]
    df["pnl_pct"] = df["pnl"] / df["cost"]

    portfolio_value = df["market_value"].sum()
    df["weight"] = df["market_value"] / portfolio_value

    summary = {
        "total_cost": df["cost"].sum(),
        "total_market_value": portfolio_value,
        "total_pnl": df["pnl"].sum(),
        "total_pnl_pct": df["pnl"].sum() / df["cost"].sum()
    }

    return df, summary


if __name__ == "__main__":
    df, summary = compute_portfolio()

    print("\n📊 Portfolio Summary")
    for k, v in summary.items():
        print(f"{k}: {v:,.2f}")

    print("\n📈 Positions")
    print(df[[
        "ticker",
        "shares",
        "buy_price",
        "close",
        "market_value",
        "pnl",
        "pnl_pct",
        "weight"
    ]])
