import sqlite3
from datetime import date
import yfinance as yf
import pandas as pd
from pathlib import Path

DB_PATH = Path("db/portfolio.db")


def get_assets(conn):
    query = """
    SELECT id, ticker
    FROM assets
    """
    return conn.execute(query).fetchall()


def get_latest_price_date(conn, asset_id):
    query = """
    SELECT MAX(date)
    FROM prices_daily
    WHERE asset_id = ?
    """
    result = conn.execute(query, (asset_id,)).fetchone()
    return result[0]


def fetch_and_store_prices(asset_id, ticker, start_date=None):
    print(f"📥 Fetching prices for {ticker}")

    df = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"⚠️ No data for {ticker}")
        return

    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    records = []
    for _, row in df.iterrows():
        records.append((
            asset_id,
            row["Date"],
            row["Open"],
            row["High"],
            row["Low"],
            row["Close"],
            row["Adj Close"],
            int(row["Volume"])
        ))

    conn = sqlite3.connect(DB_PATH)
    conn.executemany("""
        INSERT OR IGNORE INTO prices_daily
        (asset_id, date, open, high, low, close, adjusted_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()

    print(f"✅ Stored {len(records)} records for {ticker}")

def get_watchlist_and_owned_assets():
    query = """
    SELECT id, ticker
    FROM assets
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(query).fetchall()
    conn.close()
    return rows


def fetch_valuation_for_ticker(ticker):
    info = yf.Ticker(ticker).info

    return {
        "pe_forward": info.get("forwardPE"),
        "peg": info.get("pegRatio"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "revenue_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsGrowth"),
    }


def upsert_valuation(asset_id, metrics):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO valuation_metrics (
            asset_id,
            pe_forward,
            peg,
            ev_ebitda,
            revenue_growth,
            eps_growth
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
            pe_forward = excluded.pe_forward,
            peg = excluded.peg,
            ev_ebitda = excluded.ev_ebitda,
            revenue_growth = excluded.revenue_growth,
            eps_growth = excluded.eps_growth,
            updated_at = CURRENT_TIMESTAMP
    """, (
        asset_id,
        metrics["pe_forward"],
        metrics["peg"],
        metrics["ev_ebitda"],
        metrics["revenue_growth"],
        metrics["eps_growth"],
    ))
    conn.commit()
    conn.close()


def main():
    conn = sqlite3.connect(DB_PATH)
    assets = get_assets(conn)

    for asset_id, ticker in assets:
        latest_date = get_latest_price_date(conn, asset_id)

        # If no data yet, fetch max history
        start_date = None
        if latest_date:
            # Fetch from next day to avoid duplicates
            start_date = pd.to_datetime(latest_date) + pd.Timedelta(days=1)

        fetch_and_store_prices(asset_id, ticker, start_date)

    conn.close()


if __name__ == "__main__":
    main()
