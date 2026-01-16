import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("db/portfolio.db")


def load_valuation_inputs():
    query = """
    SELECT
        a.id AS asset_id,
        a.ticker,
        v.pe_forward,
        v.peg,
        v.ev_ebitda,
        v.revenue_growth,
        v.eps_growth
    FROM valuation_metrics v
    JOIN assets a ON v.asset_id = a.id
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def score_metric(value, cheap, fair):
    if value is None:
        return None
    if value < cheap:
        return "CHEAP"
    if value <= fair:
        return "FAIR"
    return "EXPENSIVE"


def valuation_decision(row):
    scores = []

    if row["peg"] is not None:
        scores.append(score_metric(row["peg"], 1.0, 1.5))

    if row["pe_forward"] is not None:
        scores.append(score_metric(row["pe_forward"], 20, 35))

    if scores.count("EXPENSIVE") >= 2:
        return "AVOID"

    if scores.count("CHEAP") >= 2:
        return "BUY"

    return "WAIT"


def run_valuation():
    df = load_valuation_inputs()
    df["valuation_action"] = df.apply(valuation_decision, axis=1)
    return df


if __name__ == "__main__":
    df = run_valuation()
    print("\n📐 Valuation Engine Output")
    print(df[[
        "ticker",
        "pe_forward",
        "peg",
        "ev_ebitda",
        "revenue_growth",
        "eps_growth",
        "valuation_action"
    ]])
