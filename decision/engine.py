import sqlite3
import pandas as pd
from pathlib import Path

from analytics.risk import compute_risk_metrics
from analytics.portfolio import compute_portfolio

DB_PATH = Path("db/portfolio.db")


def load_cost_weights():
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

    total = df["invested_cost"].sum()
    df["weight"] = df["invested_cost"] / total
    return df.set_index("ticker")


def decision_engine():
    # Load metrics
    risk = compute_risk_metrics()
    portfolio_df, _ = compute_portfolio()
    weights = load_cost_weights()

    decisions = []

    for ticker in weights.index:
        weight = weights.loc[ticker, "weight"]
        vol = risk["asset_volatility"].get(ticker)
        mdd = risk["asset_max_drawdown"].get(ticker)

        action = "HOLD"
        reasons = []

        if weight > 0.6:
            action = "REDUCE"
            reasons.append("Extreme concentration (>60%)")

        elif weight > 0.4 and vol and vol > 0.5:
            action = "REVIEW_RISK"
            reasons.append("High concentration + high volatility")

        if mdd and mdd < -0.5:
            action = "REVIEW_RISK"
            reasons.append("Severe historical drawdown")

        decisions.append({
            "ticker": ticker,
            "weight": round(weight, 2),
            "volatility": round(vol, 2) if vol else None,
            "max_drawdown": round(mdd, 2) if mdd else None,
            "action": action,
            "reasons": "; ".join(reasons)
        })

    return pd.DataFrame(decisions)


if __name__ == "__main__":
    df = decision_engine()
    print("\n🧠 Decision Engine Output")
    print(df)
