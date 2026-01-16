import streamlit as st
import pandas as pd

from analytics.portfolio import compute_portfolio
from analytics.risk import compute_risk_metrics
from decision.engine import decision_engine
from analytics.valuation import run_valuation

st.set_page_config(
    page_title="Portfolio Review",
    layout="wide"
)

st.title("📊 Portfolio Review Dashboard")

page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Positions", "Watchlist"]
)

# ======================
# OVERVIEW
# ======================
if page == "Overview":
    st.header("📌 Portfolio Overview")

    portfolio_df, summary = compute_portfolio()
    risk = compute_risk_metrics()
    decisions = decision_engine()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Cost",
        f"${summary['total_cost']:,.0f}"
    )
    col2.metric(
        "Market Value",
        f"${summary['total_market_value']:,.0f}"
    )
    col3.metric(
        "Unrealized P&L",
        f"${summary['total_pnl']:,.0f}",
        f"{summary['total_pnl_pct']:.1%}"
    )

    st.divider()

    col4, col5 = st.columns(2)
    col4.metric(
        "Portfolio Volatility",
        f"{risk['portfolio_volatility']:.1%}"
    )
    col5.metric(
        "Max Drawdown",
        f"{risk['portfolio_max_drawdown']:.1%}"
    )

    st.divider()
    st.subheader("🧠 Decision Summary")
    st.dataframe(
        decisions[["ticker", "weight", "action", "reasons"]],
        use_container_width=True
    )

# ======================
# POSITIONS
# ======================
elif page == "Positions":
    st.header("📈 Owned Positions")

    portfolio_df, _ = compute_portfolio()
    risk = compute_risk_metrics()
    decisions = decision_engine()

    merged = portfolio_df.merge(
        decisions,
        on="ticker",
        how="left"
    )

    display_cols = [
        "ticker",
        "shares",
        "buy_price",
        "close",
        "market_value",
        "pnl",
        "pnl_pct",
        "weight",
        "action",
        "reasons"
    ]

    st.dataframe(
        merged[display_cols],
        use_container_width=True
    )

    st.caption(
        "Weights are cost-based. Risk metrics are historical and EOD-based."
    )

# ======================
# WATCHLIST
# ======================
elif page == "Watchlist":
    st.header("👀 Watchlist & Valuation")

    valuation_df = run_valuation()

    if valuation_df.empty:
        st.info("No valuation data available.")
    else:
        display_cols = [
            "ticker",
            "pe_forward",
            "peg",
            "ev_ebitda",
            "revenue_growth",
            "eps_growth",
            "valuation_action"
        ]

        st.dataframe(
            valuation_df[display_cols],
            use_container_width=True
        )

        st.caption(
            "Valuation is multiples-based (auto-fetched). Decisions are band-based, not precise price targets."
        )
