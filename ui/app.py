"""
Streamlit Dashboard for Portfolio Review.

FR-13: Built using Streamlit, local execution only
FR-14: Read-only, executive-style, calm layout
FR-15: Portfolio Overview, Positions Detail, Watchlist/Valuation View
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from db import init_db, get_db
from db.repositories import CashRepository
from analytics.portfolio import compute_portfolio
from analytics.risk import compute_risk_metrics
from analytics.valuation import run_valuation
from analytics.performance import get_nav_period_returns, get_nav_series
from decision.engine import decision_engine


# Initialize database on app start
init_db()


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Portfolio Review",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    st.sidebar.title("📊 Portfolio Tracker")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Positions", "Watchlist"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Read-only dashboard")
    st.sidebar.caption("📅 Data: End-of-day prices")

    return page


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimals}%}"


def render_overview_page():
    """
    Render Portfolio Overview page.

    Shows:
    - Portfolio summary metrics
    - Risk metrics
    - Decision summary
    """
    st.header("📌 Portfolio Overview")

    try:
        portfolio_df, summary = compute_portfolio()
        risk = compute_risk_metrics()
        decisions = decision_engine()
    except ValueError as e:
        st.warning(f"⚠️ {e}")
        st.info("Add positions to see portfolio analytics.")
        return
    except Exception as e:
        st.error(f"Error loading portfolio data: {e}")
        return

    # Portfolio Value Metrics
    st.subheader("💰 Portfolio Value")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cost",
        format_currency(summary["total_cost"]),
    )
    col2.metric(
        "Market Value",
        format_currency(summary["total_market_value"]),
    )
    col3.metric(
        "Unrealized P&L",
        format_currency(summary["total_pnl"]),
        format_percentage(summary["total_pnl_pct"]),
    )
    col4.metric(
        "Positions",
        f"{len(portfolio_df)}",
    )

    st.divider()

    # Asset Allocation Pie Chart
    st.subheader("📊 Asset Allocation")

    # Get cash balance
    db = get_db()
    with db.session() as session:
        cash_repo = CashRepository(session)
        cash_balance = cash_repo.get_balance()

    # Build allocation data: Cash + each stock's market value
    allocation_data = []

    if cash_balance > 0:
        allocation_data.append({"Asset": "Cash", "Value": cash_balance})

    for _, row in portfolio_df.iterrows():
        # Use net market value (long - short)
        market_value = row.get("net", 0)
        if market_value > 0:
            allocation_data.append({"Asset": row["ticker"], "Value": market_value})

    if allocation_data:
        alloc_df = pd.DataFrame(allocation_data)
        total_value = alloc_df["Value"].sum()
        alloc_df["Percentage"] = alloc_df["Value"] / total_value * 100

        # Create pie chart
        fig = px.pie(
            alloc_df,
            values="Value",
            names="Asset",
            title="",
            hole=0.4,  # Donut chart style
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<br>Weight: %{percent}<extra></extra>",
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
        )

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.plotly_chart(fig)

        with col_table:
            # Show allocation table
            display_alloc = alloc_df.copy()
            display_alloc["Value"] = display_alloc["Value"].apply(lambda x: f"${x:,.0f}")
            display_alloc["Percentage"] = display_alloc["Percentage"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(
                display_alloc,
                hide_index=True,
            )
            st.metric("Total", format_currency(total_value))
    else:
        st.info("No assets to display. Add cash or positions.")

    # Period Returns Section (2 parts: metrics + NAV chart)
    try:
        period_returns = get_nav_period_returns()
        nav_series = get_nav_series(lookback_days=365)

        # Show section if we have either returns or NAV data
        if (any(v is not None for v in period_returns.values())) or nav_series:
            st.divider()
            st.subheader("📈 Period Returns")

            # Part 1: Return metrics (1m, 3m, 6m, 1y)
            if any(v is not None for v in period_returns.values()):
                return_cols = st.columns(len(period_returns))
                for col, (period, ret) in zip(return_cols, period_returns.items()):
                    if ret is not None:
                        col.metric(period, format_percentage(ret))
                    else:
                        col.metric(period, "N/A")

            # Part 2: NAV line chart (1Y lookback)
            if nav_series and len(nav_series.daily) > 0:
                st.markdown("")
                st.caption("Daily Net Asset Value (Last 1 Year)")

                # Build DataFrame for plotting
                nav_data = pd.DataFrame(
                    [(nav.date, nav.nav) for nav in nav_series.daily],
                    columns=["Date", "NAV"],
                )

                # Create line chart
                fig = px.line(
                    nav_data,
                    x="Date",
                    y="NAV",
                    title="",
                    labels={"NAV": "Net Asset Value ($)", "Date": "Date"},
                )
                fig.update_traces(
                    line_color="#19D3F3",
                    line_width=2,
                    hovertemplate="<b>%{x}</b><br>NAV: $%{y:,.0f}<extra></extra>",
                )
                fig.update_layout(
                    hovermode="x unified",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="#f0f0f0",
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="#f0f0f0",
                        tickformat="$,.0f",
                    ),
                    margin=dict(t=20, b=40, l=60, r=20),
                    height=300,
                )

                st.plotly_chart(fig)
            else:
                st.info("No NAV history available. Add trades or cash transactions to see chart.")
    except Exception as e:
        st.caption(f"Period returns unavailable: {e}")
        pass  # Period returns are optional

    st.divider()

    # Risk Metrics
    st.subheader("⚠️ Risk Metrics")
    col5, col6 = st.columns(2)

    col5.metric(
        "Portfolio Volatility (Ann.)",
        format_percentage(risk["portfolio_volatility"]),
        help="Annualized standard deviation of portfolio returns",
    )
    col6.metric(
        "Max Drawdown",
        format_percentage(risk["portfolio_max_drawdown"]),
        help="Maximum peak-to-trough decline",
    )

    st.divider()

    # Decision Summary
    st.subheader("🧠 Decision Summary")

    if decisions.empty:
        st.info("No positions to evaluate.")
    else:
        # Highlight actions needing attention
        review_count = len(decisions[decisions["action"].isin(["REDUCE", "REVIEW"])])

        if review_count > 0:
            st.warning(f"⚡ {review_count} position(s) need attention")

        # Style the dataframe
        display_df = decisions[["ticker", "weight", "action", "reasons"]].copy()
        display_df["weight"] = display_df["weight"].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            display_df,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "weight": st.column_config.TextColumn("Weight", width="small"),
                "action": st.column_config.TextColumn("Action", width="small"),
                "reasons": st.column_config.TextColumn("Reasons", width="large"),
            },
        )


def render_positions_page():
    """
    Render Positions Detail page.

    Shows:
    - Detailed position table
    - Individual risk metrics
    """
    st.header("📈 Owned Positions")

    try:
        portfolio_df, summary = compute_portfolio()
        risk = compute_risk_metrics()
        decisions = decision_engine()
    except ValueError as e:
        st.warning(f"⚠️ {e}")
        return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Merge portfolio with decisions
    if not decisions.empty:
        merged = portfolio_df.merge(
            decisions[["ticker", "action", "reasons"]],
            on="ticker",
            how="left",
        )
    else:
        merged = portfolio_df.copy()
        merged["action"] = "HOLD"
        merged["reasons"] = ""

    # Format for display
    display_df = merged.copy()

    # Use net_invested_avg_cost from the dataframe
    display_df["avg_cost"] = display_df["net_invested_avg_cost"]
    display_df["market_value"] = display_df["net"]  # Use net exposure
    # Use net_invested_pnl for P&L (market value - net invested)
    display_df["display_pnl"] = display_df["net_invested_pnl"]
    display_df["pnl_pct"] = (display_df["net_invested_pnl"] / display_df["net_invested"]).where(
        display_df["net_invested"] > 0, 0
    )

    # Format numeric columns for display
    display_df["avg_cost"] = display_df["avg_cost"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) and x > 0 else "N/A"
    )
    display_df["close"] = display_df["close"].apply(lambda x: f"${x:.2f}")
    display_df["market_value"] = display_df["market_value"].apply(lambda x: f"${x:,.0f}")
    display_df["display_pnl"] = display_df["display_pnl"].apply(lambda x: f"${x:+,.0f}")
    display_df["pnl_pct"] = display_df["pnl_pct"].apply(
        lambda x: f"{x:+.1%}" if pd.notna(x) and abs(x) < 100 else "N/A"
    )
    display_df["net_weight"] = display_df["net_weight"].apply(lambda x: f"{x:.1%}")

    display_cols = [
        "ticker",
        "net_shares",
        "avg_cost",
        "close",
        "market_value",
        "display_pnl",
        "pnl_pct",
        "net_weight",
        "action",
        "reasons",
    ]

    # Rename columns for display
    column_config = {
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "net_shares": st.column_config.TextColumn("Shares", width="small"),
        "avg_cost": st.column_config.TextColumn("Net Avg Cost", width="small"),
        "close": st.column_config.TextColumn("Current", width="small"),
        "market_value": st.column_config.TextColumn("Market Value", width="medium"),
        "display_pnl": st.column_config.TextColumn("P&L", width="small"),
        "pnl_pct": st.column_config.TextColumn("P&L %", width="small"),
        "net_weight": st.column_config.TextColumn("Weight", width="small"),
        "action": st.column_config.TextColumn("Action", width="small"),
        "reasons": st.column_config.TextColumn("Reasons", width="large"),
    }

    st.dataframe(
        display_df[display_cols],
        hide_index=True,
        column_config=column_config,
    )

    st.caption("💡 Weights are cost-based. Risk metrics are historical and EOD-based.")

    # Correlation matrix (if multiple positions)
    if not risk["correlation"].empty and len(risk["correlation"]) > 1:
        st.divider()
        st.subheader("🔗 Correlation Matrix")
        st.dataframe(
            risk["correlation"].round(2),
        )
        st.caption("Correlation of daily returns. Lower correlation = better diversification.")


def render_watchlist_page():
    """
    Render Watchlist & Valuation page.

    Shows:
    - Valuation metrics for all assets
    - BUY/WAIT/AVOID signals

    Future: Add asset creation UI using services.asset_service.create_asset_with_data()
    Example integration:
        from services.asset_service import create_asset_with_data, AssetStatus

        with st.form("add_asset_form"):
            ticker = st.text_input("Ticker Symbol")
            status = st.selectbox("Status", ["OWNED", "WATCHLIST"])
            submitted = st.form_submit_button("Add Asset")

            if submitted and ticker:
                result = create_asset_with_data(
                    ticker,
                    AssetStatus.OWNED if status == "OWNED" else AssetStatus.WATCHLIST
                )
                if result.success:
                    st.success(result.status_message)
                    if result.prices_fetched > 0:
                        st.metric("Prices Fetched", result.prices_fetched)
                else:
                    st.error("Failed to add asset")
    """
    st.header("👀 Watchlist & Valuation")

    try:
        valuation_df = run_valuation()
    except Exception as e:
        st.error(f"Error loading valuation data: {e}")
        return

    if valuation_df.empty:
        st.info("No valuation data available.")
        st.caption("Run the valuation fetcher to populate this view.")
        return

    # Format for display
    display_df = valuation_df.copy()

    # Format numeric columns
    for col in ["pe_forward", "peg", "ev_ebitda"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    for col in ["revenue_growth", "eps_growth"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    display_cols = [
        "ticker",
        "pe_forward",
        "peg",
        "ev_ebitda",
        "revenue_growth",
        "eps_growth",
        "valuation_action",
    ]

    # Color-code actions
    def highlight_action(row):
        action = row.get("valuation_action", "")
        if action == "BUY":
            return ["background-color: #d4edda"] * len(row)
        elif action == "AVOID":
            return ["background-color: #f8d7da"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df[display_cols],
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "pe_forward": st.column_config.TextColumn("Fwd P/E", width="small"),
            "peg": st.column_config.TextColumn("PEG", width="small"),
            "ev_ebitda": st.column_config.TextColumn("EV/EBITDA", width="small"),
            "revenue_growth": st.column_config.TextColumn("Rev Growth", width="small"),
            "eps_growth": st.column_config.TextColumn("EPS Growth", width="small"),
            "valuation_action": st.column_config.TextColumn("Signal", width="small"),
        },
    )

    # Legend
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.markdown("🟢 **BUY** — Multiple metrics suggest attractive valuation")
    col2.markdown("🟡 **WAIT** — Mixed signals, patience recommended")
    col3.markdown("🔴 **AVOID** — Multiple metrics suggest overvaluation")

    st.caption(
        "💡 Valuation is multiples-based (auto-fetched). "
        "Decisions are band-based, not precise price targets."
    )


def main():
    """Main application entry point."""
    configure_page()

    # Render navigation
    page = render_sidebar()

    # Render selected page
    if page == "Overview":
        render_overview_page()
    elif page == "Positions":
        render_positions_page()
    elif page == "Watchlist":
        render_watchlist_page()


if __name__ == "__main__":
    main()
