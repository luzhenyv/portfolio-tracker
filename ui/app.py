"""
Streamlit Dashboard for Portfolio Review.

FR-13: Built using Streamlit, local execution only
FR-14: Read-only, executive-style, calm layout
FR-15: Portfolio Overview, Positions Detail, Watchlist/Valuation View
"""

from typing import Literal, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, date, timedelta

from config import config
from db import init_db, AssetStatus, AssetType
from analytics.portfolio import compute_portfolio, portfolio_weights
from analytics.risk import compute_risk_metrics
from analytics.valuation import run_valuation
from analytics.performance import get_nav_period_returns, get_nav_series
from analytics.optimizer import compute_efficient_frontier, PortfolioOptimizer
from analytics.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
)
from decision.engine import decision_engine
from services.position_service import buy_position, sell_position
from services.asset_service import create_asset_with_data, get_all_tickers
from services.cash_service import get_cash_balance, deposit_cash, withdraw_cash, get_cash_ledger
from services.market_index_service import (
    get_all_indices,
    get_normalized_index_prices,
    get_benchmark_comparison_data,
    create_index,
    delete_index,
    sync_index_prices,
)
from services.trade_service import (
    get_recent_trades,
    get_latest_trade_ids_by_ticker,
    update_trade,
    delete_trade,
)
from services.note_service import (
    create_note_for_asset,
    create_market_note,
    create_journal_entry,
    get_recent_notes,
    get_notes_for_asset,
    get_market_symbols,
    update_note,
    archive_note,
    pin_note,
    NoteResult,
)
from services.tag_service import (
    get_all_tags_with_counts,
    create_tag,
    rename_tag,
    delete_tag,
    get_tags_for_ticker,
    set_asset_tags_by_names,
    get_assets_by_tag_names,
    get_untagged_assets,
)
from services.asset_service import delete_assets
from db.models import NoteType, NoteTargetKind


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

    # Build page list based on config
    pages = ["Overview", "Positions", "Watchlist", "Charts", "Assets & Tags", "Notes"]
    if config.ui.enable_admin_ui:
        pages.append("Admin")

    page = st.sidebar.radio(
        "Navigation",
        pages,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    if page == "Admin":
        st.sidebar.warning("⚠️ Write-enabled mode")
        st.sidebar.caption("Trade & cash operations")
    else:
        st.sidebar.caption("💡 Read-only dashboard")
        st.sidebar.caption("📅 Data: End-of-day prices")

    return page


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimals}%}"


def celebrate_and_rerun(
    msg: str = "Updating dashboard…",
    delay: float = 3,
    animation: Optional[Literal["balloons", "snow", "toast"]] = "balloons",
):
    """Display celebrations and wait before rerunning."""
    if animation == "balloons":
        st.balloons()
    elif animation == "snow":
        st.snow()
    elif animation == "toast":
        st.toast()
    with st.status(msg, expanded=False):
        time.sleep(delay)
    st.rerun()


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

    # Get cash balance for NAV calculation
    cash_balance = get_cash_balance()
    total_nav = summary["holdings_market_value"] + cash_balance

    # Portfolio Snapshot section
    st.subheader("📸 Portfolio Snapshot")

    # Show data freshness
    if summary.get("latest_price_date"):
        st.caption(f"📅 As of latest close: {summary['latest_price_date']}")

    # Row 1: NAV, Holdings, Cash, Positions
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total NAV",
        format_currency(total_nav),
        help="Net Asset Value = Holdings Market Value + Cash",
    )
    col2.metric(
        "Holdings",
        format_currency(summary["holdings_market_value"]),
        help="Net market value of all positions (long − short exposure)",
    )
    col3.metric(
        "Cash",
        format_currency(cash_balance),
        help="Available cash balance",
    )

    # Row 2: Unrealized P&L (total) and Today's P&L
    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Unrealized P&L",
        format_currency(summary["holdings_unrealized_pnl"]),
        format_percentage(summary["holdings_pnl_pct"]),
        help="Total unrealized profit/loss on holdings vs net invested capital",
    )

    # Today's P&L with color indicator
    today_pnl = summary.get("today_unrealized_pnl", 0)
    today_pnl_pct = summary.get("today_pnl_pct", 0)
    col5.metric(
        "Today P&L",
        format_currency(today_pnl),
        format_percentage(today_pnl_pct) if today_pnl != 0 else None,
        help="1-day unrealized P&L based on price change since prior close: Σ(close − prior_close) × net_shares",
    )

    col6.metric(
        "Positions",
        f"{summary.get('position_count', len(portfolio_df))}",
        help="Number of active positions",
    )

    # col7.metric(
    #     "Gross Exposure",
    #     format_currency(summary["gross_exposure"]),
    #     help="Sum of absolute market values: |long MV| + |short MV|",
    # )

    st.divider()

    # Asset Allocation Pie Chart
    st.subheader("📊 Asset Allocation")

    # Build allocation data: Cash + each stock's market value
    allocation_data = []

    if cash_balance > 0:
        allocation_data.append({"Asset": "Cash", "Type": "CASH", "Value": cash_balance})

    for _, row in portfolio_df.iterrows():
        # Use net market value (long - short)
        market_value = row.get("net", 0)
        if market_value > 0:
            allocation_data.append(
                {
                    "Asset": row["ticker"],
                    "Type": row.get("asset_type", "STOCK"),
                    "Value": market_value,
                }
            )

    if allocation_data:
        alloc_df = pd.DataFrame(allocation_data)
        total_value = alloc_df["Value"].sum()
        alloc_df["Percentage"] = alloc_df["Value"] / total_value * 100

        # Create pie charts
        col_pie1, col_pie2 = st.columns(2)

        with col_pie1:
            fig = px.pie(
                alloc_df,
                values="Value",
                names="Asset",
                title="By Ticker",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<br>Weight: %{percent}<extra></extra>",
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=30, b=20, l=20, r=20),
                height=300,
            )
            st.plotly_chart(fig)

        with col_pie2:
            type_df = alloc_df.groupby("Type")["Value"].sum().reset_index()
            fig_type = px.pie(
                type_df,
                values="Value",
                names="Type",
                title="By Asset Type",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_type.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<br>Weight: %{percent}<extra></extra>",
            )
            fig_type.update_layout(
                showlegend=False,
                margin=dict(t=30, b=20, l=20, r=20),
                height=300,
            )
            st.plotly_chart(fig_type)

        st.markdown("")
        # Show allocation table
        display_alloc = alloc_df.copy()
        display_alloc["Value"] = display_alloc["Value"].apply(lambda x: f"${x:,.0f}")
        display_alloc["Percentage"] = display_alloc["Percentage"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            display_alloc[["Asset", "Type", "Value", "Percentage"]],
            hide_index=True,
        )
    else:
        st.info("No assets to display. Add cash or positions.")

    # Period Returns Section with Benchmark Comparison
    try:
        st.divider()
        st.subheader("📈 Period Returns & Benchmark Comparison")

        # Get available benchmarks from database
        available_indices = get_all_indices()

        # Time horizon options
        time_horizons = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
        }

        # Initialize session state for benchmarks and time horizon
        if "selected_benchmarks" not in st.session_state:
            # Default to S&P 500 if available
            default_benchmarks = (
                ["SPX"]
                if available_indices and any(idx.symbol == "SPX" for idx in available_indices)
                else []
            )
            st.session_state.selected_benchmarks = default_benchmarks

        if "selected_horizon" not in st.session_state:
            st.session_state.selected_horizon = "1 Year"

        # Get the lookback days for selected horizon
        lookback_days = time_horizons[st.session_state.selected_horizon]
        selected_benchmarks = st.session_state.selected_benchmarks

        # Fetch NAV data for the selected time horizon
        nav_series = get_nav_series(lookback_days=lookback_days)
        period_returns = get_nav_period_returns()

        # Display period return metrics
        if any(v is not None for v in period_returns.values()):
            st.markdown("")
            return_cols = st.columns(len(period_returns))
            for col, (period, ret) in zip(return_cols, period_returns.items()):
                if ret is not None:
                    col.metric(period, format_percentage(ret))
                else:
                    col.metric(period, "N/A")

        # Build normalized comparison chart
        if nav_series and len(nav_series.daily) > 0:
            st.markdown("")
            st.caption(
                f"Normalized Performance (First Day = 1.0) — {st.session_state.selected_horizon}"
            )

            # Build DataFrame for portfolio NAV (normalized)
            nav_dates = [nav.date for nav in nav_series.daily]
            nav_values = [nav.nav for nav in nav_series.daily]

            # Normalize portfolio NAV (first day = 1.0)
            base_nav = nav_values[0] if nav_values and nav_values[0] > 0 else 1
            normalized_nav = [v / base_nav for v in nav_values]

            # Create DataFrame for plotting
            chart_data = pd.DataFrame(
                {
                    "Date": nav_dates,
                    "Portfolio": normalized_nav,
                }
            )
            chart_data["Date"] = pd.to_datetime(chart_data["Date"])

            # Fetch and add benchmark data
            if selected_benchmarks:
                benchmark_data = get_benchmark_comparison_data(
                    selected_benchmarks,
                    lookback_days=lookback_days,
                )

                for symbol, series in benchmark_data.items():
                    if series and series.prices:
                        # Create a date-to-value mapping for the benchmark
                        bench_df = pd.DataFrame(
                            [{"Date": p.date, symbol: p.value} for p in series.prices]
                        )
                        bench_df["Date"] = pd.to_datetime(bench_df["Date"])

                        # Merge with chart data
                        chart_data = chart_data.merge(bench_df, on="Date", how="outer")

            # Sort by date and forward-fill missing values
            chart_data = chart_data.sort_values("Date")
            chart_data = chart_data.ffill()

            # Melt DataFrame for Plotly
            value_cols = [col for col in chart_data.columns if col != "Date"]
            chart_melted = chart_data.melt(
                id_vars=["Date"],
                value_vars=value_cols,
                var_name="Asset",
                value_name="Normalized Price",
            )

            # Define color palette (portfolio highlighted)
            color_map = {"Portfolio": "#19D3F3"}  # Cyan for portfolio
            palette = px.colors.qualitative.Set2
            for i, col in enumerate(value_cols):
                if col != "Portfolio":
                    color_map[col] = palette[i % len(palette)]

            # Create line chart
            fig = px.line(
                chart_melted,
                x="Date",
                y="Normalized Price",
                color="Asset",
                title="",
                color_discrete_map=color_map,
            )

            fig.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )

            # Make Portfolio line thick solid, benchmark lines dashed
            for trace in fig.data:
                if trace.name == "Portfolio":
                    trace.line.width = 3
                    trace.line.dash = "solid"
                else:
                    trace.line.width = 1
                    trace.line.dash = "dot"

            fig.update_layout(
                hovermode="x unified",
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    title="",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    title="Normalized Price",
                    tickformat=".2f",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                margin=dict(t=40, b=40, l=60, r=20),
                height=350,
            )

            # Create layout: controls on left, chart on right
            control_col, chart_col = st.columns([1, 3])

            with control_col:

                # Benchmark selection (multiselect)
                if available_indices:
                    benchmark_options = {
                        idx.symbol: f"{idx.name} ({idx.symbol})" for idx in available_indices
                    }

                    selected_benchmarks_new = st.multiselect(
                        "Compare with Benchmarks",
                        options=list(benchmark_options.keys()),
                        default=st.session_state.selected_benchmarks,
                        format_func=lambda x: benchmark_options.get(x, x),
                        help="Select market indices to compare your portfolio performance against",
                        key="benchmark_selector",
                    )
                    if selected_benchmarks_new != st.session_state.selected_benchmarks:
                        st.session_state.selected_benchmarks = selected_benchmarks_new
                        st.rerun()
                else:
                    st.caption("No benchmarks available. Add benchmarks in Admin → Benchmarks.")

                # Time horizon selection (pill-style buttons)
                horizon = st.pills(
                    "Time horizon",
                    options=list(time_horizons.keys()),
                    default=st.session_state.selected_horizon,
                )
                if horizon != st.session_state.selected_horizon:
                    st.session_state.selected_horizon = horizon
                    st.rerun()

            with chart_col:
                st.plotly_chart(fig)
        else:
            st.info("No NAV history available. Add trades or cash transactions to see chart.")
    except Exception as e:
        st.caption(f"Period returns unavailable: {e}")
        import traceback

        st.caption(traceback.format_exc())

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

        # Apply colored emojis to Action column
        def format_action(val):
            if val == "HOLD":
                return "🔵 HOLD"
            elif val == "REDUCE":
                return "🟠 REDUCE"
            elif val == "REVIEW":
                return "🟡 REVIEW"
            return val

        display_df["action"] = display_df["action"].apply(format_action)
        display_df["weight"] = display_df["weight"].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            display_df,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "weight": st.column_config.TextColumn("Weight"),
                "action": st.column_config.TextColumn("Action"),
                "reasons": st.column_config.TextColumn("Reasons"),
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

    # Apply colored emojis to Action column
    def format_action(val):
        if val == "HOLD":
            return "🔵 HOLD"
        elif val == "REDUCE":
            return "🟠 REDUCE"
        elif val == "REVIEW":
            return "🟡 REVIEW"
        return val

    merged["action"] = merged["action"].apply(format_action)

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
        "ticker": st.column_config.TextColumn("Ticker"),
        "net_shares": st.column_config.TextColumn("Shares"),
        "avg_cost": st.column_config.TextColumn(
            "Net Avg Cost",
            help="Average cost per share based on net invested capital (cost basis / net shares)",
        ),
        "close": st.column_config.TextColumn("Current", help="Latest end-of-day price per share"),
        "market_value": st.column_config.TextColumn(
            "Market Value",
            help="Net market value = (long shares × price) - (short shares × price)",
        ),
        "display_pnl": st.column_config.TextColumn(
            "P&L",
            help="Unrealized profit/loss = net market value - net invested capital",
        ),
        "pnl_pct": st.column_config.TextColumn(
            "P&L %",
            help="Return on invested capital = (net market value - net invested) / net invested",
        ),
        "net_weight": st.column_config.TextColumn(
            "Exposure %",
            help="Share of absolute net exposure: $\\frac{|MV_{net}|}{\\sum |MV_{net}|}$ (positions only, excludes cash)",
        ),
        "action": st.column_config.TextColumn("Action"),
        "reasons": st.column_config.TextColumn("Reasons"),
    }

    st.dataframe(
        display_df[display_cols],
        hide_index=True,
        column_config=column_config,
    )

    st.caption(
        "💡 Exposure % = $\\frac{|MV_{net}|}{\\sum |MV_{net}|}$ for positions only (excludes cash). "
        "Risk metrics are historical and EOD-based."
    )

    # Correlation matrix (if multiple positions)
    if not risk["correlation"].empty and len(risk["correlation"]) > 1:
        st.divider()
        st.subheader("🔗 Correlation Matrix")
        st.dataframe(
            risk["correlation"].round(2),
        )
        st.caption("Correlation of daily returns. Lower correlation = better diversification.")

    # Efficient Frontier Analysis
    _render_efficient_frontier_section(portfolio_df, summary)


def _render_efficient_frontier_section(portfolio_df: pd.DataFrame, summary: dict):
    """
    Render the Efficient Frontier visualization and rebalancing recommendations.

    Shows:
    - Scatter plot of risk/return with current and optimal portfolios marked
    - Max Sharpe and Min Volatility portfolio details
    - Rebalancing recommendations table
    """
    import plotly.graph_objects as go

    # Get current portfolio weights
    current_weights = portfolio_df.set_index("ticker")["net_weight"]

    # Check if we have enough positions
    if len(current_weights) < 2:
        return

    st.divider()
    st.subheader("📊 Efficient Frontier Analysis")

    # Lookback period options (in trading days)
    lookback_options = {
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "3 Years": 1095,
        "All Data": None,
    }

    # Initialize session state for lookback selection
    if "frontier_lookback" not in st.session_state:
        st.session_state.frontier_lookback = "1 Year"

    # Lookback period selector
    lookback_selection = st.pills(
        "Historical Data Period",
        options=list(lookback_options.keys()),
        default=st.session_state.frontier_lookback,
        help="Select how many years of historical data to use for computing expected returns and risk metrics",
    )
    if lookback_selection != st.session_state.frontier_lookback:
        st.session_state.frontier_lookback = lookback_selection
        st.rerun()

    lookback_days = lookback_options[st.session_state.frontier_lookback]

    # Compute efficient frontier
    with st.spinner("Computing optimal portfolios..."):
        frontier = compute_efficient_frontier(
            current_weights=current_weights,
            n_portfolios=10000,
            lookback_days=lookback_days,
        )

    if frontier is None:
        st.info("📊 Insufficient price history for frontier analysis (minimum 60 trading days required).")
        return

    # Create the efficient frontier chart
    fig = go.Figure()

    # Add random portfolios as scatter points (colored by Sharpe ratio)
    fig.add_trace(
        go.Scatter(
            x=frontier.portfolios["volatility"],
            y=frontier.portfolios["expected_return"],
            mode="markers",
            marker=dict(
                size=6,
                color=frontier.portfolios["sharpe_ratio"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe<br>Ratio"),
                opacity=0.7,
            ),
            name="Simulated Portfolios",
            hovertemplate=(
                "<b>Portfolio</b><br>"
                + "Expected Return: %{y:.1%}<br>"
                + "Volatility: %{x:.1%}<br>"
                + "Sharpe Ratio: %{marker.color:.2f}<extra></extra>"
            ),
        )
    )

    # Add Max Sharpe Portfolio marker
    fig.add_trace(
        go.Scatter(
            x=[frontier.max_sharpe_portfolio.volatility],
            y=[frontier.max_sharpe_portfolio.expected_return],
            mode="markers",
            marker=dict(
                symbol="star",
                size=20,
                color="red",
                line=dict(width=2, color="white"),
            ),
            name=f"Max Sharpe ({frontier.max_sharpe_portfolio.sharpe_ratio:.2f})",
            hovertemplate=(
                "<b>Max Sharpe Portfolio ⭐</b><br>"
                + f"Expected Return: {frontier.max_sharpe_portfolio.expected_return:.1%}<br>"
                + f"Volatility: {frontier.max_sharpe_portfolio.volatility:.1%}<br>"
                + f"Sharpe Ratio: {frontier.max_sharpe_portfolio.sharpe_ratio:.2f}<extra></extra>"
            ),
        )
    )

    # Add Min Volatility Portfolio marker
    fig.add_trace(
        go.Scatter(
            x=[frontier.min_volatility_portfolio.volatility],
            y=[frontier.min_volatility_portfolio.expected_return],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=16,
                color="blue",
                line=dict(width=2, color="white"),
            ),
            name=f"Min Volatility ({frontier.min_volatility_portfolio.volatility:.1%})",
            hovertemplate=(
                "<b>Min Volatility Portfolio 🛡️</b><br>"
                + f"Expected Return: {frontier.min_volatility_portfolio.expected_return:.1%}<br>"
                + f"Volatility: {frontier.min_volatility_portfolio.volatility:.1%}<br>"
                + f"Sharpe Ratio: {frontier.min_volatility_portfolio.sharpe_ratio:.2f}<extra></extra>"
            ),
        )
    )

    # Add Current Portfolio marker
    if frontier.current_portfolio:
        fig.add_trace(
            go.Scatter(
                x=[frontier.current_portfolio.volatility],
                y=[frontier.current_portfolio.expected_return],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=18,
                    color="orange",
                    line=dict(width=3, color="white"),
                ),
                name=f"Your Portfolio ({frontier.current_portfolio.sharpe_ratio:.2f})",
                hovertemplate=(
                    "<b>Your Current Portfolio 📍</b><br>"
                    + f"Expected Return: {frontier.current_portfolio.expected_return:.1%}<br>"
                    + f"Volatility: {frontier.current_portfolio.volatility:.1%}<br>"
                    + f"Sharpe Ratio: {frontier.current_portfolio.sharpe_ratio:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Efficient Frontier with Actual Stock Data",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Volatility (Risk)",
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            title="Expected Return",
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.5)",
        ),
        hovermode="closest",
        height=500,
    )

    st.plotly_chart(fig)

    # Portfolio comparison metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Your Portfolio**")
        if frontier.current_portfolio:
            st.metric("Expected Return", f"{frontier.current_portfolio.expected_return:.1%}")
            st.metric("Volatility", f"{frontier.current_portfolio.volatility:.1%}")
            st.metric("Sharpe Ratio", f"{frontier.current_portfolio.sharpe_ratio:.2f}")
        else:
            st.info("No current portfolio data")

    with col2:
        st.markdown("**⭐ Max Sharpe Portfolio**")
        st.metric("Expected Return", f"{frontier.max_sharpe_portfolio.expected_return:.1%}")
        st.metric("Volatility", f"{frontier.max_sharpe_portfolio.volatility:.1%}")
        st.metric("Sharpe Ratio", f"{frontier.max_sharpe_portfolio.sharpe_ratio:.2f}")

    with col3:
        st.markdown("**🛡️ Min Volatility Portfolio**")
        st.metric("Expected Return", f"{frontier.min_volatility_portfolio.expected_return:.1%}")
        st.metric("Volatility", f"{frontier.min_volatility_portfolio.volatility:.1%}")
        st.metric("Sharpe Ratio", f"{frontier.min_volatility_portfolio.sharpe_ratio:.2f}")

    # Rebalancing Recommendations
    st.divider()
    st.subheader("📋 Rebalancing Recommendations")
    st.caption("Suggested trades to achieve the Max Sharpe Ratio portfolio allocation")

    # Build recommendations table
    current_shares = portfolio_df.set_index("ticker")["net_shares"]
    current_prices = portfolio_df.set_index("ticker")["close"]
    portfolio_value = summary.get("holdings_market_value", 0)

    # Generate recommendations
    optimizer = PortfolioOptimizer()
    recommendations = optimizer.generate_rebalance_recommendations(
        current_weights=current_weights,
        current_shares=current_shares,
        current_prices=current_prices,
        target_portfolio=frontier.max_sharpe_portfolio,
        portfolio_value=portfolio_value,
        threshold=0.02,  # 2% threshold
    )

    if recommendations:
        rebal_data = []
        for rec in recommendations:
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(rec.action, "")
            rebal_data.append({
                "Ticker": rec.ticker,
                "Current %": f"{rec.current_weight:.1%}",
                "Optimal %": f"{rec.optimal_weight:.1%}",
                "Diff": f"{rec.weight_diff:+.1%}",
                "Current Shares": f"{rec.current_shares:.0f}",
                "Target Shares": f"{rec.target_shares:.1f}",
                "Action": f"{action_emoji} {rec.action}",
                "Trade Size": f"{abs(rec.shares_to_trade):.1f} shares",
                "Est. Value": f"${abs(rec.trade_value):,.0f}",
            })

        rebal_df = pd.DataFrame(rebal_data)
        st.dataframe(
            rebal_df,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Current %": st.column_config.TextColumn("Current %", help="Current portfolio weight"),
                "Optimal %": st.column_config.TextColumn("Optimal %", help="Optimal weight for max Sharpe"),
                "Diff": st.column_config.TextColumn("Diff", help="Weight difference"),
                "Current Shares": st.column_config.TextColumn("Current Shares"),
                "Target Shares": st.column_config.TextColumn("Target Shares"),
                "Action": st.column_config.TextColumn("Action"),
                "Trade Size": st.column_config.TextColumn("Trade Size"),
                "Est. Value": st.column_config.TextColumn("Est. Value", help="Estimated trade value"),
            },
        )

        # Summary of rebalancing
        buys = [r for r in recommendations if r.action == "BUY"]
        sells = [r for r in recommendations if r.action == "SELL"]

        if buys or sells:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if buys:
                    buy_total = sum(r.trade_value for r in buys)
                    st.success(f"💰 Total to Buy: ${buy_total:,.0f}")
            with col2:
                if sells:
                    sell_total = sum(abs(r.trade_value) for r in sells)
                    st.warning(f"💵 Total to Sell: ${sell_total:,.0f}")

    st.caption(
        "⚠️ **Disclaimer:** These recommendations are based on historical data and Modern Portfolio Theory. "
        "Past performance does not guarantee future results. Consider your investment goals, risk tolerance, "
        "and tax implications before rebalancing."
    )


def get_current_cash_balance() -> float:
    """Get current cash balance."""
    return get_cash_balance()


def render_admin_page():
    """
    Render Admin page for trading and cash operations.

    Allows users to:
    - Buy/Sell stocks (with cash validation)
    - Deposit/Withdraw cash
    - View recent activity
    """
    st.header("⚙️ Admin: Trading & Cash")
    st.warning("⚠️ **Write-enabled mode** - Operations will modify your portfolio data")

    # Get current cash balance
    cash_balance = get_current_cash_balance()

    # Get existing tickers for selectbox
    existing_tickers = get_all_tickers()

    # Display current cash prominently
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("💵 Available Cash", format_currency(cash_balance))

    st.divider()

    # Two-column layout: Trade Ticket | Cash Operations
    trade_col, cash_col = st.columns(2)

    # --- TRADE TICKET ---
    with trade_col:
        st.subheader("📈 Trade Ticket")

        with st.form("trade_form", clear_on_submit=True):
            ticker_selection = st.selectbox(
                "Ticker Symbol *",
                options=existing_tickers,
                index=None,
                placeholder="Select or enter ticker (e.g. AAPL)",
                help="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)",
            )
            ticker = ticker_selection.upper().strip() if ticker_selection else ""

            action = st.radio(
                "Action *",
                ["BUY", "SELL"],
                horizontal=True,
            )

            shares = st.number_input(
                "Shares *",
                min_value=0.01,
                value=1.0,
                step=1.0,
                format="%.2f",
                help="Number of shares to trade",
            )

            price = st.number_input(
                "Price per Share ($) *",
                min_value=0.01,
                value=100.0,
                step=0.01,
                format="%.2f",
                help="Execution price per share",
            )

            # Trade date and time inputs
            trade_at = st.datetime_input(
                "Trade Date & Time",
                value=datetime.now(),
                help="Date and time of trade execution",
            )

            fees = st.number_input(
                "Fees ($)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="Brokerage fees and commissions",
            )

            submitted = st.form_submit_button(
                "Execute",
                type="primary",
            )

            if submitted:
                if not ticker:
                    st.error("❌ Ticker symbol is required")
                else:
                    # Calculate trade cost
                    trade_cost = shares * price + fees

                    # Check cash sufficiency for BUY orders
                    if action == "BUY":
                        if trade_cost > cash_balance:
                            st.error(
                                f"❌ **Insufficient cash**\n\n"
                                f"Trade cost: ${trade_cost:,.2f}\n\n"
                                f"Available: ${cash_balance:,.2f}\n\n"
                                f"Shortfall: ${trade_cost - cash_balance:,.2f}\n\n"
                                f"⚠️ Negative cash balance is not allowed. Please deposit funds first."
                            )
                        else:
                            # Execute BUY
                            with st.spinner("Executing buy order..."):
                                result = buy_position(
                                    ticker=ticker,
                                    shares=shares,
                                    price=price,
                                    trade_at=trade_at,
                                    fees=fees,
                                )

                            if result.success:
                                st.success(
                                    f"✅ **BUY executed**\n\n"
                                    f"{shares:,.2f} shares of {ticker} @ ${price:,.2f}\n\n"
                                    f"Total cost: ${trade_cost:,.2f}\n\n"
                                    f"{result.status_message}"
                                )
                                new_balance = get_current_cash_balance()
                                st.info(f"💵 New cash balance: ${new_balance:,.2f}")
                                celebrate_and_rerun()
                            else:
                                st.error(f"❌ Trade failed: {', '.join(result.errors)}")

                    elif action == "SELL":
                        # Execute SELL (may create short position)
                        with st.spinner("Executing sell order..."):
                            result = sell_position(
                                ticker=ticker,
                                shares=shares,
                                price=price,
                                trade_at=trade_at,
                                fees=fees,
                            )

                        if result.success:
                            proceeds = shares * price - fees
                            st.success(
                                f"✅ **SELL executed**\n\n"
                                f"{shares:,.2f} shares of {ticker} @ ${price:,.2f}\n\n"
                                f"Net proceeds: ${proceeds:,.2f}\n\n"
                                f"{result.status_message}"
                            )
                            if result.realized_pnl and result.realized_pnl != 0:
                                pnl_emoji = "📈" if result.realized_pnl > 0 else "📉"
                                st.info(f"{pnl_emoji} Realized P&L: ${result.realized_pnl:+,.2f}")
                            new_balance = get_current_cash_balance()
                            st.info(f"💵 New cash balance: ${new_balance:,.2f}")
                            celebrate_and_rerun()
                        else:
                            st.error(f"❌ Trade failed: {', '.join(result.errors)}")

    # --- CASH OPERATIONS ---
    with cash_col:
        st.subheader("💵 Cash Operations")

        with st.form("cash_form", clear_on_submit=True):
            cash_action = st.radio(
                "Action *",
                ["DEPOSIT", "WITHDRAW"],
                horizontal=True,
            )

            amount = st.number_input(
                "Amount ($) *",
                min_value=0.01,
                value=1000.0,
                step=100.0,
                format="%.2f",
                help="Amount to deposit or withdraw",
            )

            cash_date = st.datetime_input(
                "Transaction Date & Time",
                value=datetime.now(),
                help="Date & time of cash transaction",
            )

            description = st.text_input(
                "Description",
                placeholder="Initial capital, personal expense, etc.",
                help="Optional note for this transaction",
            )

            cash_submitted = st.form_submit_button(
                "Execute",
                type="primary",
            )

            if cash_submitted:
                if cash_action == "DEPOSIT":
                    # Execute deposit
                    tx = deposit_cash(
                        amount=amount,
                        transaction_at=cash_date,
                        description=description or None,
                    )

                    new_balance = get_current_cash_balance()
                    st.success(
                        f"✅ **Deposited ${amount:,.2f}**\n\n"
                        f"Date & Time: {cash_date}\n\n"
                        f"💵 New balance: ${new_balance:,.2f}"
                    )
                    celebrate_and_rerun()

                elif cash_action == "WITHDRAW":
                    # Check sufficient balance
                    if amount > cash_balance:
                        st.error(
                            f"❌ **Insufficient cash**\n\n"
                            f"Withdrawal: ${amount:,.2f}\n\n"
                            f"Available: ${cash_balance:,.2f}\n\n"
                            f"Shortfall: ${amount - cash_balance:,.2f}\n\n"
                            f"⚠️ Negative cash balance is not allowed."
                        )
                    else:
                        # Execute withdrawal
                        tx = withdraw_cash(
                            amount=amount,
                            transaction_at=cash_date,
                            description=description or None,
                        )

                        new_balance = get_current_cash_balance()
                        st.success(
                            f"✅ **Withdrew ${amount:,.2f}**\n\n"
                            f"Date & Time: {cash_date}\n\n"
                            f"💵 New balance: ${new_balance:,.2f}"
                        )
                    celebrate_and_rerun()
    # --- RECENT ACTIVITY ---
    st.divider()
    st.subheader("📒 Recent Activity")

    activity_tabs = st.tabs(["Recent Trades", "Cash Ledger"])

    with activity_tabs[0]:
        # Show recent trades with inline actions
        recent_trades = get_recent_trades(limit=20)

        if recent_trades:
            # Get latest trade IDs per ticker (only these are editable)
            latest_trade_ids = get_latest_trade_ids_by_ticker()
            editable_trade_ids = set(latest_trade_ids.values())

            st.caption("💡 Only the latest trade per ticker can be edited or deleted.")

            # Define dialog functions at the module level (before the loop)
            @st.dialog("Edit Trade")
            def edit_trade_dialog(trade_row):
                with st.form("edit_trade_form"):
                    st.text_input("Ticker", value=trade_row["Ticker"], disabled=True)
                    st.text_input("Action", value=trade_row["Action"], disabled=True)
                    trade_at = st.datetime_input(
                        "Trade Date & Time",
                        value=trade_row["Date"],
                    )
                    shares = st.number_input(
                        "Shares",
                        min_value=0.01,
                        value=float(trade_row["Shares"]),
                        step=1.0,
                        format="%.2f",
                    )
                    price = st.number_input(
                        "Price",
                        min_value=0.01,
                        value=float(trade_row["Price"]),
                        step=0.01,
                        format="%.2f",
                    )
                    fees = st.number_input(
                        "Fees",
                        min_value=0.0,
                        value=float(trade_row["Fees"]),
                        step=0.01,
                        format="%.2f",
                    )

                    col_a, col_b = st.columns(2)
                    save = col_a.form_submit_button("Save Changes", type="primary")
                    cancel = col_b.form_submit_button("Cancel")

                if cancel:
                    st.rerun()

                if save:
                    shares_changed = shares != float(trade_row["Shares"])
                    price_changed = price != float(trade_row["Price"])
                    fees_changed = fees != float(trade_row["Fees"])
                    date_changed = trade_at != trade_row["Date"]

                    if not any([shares_changed, price_changed, fees_changed, date_changed]):
                        st.info("No changes detected")
                        return

                    result = update_trade(
                        trade_id=int(trade_row["trade_id"]),
                        shares=float(shares) if shares_changed else None,
                        price=float(price) if price_changed else None,
                        fees=float(fees) if fees_changed else None,
                        trade_at=trade_at if date_changed else None,
                    )

                    if result.success:
                        st.success(result.message)
                        celebrate_and_rerun(msg="Updating trade...", animation=None, delay=1)
                    else:
                        st.error(result.message)

            @st.dialog("Delete Trade")
            def delete_trade_dialog(trade_row):
                st.warning(
                    f"**Are you sure you want to delete this trade?**\n\n"
                    f"Trade #{int(trade_row['trade_id'])}: {trade_row['Ticker']} "
                    f"{trade_row['Action']} {trade_row['Shares']} shares @ ${trade_row['Price']:.2f}"
                )
                st.caption(f"Date: {trade_row['Date']}")

                confirm = st.text_input('Type "DELETE" to confirm', value="")
                col_a, col_b = st.columns(2)
                do_delete = col_a.button(
                    "Delete Trade",
                    type="primary",
                    disabled=(confirm != "DELETE"),
                )
                cancel = col_b.button("Cancel")

                if cancel:
                    st.rerun()

                if do_delete:
                    result = delete_trade(int(trade_row["trade_id"]))
                    if result.success:
                        st.success(result.message)
                        celebrate_and_rerun(msg="Deleting trade...", animation=None, delay=1)
                    else:
                        st.error(result.message)

            # Display trades as cards for better UX
            for trade in recent_trades:
                ticker = trade.asset.ticker if trade.asset else "?"
                trade_id = trade.id
                is_editable = trade_id in editable_trade_ids

                # Create a container for each trade
                with st.container(border=True):
                    # Use columns for layout: [Trade Info | Actions]
                    col_info, col_actions = st.columns([5, 1])

                    with col_info:
                        # Display trade details in a readable format
                        date_str = trade.trade_at.strftime("%Y-%m-%d %H:%M:%S")
                        action_emoji = "🟢" if trade.action.value == "BUY" else "🔴"

                        st.markdown(
                            f"**{action_emoji} {ticker}** / {trade.action.value} / "
                            f"{int(trade.shares)} shares @ \\${trade.price:.2f} / "
                            f"Fees: \\${trade.fees or 0:.2f} / "
                            f"P&L: \\${trade.realized_pnl or 0:.2f}"
                        )
                        st.caption(f"ID: {trade_id} · {date_str}")

                    with col_actions:
                        # Action buttons in a compact layout
                        if is_editable:
                            col_edit, col_delete = st.columns(2)
                            with col_edit:
                                if st.button("✏️", key=f"edit_{trade_id}", help="Edit trade"):
                                    edit_trade_dialog(
                                        {
                                            "trade_id": trade_id,
                                            "Date": trade.trade_at,
                                            "Ticker": ticker,
                                            "Action": trade.action.value,
                                            "Shares": int(trade.shares),
                                            "Price": trade.price,
                                            "Fees": trade.fees or 0.0,
                                        }
                                    )
                            with col_delete:
                                if st.button("🗑️", key=f"delete_{trade_id}", help="Delete trade"):
                                    delete_trade_dialog(
                                        {
                                            "trade_id": trade_id,
                                            "Date": trade.trade_at,
                                            "Ticker": ticker,
                                            "Action": trade.action.value,
                                            "Shares": int(trade.shares),
                                            "Price": trade.price,
                                            "Fees": trade.fees or 0.0,
                                        }
                                    )
                        else:
                            st.markdown(
                                "<div style='text-align: center; opacity: 0.3;'>🔒</div>",
                                unsafe_allow_html=True,
                            )

        else:
            st.info("No trades yet")

    with activity_tabs[1]:
        # Show cash ledger
        ledger = get_cash_ledger(limit=10)

        if ledger:
            ledger_data = []
            for entry in ledger:
                ledger_data.append(
                    {
                        "Date": entry["date"],
                        "Type": entry["type"],
                        "Amount": f"${entry['amount']:+,.2f}",
                        "Balance": f"${entry['balance']:,.2f}",
                        "Description": entry["description"][:40] if entry["description"] else "-",
                    }
                )

            st.dataframe(
                pd.DataFrame(ledger_data),
                hide_index=True,
            )
        else:
            st.info("No cash transactions yet")

    # --- BENCHMARK MANAGEMENT ---
    st.divider()
    st.subheader("📊 Benchmark Management")
    st.caption("Add or remove market indices for portfolio comparison")

    # Get current benchmarks
    current_indices = get_all_indices()

    bench_col1, bench_col2 = st.columns(2)

    with bench_col1:
        st.markdown("**Current Benchmarks**")
        if current_indices:
            for idx in current_indices:
                col_name, col_cat, col_del = st.columns([3, 2, 1])
                with col_name:
                    st.text(f"{idx.name} ({idx.symbol})")
                with col_cat:
                    st.caption(idx.category)
                with col_del:
                    if st.button("🗑️", key=f"del_bench_{idx.symbol}", help=f"Remove {idx.symbol}"):
                        if delete_index(idx.symbol):
                            st.success(f"Removed {idx.symbol}")
                            st.rerun()
                        else:
                            st.error(f"Failed to remove {idx.symbol}")
        else:
            st.info("No benchmarks configured")

    with bench_col2:
        st.markdown("**Add New Benchmark**")

        # Common indices to add
        common_indices = [
            ("SPX", "S&P 500", "^GSPC", "EQUITY"),
            ("RUT", "Russell 2000", "^RUT", "EQUITY"),
            ("IXIC", "NASDAQ Composite", "^IXIC", "EQUITY"),
            ("DJI", "Dow Jones Industrial", "^DJI", "EQUITY"),
            ("VIX", "CBOE Volatility Index", "^VIX", "VOLATILITY"),
        ]

        # Filter out already added indices
        existing_symbols = {idx.symbol for idx in current_indices}
        available_to_add = [
            (sym, name, yahoo, cat)
            for sym, name, yahoo, cat in common_indices
            if sym not in existing_symbols
        ]

        if available_to_add:
            with st.form("add_benchmark_form", clear_on_submit=True):
                selected_index = st.selectbox(
                    "Select Index",
                    options=[(sym, name) for sym, name, _, _ in available_to_add],
                    format_func=lambda x: f"{x[1]} ({x[0]})",
                    index=0,
                )

                add_bench_submitted = st.form_submit_button("Add Benchmark", type="primary")

                if add_bench_submitted and selected_index:
                    symbol, name = selected_index
                    # Find the full config
                    yahoo_symbol = None
                    category = "EQUITY"
                    for sym, nm, yh, cat in common_indices:
                        if sym == symbol:
                            yahoo_symbol = yh
                            category = cat
                            break

                    try:
                        create_index(
                            symbol=symbol,
                            name=name,
                            category=category,
                        )
                        # Sync prices for the new index
                        with st.spinner(f"Fetching price data for {symbol}..."):
                            sync_result = sync_index_prices(symbol=symbol, lookback_days=365)
                            if sync_result and sync_result[0].success:
                                st.success(
                                    f"Added {name} with {sync_result[0].records_added} price records"
                                )
                            else:
                                st.warning(
                                    f"Added {name} but price sync failed. Run update job to fetch prices."
                                )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to add benchmark: {e}")
        else:
            st.info("All common indices have been added")

        # Custom index input
        with st.expander("➕ Add Custom Index"):
            with st.form("custom_benchmark_form", clear_on_submit=True):
                custom_symbol = st.text_input(
                    "Symbol",
                    placeholder="e.g., FTSE",
                    help="Internal symbol for the index",
                )
                custom_name = st.text_input(
                    "Name",
                    placeholder="e.g., FTSE 100",
                )
                custom_yahoo = st.text_input(
                    "Yahoo Finance Symbol",
                    placeholder="e.g., ^FTSE",
                    help="Symbol used to fetch prices from Yahoo Finance",
                )
                custom_category = st.selectbox(
                    "Category",
                    options=["EQUITY", "VOLATILITY", "COMMODITY", "BOND", "CURRENCY"],
                    index=0,
                )

                custom_submitted = st.form_submit_button("Add Custom Index")

                if custom_submitted:
                    if custom_symbol and custom_name:
                        try:
                            create_index(
                                symbol=custom_symbol.upper().strip(),
                                name=custom_name.strip(),
                                category=custom_category,
                            )
                            st.success(f"Added {custom_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to add: {e}")
                    else:
                        st.error("Symbol and Name are required")


def render_watchlist_page():
    """
    Render Watchlist & Valuation page with Yahoo-style metrics.

    Shows:
    - Yahoo-aligned Valuation Measures table (read-only)
    - Yahoo-aligned Financial Highlights table (read-only)
    - BUY/WAIT/AVOID signals
    - Multi-row selection for bulk delete
    - Tag filtering with pills
    """
    from services.asset_service import delete_assets

    st.header("👀 Watchlist & Valuation")

    # Get all tags for filtering
    tag_result = get_all_tags_with_counts()
    all_tag_names = [t.tag.name for t in tag_result.tags] if tag_result.tags else []

    # Tag filter using pills
    if all_tag_names:
        st.caption("🏷️ Filter by Tags:")
        selected_tags = st.pills(
            "Tag Filter",
            options=all_tag_names,
            selection_mode="multi",
            default=None,
            label_visibility="collapsed",
        )
        st.divider()
    else:
        selected_tags = []

    try:
        valuation_df = run_valuation()
    except Exception as e:
        st.error(f"Error loading valuation data: {e}")
        return

    if valuation_df.empty:
        st.info("No valuation data available.")
        st.caption("Run the valuation fetcher to populate this view.")
        return

    # Filter by selected tags if any
    if selected_tags:
        filtered_assets = get_assets_by_tag_names(selected_tags)
        filtered_tickers = {asset.ticker for asset in filtered_assets}
        valuation_df = valuation_df[valuation_df["ticker"].isin(filtered_tickers)]
        
        if valuation_df.empty:
            st.info(f"No assets found with selected tags: {', '.join(selected_tags)}")
            return

    # Show "As of" date from the most recent updated_at
    if "updated_at" in valuation_df.columns:
        latest_update = valuation_df["updated_at"].dropna().max()
        if pd.notna(latest_update):
            if hasattr(latest_update, "strftime"):
                as_of_date = latest_update.strftime("%m/%d/%Y")
            else:
                as_of_date = str(latest_update)[:10]
            st.caption(f"📅 Data as of: {as_of_date}")

    # ====== VALUATION MEASURES TABLE ======
    st.subheader("📊 Valuation Measures")

    # Create display dataframe for valuation measures
    vm_df = valuation_df[["ticker"]].copy()

    # Large number fields for VM
    vm_large_fields = ["market_cap", "enterprise_value"]

    # Add valuation measure columns with formatting
    for col in [
        "market_cap",
        "enterprise_value",
        "pe_trailing",
        "pe_forward",
        "peg",
        "price_to_sales",
        "price_to_book",
        "ev_to_revenue",
        "ev_ebitda",
    ]:
        if col in valuation_df.columns:
            if col in vm_large_fields:
                vm_df[col] = valuation_df[col].apply(_format_large_number)
            else:
                vm_df[col] = valuation_df[col]
        else:
            vm_df[col] = None

    # Apply colored emojis to Signal column
    def format_signal(val):
        if val == "BUY":
            return "🟢 BUY"
        elif val == "WAIT":
            return "🟡 WAIT"
        elif val == "AVOID":
            return "🔴 AVOID"
        return val

    vm_df["valuation_action"] = valuation_df["valuation_action"].apply(format_signal)

    # Display with row selection
    vm_event = st.dataframe(
        vm_df,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "market_cap": st.column_config.TextColumn(
                "Market Cap",
                help="Market Capitalization = Share Price × Shares Outstanding",
            ),
            "enterprise_value": st.column_config.TextColumn(
                "Enterprise Value",
                help="EV = Market Cap + Debt - Cash",
            ),
            "pe_trailing": st.column_config.NumberColumn(
                "Trailing P/E",
                format="%.2f",
                help="Trailing Price-to-Earnings: market price / TTM EPS",
            ),
            "pe_forward": st.column_config.NumberColumn(
                "Forward P/E",
                format="%.2f",
                help="Forward Price-to-Earnings: market price / estimated next-year EPS",
            ),
            "peg": st.column_config.NumberColumn(
                "PEG Ratio",
                format="%.2f",
                help="P/E-to-Growth: (P/E) / EPS growth % (< 1 = undervalued)",
            ),
            "price_to_sales": st.column_config.NumberColumn(
                "Price/Sales",
                format="%.2f",
                help="Price-to-Sales ratio (trailing 12 months)",
            ),
            "price_to_book": st.column_config.NumberColumn(
                "Price/Book",
                format="%.2f",
                help="Price-to-Book ratio (most recent quarter)",
            ),
            "ev_to_revenue": st.column_config.NumberColumn(
                "EV/Revenue", format="%.2f", help="Enterprise Value / Revenue"
            ),
            "ev_ebitda": st.column_config.NumberColumn(
                "EV/EBITDA", format="%.2f", help="Enterprise Value / EBITDA"
            ),
            "valuation_action": st.column_config.TextColumn(
                "Signal",
                help="BUY = attractive valuation | WAIT = mixed | AVOID = overvalued",
            ),
        },
        key="valuation_measures_table",
    )

    # ====== BULK DELETE SELECTED ROWS ======
    selected_rows = vm_event.selection.rows
    if selected_rows:
        st.divider()
        st.subheader("🗑️ Delete Selected Assets")

        # Get selected tickers
        selected_tickers = [vm_df.iloc[idx]["ticker"] for idx in selected_rows]

        st.warning(
            f"⚠️ You have selected {len(selected_tickers)} asset(s): {', '.join(selected_tickers)}"
        )
        st.caption(
            "This will delete these watchlist assets and all related data (prices, fundamentals, valuation metrics, notes)."
        )
        st.caption("Assets with active positions or trade history will be blocked by default.")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Delete Selected", type="primary", key="delete_selected_btn"):
                with st.spinner("Deleting assets..."):
                    delete_result = delete_assets(
                        tickers=selected_tickers,
                        allow_owned=True,
                        allow_with_trades=False,
                        allow_with_active_position=False,
                    )

                # Display results
                if delete_result.deleted:
                    st.success(
                        f"✅ Deleted {len(delete_result.deleted)} asset(s): {', '.join(delete_result.deleted)}"
                    )

                if delete_result.blocked:
                    st.error("❌ Blocked deletions:")
                    for block in delete_result.blocked:
                        st.error(f"  • {block['ticker']}: {block['reason']}")

                if delete_result.not_found:
                    st.warning(f"⚠️ Not found: {', '.join(delete_result.not_found)}")

                if delete_result.errors:
                    st.error("❌ Errors:")
                    for err in delete_result.errors:
                        st.error(f"  • {err['ticker']}: {err['error']}")

                time.sleep(1)
                st.rerun()

    st.divider()

    # ====== FINANCIAL HIGHLIGHTS TABLE ======
    st.subheader("💰 Financial Highlights")

    # Create display dataframe for financial highlights
    fh_df = valuation_df[["ticker"]].copy()

    # Large number fields for FH
    fh_large_fields = [
        "revenue_ttm",
        "net_income_ttm",
        "total_cash",
        "levered_free_cash_flow",
    ]

    # Profitability metrics (stored as decimals, display as %)
    for col in ["profit_margin", "return_on_assets", "return_on_equity"]:
        if col in valuation_df.columns:
            fh_df[col] = valuation_df[col].apply(lambda x: x * 100 if pd.notna(x) else None)
        else:
            fh_df[col] = None

    # Income statement metrics (large numbers)
    for col in ["revenue_ttm", "net_income_ttm", "diluted_eps_ttm"]:
        if col in valuation_df.columns:
            if col in fh_large_fields:
                fh_df[col] = valuation_df[col].apply(_format_large_number)
            else:
                fh_df[col] = valuation_df[col]
        else:
            fh_df[col] = None

    # Balance sheet & cash flow
    for col in ["total_cash", "total_debt_to_equity", "levered_free_cash_flow"]:
        if col in valuation_df.columns:
            if col == "total_debt_to_equity":
                # D/E is already a percentage in yfinance
                fh_df[col] = valuation_df[col]
            elif col in fh_large_fields:
                fh_df[col] = valuation_df[col].apply(_format_large_number)
            else:
                fh_df[col] = valuation_df[col]
        else:
            fh_df[col] = None

    st.dataframe(
        fh_df,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            # Profitability
            "profit_margin": st.column_config.NumberColumn(
                "Profit Margin",
                format="%.2f%%",
                help="Net Income / Revenue (trailing 12 months)",
            ),
            "return_on_assets": st.column_config.NumberColumn(
                "ROA (ttm)",
                format="%.2f%%",
                help="Return on Assets (trailing 12 months)",
            ),
            "return_on_equity": st.column_config.NumberColumn(
                "ROE (ttm)",
                format="%.2f%%",
                help="Return on Equity (trailing 12 months)",
            ),
            # Income Statement
            "revenue_ttm": st.column_config.TextColumn(
                "Revenue (ttm)",
                help="Total Revenue (trailing 12 months)",
            ),
            "net_income_ttm": st.column_config.TextColumn(
                "Net Income (ttm)",
                help="Net Income Available to Common (trailing 12 months)",
            ),
            "diluted_eps_ttm": st.column_config.NumberColumn(
                "Diluted EPS",
                format="%.2f",
                help="Diluted Earnings Per Share (trailing 12 months)",
            ),
            # Balance Sheet & Cash Flow
            "total_cash": st.column_config.TextColumn(
                "Total Cash (mrq)",
                help="Total Cash (most recent quarter)",
            ),
            "total_debt_to_equity": st.column_config.NumberColumn(
                "Debt/Equity (mrq)",
                format="%.2f%%",
                help="Total Debt / Equity (most recent quarter)",
            ),
            "levered_free_cash_flow": st.column_config.TextColumn(
                "Levered FCF (ttm)",
                help="Levered Free Cash Flow (trailing 12 months)",
            ),
        },
        key="financial_highlights_table",
    )

    # Legend
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.markdown("🟢 **BUY** — Multiple metrics suggest attractive valuation")
    col2.markdown("🟡 **WAIT** — Mixed signals, patience recommended")
    col3.markdown("🔴 **AVOID** — Multiple metrics suggest overvaluation")

    st.caption(
        "💡 Read-only view of valuation data fetched from Yahoo Finance. "
        "Select rows in the Valuation Measures table to delete watchlist assets. "
        "Edit valuations and overrides in the Admin page."
    )


def _format_large_number(value: float | None) -> str:
    """Format large numbers in B/T notation like Yahoo Finance."""
    if value is None or pd.isna(value):
        return "—"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if abs_val >= 1e12:
        return f"{sign}{abs_val / 1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.2f}K"
    else:
        return f"{sign}{abs_val:.2f}"


# ============================================================================
# CHARTS PAGE
# ============================================================================

@st.cache_data(ttl=3600)
def _fetch_ohlcv_data(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker from the database.
    
    Falls back to yfinance direct fetch if not in database.
    """
    from db import get_db
    from db.repositories import AssetRepository, PriceRepository, MarketIndexRepository, IndexPriceRepository
    import yfinance as yf
    
    db = get_db()
    
    with db.session() as session:
        # Try to find as asset first
        asset_repo = AssetRepository(session)
        price_repo = PriceRepository(session)
        index_repo = MarketIndexRepository(session)
        index_price_repo = IndexPriceRepository(session)
        
        asset = asset_repo.get_by_ticker(ticker)
        
        if asset:
            # Fetch from prices_daily table
            prices = price_repo.get_price_history(asset.id, start_date=start_date)
            if prices:
                records = [
                    {
                        "date": p.date,
                        "open": p.open,
                        "high": p.high,
                        "low": p.low,
                        "close": p.close,
                        "volume": p.volume,
                    }
                    for p in prices
                ]
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date").sort_index()
        
        # Try as market index
        index = index_repo.get_by_symbol(ticker)
        if index:
            prices = index_price_repo.get_price_history(index.id, start_date=start_date)
            if prices:
                records = [
                    {
                        "date": p.date,
                        "open": p.open if hasattr(p, 'open') else p.close,
                        "high": p.high if hasattr(p, 'high') else p.close,
                        "low": p.low if hasattr(p, 'low') else p.close,
                        "close": p.close,
                        "volume": p.volume if hasattr(p, 'volume') else 0,
                    }
                    for p in prices
                ]
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date").sort_index()
    
    # Fallback: fetch directly from yfinance
    try:
        data = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=False,
            timeout=10,
        )
        if not data.empty:
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            return df.sort_index()
    except Exception:
        pass
    
    return pd.DataFrame()


def _get_available_tickers() -> dict[str, list[str]]:
    """
    Get categorized tickers (owned + watchlist + indices).
    
    Returns:
        Dict with keys 'owned', 'watchlist', 'indices' mapping to ticker lists.
    """
    from db import get_db
    from db.repositories import AssetRepository, MarketIndexRepository
    
    result = {
        "owned": [],
        "watchlist": [],
        "indices": [],
    }
    db = get_db()
    
    with db.session() as session:
        asset_repo = AssetRepository(session)
        index_repo = MarketIndexRepository(session)
        
        # Get owned assets with names
        owned = asset_repo.get_by_status(AssetStatus.OWNED)
        result["owned"] = sorted(
            [(a.ticker, a.name or a.ticker) for a in owned],
            key=lambda x: x[0]
        )
        
        # Get watchlist assets with names
        watchlist = asset_repo.get_by_status(AssetStatus.WATCHLIST)
        result["watchlist"] = sorted(
            [(a.ticker, a.name or a.ticker) for a in watchlist],
            key=lambda x: x[0]
        )
        
        # Get market indices with names
        indices = index_repo.get_all()
        result["indices"] = sorted(
            [(idx.symbol, idx.name or idx.symbol) for idx in indices],
            key=lambda x: x[0]
        )
    
    return result


def _flatten_tickers(categorized: dict[str, list[tuple[str, str]]]) -> list[str]:
    """Flatten categorized tickers into a single list of ticker symbols."""
    all_tickers = []
    for ticker, _ in categorized.get("owned", []):
        all_tickers.append(ticker)
    for ticker, _ in categorized.get("watchlist", []):
        all_tickers.append(ticker)
    for ticker, _ in categorized.get("indices", []):
        all_tickers.append(ticker)
    return all_tickers


def _calculate_timeframe_start(timeframe: str) -> str:
    """Calculate start date based on selected timeframe."""
    today = datetime.now()
    
    if timeframe == "1D":
        # For intraday, we still fetch daily data but just last day
        start = today - timedelta(days=5)
    elif timeframe == "1W":
        start = today - timedelta(weeks=1)
    elif timeframe == "1M":
        start = today - timedelta(days=30)
    elif timeframe == "3M":
        start = today - timedelta(days=90)
    elif timeframe == "6M":
        start = today - timedelta(days=180)
    elif timeframe == "1Y":
        start = today - timedelta(days=365)
    elif timeframe == "2Y":
        start = today - timedelta(days=730)
    else:  # "All"
        start = today - timedelta(days=365 * 10)  # 10 years max
    
    return start.strftime("%Y-%m-%d")


def _build_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    ticker_name: str,
    show_sma: list[int],
    show_ema: list[int],
    show_bollinger: bool,
    show_volume: bool,
    show_rsi: bool,
    show_macd: bool,
) -> go.Figure:
    """
    Build interactive candlestick chart with technical indicators.
    
    Returns a Plotly figure with:
    - Main candlestick chart with overlays (SMA, EMA, Bollinger)
    - Volume subplot
    - RSI subplot (if enabled)
    - MACD subplot (if enabled)
    """
    # Determine number of rows for subplots
    num_rows = 1  # Main chart
    if show_volume:
        num_rows += 1
    if show_rsi:
        num_rows += 1
    if show_macd:
        num_rows += 1
    
    # Calculate row heights
    row_heights = [0.5]  # Main chart gets more space
    if show_volume:
        row_heights.append(0.1)
    if show_rsi:
        row_heights.append(0.15)
    if show_macd:
        row_heights.append(0.15)
    
    # Normalize heights
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]
    
    # Create subplots - use ticker and name in title
    chart_title = f"{ticker}" if ticker == ticker_name else f"{ticker}"
    subplot_titles = [f"{chart_title} Price"]
    if show_volume:
        subplot_titles.append("Volume")
    if show_rsi:
        subplot_titles.append("RSI (14)")
    if show_macd:
        subplot_titles.append("MACD (12,26,9)")
    
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )
    
    # Format dates as strings to use with category x-axis (skips non-trading days)
    date_strings = df.index.strftime("%Y-%m-%d")
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=date_strings,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )
    
    # SMA overlays
    sma_colors = {20: "#ffa726", 50: "#42a5f5", 200: "#ab47bc"}
    for period in show_sma:
        if len(df) >= period:
            sma = calculate_sma(df["close"], period)
            fig.add_trace(
                go.Scatter(
                    x=date_strings,
                    y=sma,
                    name=f"SMA {period}",
                    line=dict(color=sma_colors.get(period, "#888888"), width=1.5),
                    hovertemplate=f"SMA {period}: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    
    # EMA overlays
    ema_colors = {12: "#66bb6a", 26: "#ec407a"}
    for period in show_ema:
        if len(df) >= period:
            ema = calculate_ema(df["close"], period)
            fig.add_trace(
                go.Scatter(
                    x=date_strings,
                    y=ema,
                    name=f"EMA {period}",
                    line=dict(color=ema_colors.get(period, "#888888"), width=1.5, dash="dot"),
                    hovertemplate=f"EMA {period}: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    
    # Bollinger Bands
    if show_bollinger and len(df) >= 20:
        upper, middle, lower = calculate_bollinger_bands(df["close"])
        
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=upper,
                name="BB Upper",
                line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
                hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        
        # Lower band
        fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=lower,
                name="BB Lower",
                line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.1)",
                hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    
    current_row = 2
    
    # Volume subplot
    if show_volume:
        # Color volume bars based on price direction
        colors = [
            "#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
            for i in range(len(df))
        ]
        
        fig.add_trace(
            go.Bar(
                x=date_strings,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )
        current_row += 1
    
    # RSI subplot
    if show_rsi and len(df) >= 14:
        rsi = calculate_rsi(df["close"])
        
        fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=rsi,
                name="RSI",
                line=dict(color="#7e57c2", width=1.5),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=current_row, col=1)
        
        # Set RSI y-axis range
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1
    
    # MACD subplot
    if show_macd and len(df) >= 26:
        macd_line, signal_line, histogram = calculate_macd(df["close"])
        
        # MACD histogram
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]
        fig.add_trace(
            go.Bar(
                x=date_strings,
                y=histogram,
                name="MACD Histogram",
                marker_color=hist_colors,
                opacity=0.5,
                hovertemplate="Histogram: %{y:.3f}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=macd_line,
                name="MACD",
                line=dict(color="#42a5f5", width=1.5),
                hovertemplate="MACD: %{y:.3f}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=date_strings,
                y=signal_line,
                name="Signal",
                line=dict(color="#ffa726", width=1.5),
                hovertemplate="Signal: %{y:.3f}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    
    # Use category type for x-axis to skip non-trading days (weekends + holidays)
    # This eliminates gaps in the chart by only showing dates with actual data
    fig.update_xaxes(
        type="category",
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        tickangle=-45,
        nticks=20,  # Limit number of date labels for readability
    )
    
    # Update y-axes
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
    )
    
    return fig


def render_charts_page():
    """
    Render Technical Charts page.
    
    Shows:
    - Ticker selector (owned + watchlist + indices)
    - Timeframe selector
    - Indicator toggles
    - Interactive candlestick chart with overlays and subplots
    """
    st.header("📈 Technical Charts")
    
    # Initialize session state for chart preferences
    if "chart_ticker" not in st.session_state:
        st.session_state.chart_ticker = None
    if "chart_timeframe" not in st.session_state:
        st.session_state.chart_timeframe = "6M"
    if "chart_sma" not in st.session_state:
        st.session_state.chart_sma = [20, 50]
    if "chart_ema" not in st.session_state:
        st.session_state.chart_ema = []
    if "chart_bollinger" not in st.session_state:
        st.session_state.chart_bollinger = False
    if "chart_volume" not in st.session_state:
        st.session_state.chart_volume = True
    if "chart_rsi" not in st.session_state:
        st.session_state.chart_rsi = True
    if "chart_macd" not in st.session_state:
        st.session_state.chart_macd = True
    
    try:
        categorized_tickers = _get_available_tickers()
        all_tickers = _flatten_tickers(categorized_tickers)
        
        if not all_tickers:
            st.warning("No tickers available. Add assets to your portfolio or watchlist first.")
            return
        
        # Set default ticker if not set - prefer SPX (S&P 500) if available
        if st.session_state.chart_ticker is None or st.session_state.chart_ticker not in all_tickers:
            if "SPX" in all_tickers:
                st.session_state.chart_ticker = "SPX"
            else:
                st.session_state.chart_ticker = all_tickers[0]
        
        # Build display options with category labels and names
        ticker_options = []
        ticker_to_display = {}
        ticker_to_name = {}
        display_to_ticker = {}
        
        # Add owned assets (tuple of ticker, name)
        for ticker, name in categorized_tickers.get("owned", []):
            display = f"📈 {ticker} — {name}" if name != ticker else f"📈 {ticker}"
            ticker_options.append(display)
            ticker_to_display[ticker] = display
            ticker_to_name[ticker] = name
            display_to_ticker[display] = ticker
        
        # Add watchlist assets
        for ticker, name in categorized_tickers.get("watchlist", []):
            display = f"👁 {ticker} — {name}" if name != ticker else f"👁 {ticker}"
            ticker_options.append(display)
            ticker_to_display[ticker] = display
            ticker_to_name[ticker] = name
            display_to_ticker[display] = ticker
        
        # Add indices
        for ticker, name in categorized_tickers.get("indices", []):
            display = f"📊 {ticker} — {name}" if name != ticker else f"📊 {ticker}"
            ticker_options.append(display)
            ticker_to_display[ticker] = display
            ticker_to_name[ticker] = name
            display_to_ticker[display] = ticker
        
        # Get current display value
        current_display = ticker_to_display.get(
            st.session_state.chart_ticker, 
            ticker_options[0] if ticker_options else None
        )
        
        # Controls row
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Ticker selector with categorized display
            selected_display = st.selectbox(
                "Ticker",
                options=ticker_options,
                index=ticker_options.index(current_display) if current_display in ticker_options else 0,
                placeholder="Select Ticker ...",
                key="ticker_select",
                help="📈 Owned | 👁 Watchlist | 📊 Index",
            )
            selected_ticker = display_to_ticker.get(selected_display, all_tickers[0])
            selected_name = ticker_to_name.get(selected_ticker, selected_ticker)
            st.session_state.chart_ticker = selected_ticker
        
        with col2:
            # Timeframe pills
            timeframes = ["1W", "1M", "3M", "6M", "1Y", "2Y", "All"]
            selected_timeframe = st.pills(
                "Timeframe",
                options=timeframes,
                default=st.session_state.chart_timeframe,
                key="timeframe_pills",
            )
            if selected_timeframe:
                st.session_state.chart_timeframe = selected_timeframe
        
        # Indicator controls in expander
        with st.expander("📊 Indicator Settings", expanded=False):
            ind_col1, ind_col2, ind_col3 = st.columns(3)
            
            with ind_col1:
                st.markdown("**Moving Averages**")
                sma_options = st.multiselect(
                    "SMA Periods",
                    options=[20, 50, 200],
                    default=st.session_state.chart_sma,
                    key="sma_select",
                )
                st.session_state.chart_sma = sma_options
                
                ema_options = st.multiselect(
                    "EMA Periods",
                    options=[12, 26],
                    default=st.session_state.chart_ema,
                    key="ema_select",
                )
                st.session_state.chart_ema = ema_options
            
            with ind_col2:
                st.markdown("**Bands & Volume**")
                show_bollinger = st.checkbox(
                    "Bollinger Bands (20,2)",
                    value=st.session_state.chart_bollinger,
                    key="bollinger_check",
                )
                st.session_state.chart_bollinger = show_bollinger
                
                show_volume = st.checkbox(
                    "Volume",
                    value=st.session_state.chart_volume,
                    key="volume_check",
                )
                st.session_state.chart_volume = show_volume
            
            with ind_col3:
                st.markdown("**Oscillators**")
                show_rsi = st.checkbox(
                    "RSI (14)",
                    value=st.session_state.chart_rsi,
                    key="rsi_check",
                )
                st.session_state.chart_rsi = show_rsi
                
                show_macd = st.checkbox(
                    "MACD (12,26,9)",
                    value=st.session_state.chart_macd,
                    key="macd_check",
                )
                st.session_state.chart_macd = show_macd
        
        # Calculate start date based on timeframe
        start_date = _calculate_timeframe_start(st.session_state.chart_timeframe)
        
        # Fetch data
        with st.spinner(f"Loading {selected_ticker} data..."):
            df = _fetch_ohlcv_data(selected_ticker, start_date)
        
        if df.empty:
            st.error(f"No price data available for {selected_ticker}. Try syncing prices in the Admin page.")
            return
        
        # Display asset/index name as subheader
        st.subheader(f"{selected_ticker} — {selected_name}")
        
        # Display current price info
        if len(df) >= 2:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            price_change = latest["close"] - prev["close"]
            pct_change = (price_change / prev["close"]) * 100
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            
            with info_col1:
                st.metric(
                    "Close",
                    f"${latest['close']:.2f}",
                    f"{price_change:+.2f} ({pct_change:+.2f}%)",
                )
            with info_col2:
                st.metric("Open", f"${latest['open']:.2f}")
            with info_col3:
                st.metric("High", f"${latest['high']:.2f}")
            with info_col4:
                st.metric("Low", f"${latest['low']:.2f}")
        
        # Build and display chart
        fig = _build_candlestick_chart(
            df=df,
            ticker=selected_ticker,
            ticker_name=selected_name,
            show_sma=st.session_state.chart_sma,
            show_ema=st.session_state.chart_ema,
            show_bollinger=st.session_state.chart_bollinger,
            show_volume=st.session_state.chart_volume,
            show_rsi=st.session_state.chart_rsi,
            show_macd=st.session_state.chart_macd,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        st.caption(
            f"📅 Showing {len(df)} trading days from {df.index[0].strftime('%Y-%m-%d')} "
            f"to {df.index[-1].strftime('%Y-%m-%d')}"
        )
        
    except Exception as e:
        st.error(f"Error loading chart: {e}")
        import traceback
        st.caption(traceback.format_exc())


def render_notes_page():
    """
    Render Notes page for investment journal and research notes.

    Shows:
    - Recent notes across all targets
    - Filter by target type (Asset, Market, Journal)
    - Create new notes (when admin UI enabled)
    - View/edit existing notes
    """
    st.header("📝 Investment Notes")

    # Get existing tickers and market symbols for filters
    existing_tickers = get_all_tickers()
    market_symbols = get_market_symbols()

    # Tabs: Recent | By Asset | By Market | Create
    if config.ui.enable_admin_ui:
        tabs = st.tabs(["Recent Notes", "By Asset", "By Market", "Create Note"])
    else:
        tabs = st.tabs(["Recent Notes", "By Asset", "By Market"])

    # --- RECENT NOTES TAB ---
    with tabs[0]:
        st.subheader("🕐 Recent Notes")

        recent_notes = get_recent_notes(limit=20)

        if recent_notes:
            for note in recent_notes:
                _render_note_card(note, show_target=True)
        else:
            st.info("No notes yet. Create your first note to get started!")

    # --- BY ASSET TAB ---
    with tabs[1]:
        st.subheader("📈 Notes by Asset")

        if existing_tickers:
            selected_ticker = st.selectbox(
                "Select Asset",
                options=existing_tickers,
                index=None,
                placeholder="Choose a ticker...",
                key="notes_asset_filter",
            )

            if selected_ticker:
                asset_notes = get_notes_for_asset(selected_ticker)

                if asset_notes:
                    for note in asset_notes:
                        _render_note_card(note)
                else:
                    st.info(f"No notes for {selected_ticker} yet.")
        else:
            st.info("No assets in portfolio. Add assets first.")

    # --- BY MARKET TAB ---
    with tabs[2]:
        st.subheader("🌍 Market Notes")

        # Common market symbols
        common_markets = [
            {"symbol": "^GSPC", "name": "S&P 500"},
            {"symbol": "^DJI", "name": "Dow Jones"},
            {"symbol": "^IXIC", "name": "NASDAQ"},
            {"symbol": "^VIX", "name": "VIX"},
            {"symbol": "^TNX", "name": "10Y Treasury"},
        ]

        # Combine with existing market symbols
        all_markets = {m["symbol"]: m["name"] for m in common_markets}
        for m in market_symbols:
            all_markets[m["symbol"]] = m["name"]

        market_options = [f"{sym} ({name})" for sym, name in all_markets.items()]

        selected_market = st.selectbox(
            "Select Market/Index",
            options=market_options,
            index=None,
            placeholder="Choose a market symbol...",
            key="notes_market_filter",
        )

        if selected_market:
            # Extract symbol from selection
            symbol = selected_market.split(" (")[0]

            # Get notes for this market symbol
            from db import get_db
            from db.repositories import NoteRepository, NoteTargetRepository

            db = get_db()
            with db.session() as session:
                target_repo = NoteTargetRepository(session)
                note_repo = NoteRepository(session)

                # Find target for this symbol
                from db.models import NoteTargetKind
                from sqlalchemy import select
                from db.models import NoteTarget

                stmt = select(NoteTarget).where(
                    NoteTarget.kind == NoteTargetKind.MARKET,
                    NoteTarget.symbol == symbol,
                )
                target = session.scalar(stmt)

                if target:
                    market_notes = note_repo.list_by_target(target.id)
                    if market_notes:
                        for note in market_notes:
                            _render_note_card(note)
                    else:
                        st.info(f"No notes for {symbol} yet.")
                else:
                    st.info(f"No notes for {symbol} yet.")

    # --- CREATE NOTE TAB (Admin only) ---
    if config.ui.enable_admin_ui and len(tabs) > 3:
        with tabs[3]:
            st.subheader("✏️ Create New Note")

            with st.form("create_note_form", clear_on_submit=True):
                # Target selection
                target_type = st.radio(
                    "Note Target",
                    ["Asset", "Market/Index", "Journal"],
                    horizontal=True,
                )

                # Target-specific fields
                target_ticker = None
                target_symbol = None
                target_symbol_name = None

                if target_type == "Asset":
                    target_ticker = st.selectbox(
                        "Select Asset",
                        options=existing_tickers,
                        index=None,
                        placeholder="Choose a ticker...",
                        accept_new_options=True,
                    )
                elif target_type == "Market/Index":
                    col1, col2 = st.columns(2)
                    with col1:
                        target_symbol = st.text_input(
                            "Symbol",
                            placeholder="^GSPC, ^VIX, etc.",
                        )
                    with col2:
                        target_symbol_name = st.text_input(
                            "Display Name (optional)",
                            placeholder="S&P 500, VIX, etc.",
                        )

                # Note type
                note_type_options = [t.value for t in NoteType]
                selected_type = st.selectbox(
                    "Note Type",
                    options=note_type_options,
                    index=note_type_options.index("JOURNAL"),
                )

                # Note content
                title = st.text_input(
                    "Title (optional)",
                    placeholder="Brief title for the note",
                )

                summary = st.text_area(
                    "Summary (optional)",
                    placeholder="Brief summary for table display (max 500 chars)",
                    max_chars=500,
                    height=80,
                )

                key_points = st.text_area(
                    "Key Points (optional)",
                    placeholder="Key takeaways, one per line",
                    height=80,
                )

                body_md = st.text_area(
                    "Note Content *",
                    placeholder="Full note content in Markdown...",
                    height=200,
                )

                tags = st.text_input(
                    "Tags (optional)",
                    placeholder="Comma-separated: bullish, earnings, risk",
                )

                submitted = st.form_submit_button("Create Note", type="primary")

                if submitted:
                    if not body_md.strip():
                        st.error("Note content is required")
                    else:
                        note_type = NoteType(selected_type)

                        if target_type == "Asset":
                            if not target_ticker:
                                st.error("Please select an asset")
                            else:
                                result = create_note_for_asset(
                                    ticker=target_ticker.upper().strip(),
                                    body_md=body_md,
                                    note_type=note_type,
                                    title=title or None,
                                    summary=summary or None,
                                    key_points=key_points or None,
                                    tags=tags or None,
                                )
                                if result.success:
                                    st.success(result.status_message)
                                    celebrate_and_rerun("Saving note...", delay=1, animation=None)
                                else:
                                    st.error(result.status_message)

                        elif target_type == "Market/Index":
                            if not target_symbol:
                                st.error("Please enter a market symbol")
                            else:
                                result = create_market_note(
                                    symbol=target_symbol.upper().strip(),
                                    body_md=body_md,
                                    name=target_symbol_name or None,
                                    note_type=note_type,
                                    title=title or None,
                                    summary=summary or None,
                                    key_points=key_points or None,
                                    tags=tags or None,
                                )
                                if result.success:
                                    st.success(result.status_message)
                                    celebrate_and_rerun("Saving note...", delay=1, animation=None)
                                else:
                                    st.error(result.status_message)

                        else:  # Journal
                            result = create_journal_entry(
                                body_md=body_md,
                                note_type=note_type,
                                title=title or None,
                                summary=summary or None,
                                key_points=key_points or None,
                                tags=tags or None,
                            )
                            if result.success:
                                st.success(result.status_message)
                                celebrate_and_rerun("Saving note...", delay=1, animation=None)
                            else:
                                st.error(result.status_message)


def _render_note_card(note, show_target: bool = False):
    """Render a single note as an expandable card."""
    # Build header
    type_emoji = {
        NoteType.JOURNAL: "📓",
        NoteType.THESIS: "💡",
        NoteType.RISK: "⚠️",
        NoteType.TRADE_PLAN: "📋",
        NoteType.TRADE_REVIEW: "🔍",
        NoteType.MARKET_VIEW: "🌍",
        NoteType.EARNINGS: "📊",
        NoteType.NEWS: "📰",
        NoteType.OTHER: "📝",
    }.get(note.note_type, "📝")

    # Format date
    created_str = note.created_at.strftime("%Y-%m-%d %H:%M") if note.created_at else ""

    # Build title
    display_title = note.title or note.summary or f"{note.note_type.value} Note"
    if note.pinned:
        display_title = f"📌 {display_title}"

    # Target info
    target_info = ""
    if show_target and note.target:
        if note.target.kind == NoteTargetKind.ASSET and note.target.asset:
            target_info = f" • {note.target.asset.ticker}"
        elif note.target.kind == NoteTargetKind.MARKET:
            target_info = f" • {note.target.symbol_name or note.target.symbol}"
        elif note.target.kind == NoteTargetKind.TRADE:
            target_info = f" • Trade #{note.target.trade_id}"

    with st.expander(f"{type_emoji} {display_title}{target_info} — {created_str}"):
        # Summary and key points
        if note.summary:
            st.caption(note.summary)

        if note.key_points:
            st.markdown("**Key Points:**")
            for point in note.key_points.split("\n"):
                if point.strip():
                    st.markdown(f"• {point.strip()}")

        st.divider()

        # Full content
        st.markdown(note.body_md)

        # Tags
        if note.tags:
            st.markdown("---")
            tag_list = [t.strip() for t in note.tags.split(",") if t.strip()]
            st.caption("Tags: " + " • ".join(f"`{tag}`" for tag in tag_list))

        # Actions (admin only)
        if config.ui.enable_admin_ui:
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("📌 Pin" if not note.pinned else "📌 Unpin", key=f"pin_{note.id}"):
                    pin_note(note.id, not note.pinned)
                    st.rerun()
            with col2:
                if st.button("🗑️ Archive", key=f"archive_{note.id}"):
                    archive_note(note.id)
                    st.rerun()


@st.dialog("Rename Tag")
def rename_tag_dialog(tag):
    """Dialog for renaming a tag."""
    st.markdown(f"**Rename tag: {tag.name}**")

    new_name = st.text_input(
        "New Name",
        value=tag.name,
        key=f"dialog_rename_input_{tag.id}",
    )

    col1, col2 = st.columns(2)

    if col1.button("Save", type="primary", key=f"dialog_rename_save_{tag.id}"):
        if new_name.strip() and new_name.strip() != tag.name:
            result = rename_tag(tag.id, new_name.strip())
            if result.success:
                st.success(result.message)
                time.sleep(1)
                st.rerun()
            else:
                st.error(result.message)
        else:
            st.rerun()

    if col2.button("Cancel", key=f"dialog_rename_cancel_{tag.id}"):
        st.rerun()


@st.dialog("Edit Tags")
def edit_asset_tags_dialog(asset, current_tags: list, all_tag_names: list):
    """Dialog for editing tags on an asset."""
    st.markdown(f"**Edit tags for {asset.ticker}**")

    current_tag_names = [tag.name for tag in current_tags]

    if all_tag_names:
        selected_tags = st.multiselect(
            "Select existing tags",
            options=all_tag_names,
            default=[t for t in current_tag_names if t in all_tag_names],
            key=f"dialog_select_tags_{asset.id}",
        )
    else:
        selected_tags = []
        st.caption("No existing tags")

    new_tags_input = st.text_input(
        "Add new tags (comma-separated)",
        placeholder="AI, Semiconductor, Cloud",
        key=f"dialog_new_tags_input_{asset.id}",
    )

    col1, col2 = st.columns(2)

    if col1.button("Save Tags", type="primary", key=f"dialog_save_{asset.id}"):
        all_tags = selected_tags.copy()
        if new_tags_input.strip():
            new_tags_list = [
                t.strip() for t in new_tags_input.split(",") if t.strip()
            ]
            all_tags.extend(new_tags_list)
        all_tags = list(set(all_tags))

        result = set_asset_tags_by_names(asset.id, all_tags)
        if result.success:
            st.success(result.message)
            time.sleep(1)
            st.rerun()
        else:
            st.error(result.message)

    if col2.button("Cancel", key=f"dialog_cancel_{asset.id}"):
        st.rerun()


def render_assets_tags_page():
    """
    Render Assets & Tags management page.

    Provides:
    - Asset CRUD (add, view, delete assets)
    - Tag CRUD (create, rename, delete tags)
    - Asset-tag associations (attach/detach tags)
    - Asset filtering by tags
    """
    from db import get_db
    from db.repositories import AssetRepository

    st.header("🏷️ Assets & Tags")
    st.caption("Manage your assets and organize them with tags")

    # ═══════════════════════════════════════════════════════════════════════════
    # TABS: Assets | Tags
    # ═══════════════════════════════════════════════════════════════════════════
    assets_tab, tags_tab = st.tabs(["📊 Assets", "🏷️ Tags"])

    # Get tag data for both tabs
    tag_result = get_all_tags_with_counts()
    all_tag_names = [t.tag.name for t in tag_result.tags] if tag_result.tags else []

    # ═══════════════════════════════════════════════════════════════════════════
    # ASSETS TAB
    # ═══════════════════════════════════════════════════════════════════════════
    with assets_tab:
        # --- Filter Controls ---
        st.subheader("🔍 Filter Assets")

        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

        with filter_col1:
            status_filter = st.selectbox(
                "Status",
                options=["All", "OWNED", "WATCHLIST"],
                index=0,
                key="asset_status_filter",
            )

        with filter_col2:
            if all_tag_names:
                tag_filter = st.multiselect(
                    "Filter by Tags",
                    options=all_tag_names,
                    placeholder="Select tags to filter...",
                    key="asset_tag_filter",
                )
            else:
                tag_filter = []
                st.caption("No tags available")

        with filter_col3:
            show_untagged = st.checkbox("Untagged only", key="show_untagged_only")

        st.divider()

        # --- Fetch and filter assets ---
        db = get_db()
        with db.session() as session:
            asset_repo = AssetRepository(session)

            if show_untagged:
                # Show only untagged assets
                status_enum = None
                if status_filter == "OWNED":
                    status_enum = AssetStatus.OWNED
                elif status_filter == "WATCHLIST":
                    status_enum = AssetStatus.WATCHLIST
                assets_list = get_untagged_assets(status=status_enum)
            elif tag_filter:
                # Filter by selected tags
                status_enum = None
                if status_filter == "OWNED":
                    status_enum = AssetStatus.OWNED
                elif status_filter == "WATCHLIST":
                    status_enum = AssetStatus.WATCHLIST
                assets_list = get_assets_by_tag_names(tag_filter, status=status_enum)
            else:
                # Show all assets based on status filter
                if status_filter == "OWNED":
                    assets_list = list(asset_repo.get_by_status(AssetStatus.OWNED))
                elif status_filter == "WATCHLIST":
                    assets_list = list(asset_repo.get_by_status(AssetStatus.WATCHLIST))
                else:
                    assets_list = list(asset_repo.get_all())

        # Sort assets by ticker
        assets_list = sorted(assets_list, key=lambda a: a.ticker)

        # --- Asset List with Tag Editing ---
        st.caption(f"**{len(assets_list)} Assets**")
        with st.container(height=400, border=False):
            if assets_list:
                for asset in assets_list:
                    with st.container(border=True):
                        col_info, col_tags, col_actions = st.columns([2, 4, 1])

                        with col_info:
                            st.markdown(f"**{asset.ticker}**")
                            status_badge = "🟢" if asset.status == AssetStatus.OWNED else "👁️"
                            st.caption(
                                f"{status_badge} {asset.status.value} · {asset.asset_type.value}"
                            )
                            if asset.name:
                                st.caption(
                                    asset.name[:35] + "..."
                                    if len(asset.name or "") > 35
                                    else asset.name
                                )

                        with col_tags:
                            current_tags = get_tags_for_ticker(asset.ticker)
                            if current_tags:
                                tag_pills = " ".join([f"`{tag.name}`" for tag in current_tags])
                                st.markdown(f"Tags: {tag_pills}")
                            else:
                                st.caption("_No tags_")

                        with col_actions:
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("🏷️", key=f"edit_tags_{asset.id}", help="Edit tags"):
                                    edit_asset_tags_dialog(asset, current_tags, all_tag_names)
                            with btn_col2:
                                if st.button(
                                    "🗑️", key=f"delete_asset_{asset.id}", help="Delete asset"
                                ):
                                    st.session_state[f"deleting_asset_{asset.id}"] = True
                                    st.rerun()

                        # --- Delete Confirmation ---
                        if st.session_state.get(f"deleting_asset_{asset.id}", False):
                            st.warning(
                                f"⚠️ **Delete {asset.ticker}?**\n\n"
                                "This will remove the asset and all related data (prices, fundamentals, valuations, notes)."
                            )

                            if asset.status == AssetStatus.OWNED:
                                st.error(
                                    "⚠️ This asset is OWNED. Deletion will be blocked if it has positions or trades."
                                )

                            del_col1, del_col2 = st.columns(2)

                            if del_col1.button(
                                "🗑️ Confirm Delete", key=f"confirm_del_{asset.id}", type="primary"
                            ):
                                result = delete_assets(
                                    tickers=[asset.ticker],
                                    allow_owned=True,
                                    allow_with_trades=False,
                                    allow_with_active_position=False,
                                )

                                if result.deleted:
                                    st.success(f"✅ Deleted {asset.ticker}")
                                    del st.session_state[f"deleting_asset_{asset.id}"]
                                    time.sleep(1)
                                    st.rerun()
                                elif result.blocked:
                                    st.error(f"❌ Blocked: {result.blocked[0]['reason']}")
                                else:
                                    st.error("❌ Delete failed")

                            if del_col2.button("Cancel", key=f"cancel_del_{asset.id}"):
                                del st.session_state[f"deleting_asset_{asset.id}"]
                                st.rerun()

            else:
                st.info("No assets match the current filters.")

        st.divider()

        # --- Add New Asset ---
        st.subheader("➕ Add New Asset")

        with st.form("add_new_asset_form", clear_on_submit=True):
            add_col1, add_col2, add_col3 = st.columns([2, 2, 2])

            with add_col1:
                new_ticker_input = st.text_input(
                    "Ticker Symbol *",
                    placeholder="e.g., AAPL, TSLA",
                )
                new_ticker = new_ticker_input.upper().strip() if new_ticker_input else ""

            with add_col2:
                new_asset_status = st.text_input(
                    "Status",
                    value="WATCHLIST",
                    help="OWNED = in portfolio | WATCHLIST = monitoring",
                    disabled=True,
                )

            with add_col3:
                new_asset_type = st.selectbox(
                    "Asset Type",
                    options=["STOCK", "ETF", "CRYPTO", "BOND", "DERIVATIVE"],
                    index=0,
                )

            # Tags for new asset
            if all_tag_names:
                new_asset_tags = st.multiselect(
                    "Tags (optional)",
                    options=all_tag_names,
                    placeholder="Select tags for this asset...",
                    key="new_asset_tags_select",
                )
            else:
                new_asset_tags = []

            add_submitted = st.form_submit_button("Add Asset", type="primary")

            if add_submitted:
                if not new_ticker:
                    st.error("❌ Ticker symbol is required")
                else:
                    with st.spinner(f"Validating {new_ticker}..."):
                        import yfinance as yf

                        try:
                            ticker_obj = yf.Ticker(new_ticker)
                            info = ticker_obj.info

                            if not info or len(info) <= 5 or info.get("regularMarketPrice") is None:
                                st.error(
                                    f"❌ **Ticker '{new_ticker}' not found**\n\n"
                                    "The ticker symbol does not exist in Yahoo Finance or has no price data."
                                )
                            else:
                                with st.spinner(f"Adding {new_ticker}..."):
                                    status = (
                                        AssetStatus.OWNED
                                        if new_asset_status == "OWNED"
                                        else AssetStatus.WATCHLIST
                                    )
                                    result = create_asset_with_data(
                                        new_ticker, status, asset_type=AssetType(new_asset_type)
                                    )

                                if result.success:
                                    # Attach tags if selected
                                    if new_asset_tags and result.asset:
                                        set_asset_tags_by_names(result.asset.id, new_asset_tags)

                                    st.success(
                                        f"✅ Added {new_ticker}\n\n"
                                        f"Prices fetched: {result.prices_fetched}\n\n"
                                        f"{result.status_message}"
                                    )
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed: {', '.join(result.errors)}")
                        except Exception as e:
                            st.error(
                                f"❌ **Unable to validate ticker '{new_ticker}'**\n\n"
                                f"Error: {str(e)}"
                            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAGS TAB
    # ═══════════════════════════════════════════════════════════════════════════
    with tags_tab:
        tag_col1, tag_col2 = st.columns([1, 1])

        # --- Left Column: Tag List ---
        with tag_col1:
            st.subheader("📋 Tag List")

            with st.container(height=360, border=False):
                if tag_result.tags:
                    for tag_with_count in tag_result.tags:
                        tag = tag_with_count.tag
                        count = tag_with_count.asset_count

                        with st.container(border=True):
                            col_name, col_count, col_actions = st.columns([3, 1, 2])

                            with col_name:
                                st.markdown(f"**{tag.name}**")
                                if tag.description:
                                    st.caption(tag.description[:50])

                            with col_count:
                                st.metric("Assets", count, label_visibility="collapsed")

                            with col_actions:
                                act_col1, act_col2 = st.columns(2)

                                with act_col1:
                                    if st.button("✏️", key=f"edit_tag_{tag.id}", help="Rename tag"):
                                        rename_tag_dialog(tag)

                                with act_col2:
                                    if st.button("🗑️", key=f"delete_tag_{tag.id}", help="Delete tag"):
                                        st.session_state[f"deleting_tag_{tag.id}"] = True
                                        st.rerun()

                            # --- Delete Confirmation ---
                            if st.session_state.get(f"deleting_tag_{tag.id}", False):
                                st.warning(
                                    f"⚠️ Delete tag **{tag.name}**? ({count} assets will be untagged)"
                                )

                                del_col1, del_col2 = st.columns(2)

                                if del_col1.button(
                                    "🗑️ Confirm", key=f"confirm_del_tag_{tag.id}", type="primary"
                                ):
                                    result = delete_tag(tag.id)
                                    if result.success:
                                        st.success(result.message)
                                        del st.session_state[f"deleting_tag_{tag.id}"]
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(result.message)

                                if del_col2.button("Cancel", key=f"cancel_del_tag_{tag.id}"):
                                    del st.session_state[f"deleting_tag_{tag.id}"]
                                    st.rerun()

                else:
                    st.info("No tags yet. Create your first tag!")

        # --- Right Column: Create Tag ---
        with tag_col2:
            st.subheader("➕ Create New Tag")

            with st.form("create_tag_form", clear_on_submit=True):
                tag_name = st.text_input(
                    "Tag Name *",
                    placeholder="e.g., AI, Dividend, Growth",
                )
                tag_description = st.text_input(
                    "Description (optional)",
                    placeholder="Brief description of this tag",
                )

                create_submitted = st.form_submit_button("Create Tag", type="primary")

                if create_submitted:
                    if not tag_name.strip():
                        st.error("Tag name is required")
                    else:
                        result = create_tag(
                            name=tag_name.strip(),
                            description=tag_description.strip() or None,
                        )
                        if result.success:
                            st.success(result.message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(result.message)


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
    elif page == "Charts":
        render_charts_page()
    elif page == "Assets & Tags":
        render_assets_tags_page()
    elif page == "Notes":
        render_notes_page()
    elif page == "Admin":
        render_admin_page()


if __name__ == "__main__":
    main()
