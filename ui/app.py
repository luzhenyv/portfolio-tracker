"""
Streamlit Dashboard for Portfolio Review.

FR-13: Built using Streamlit, local execution only
FR-14: Read-only, executive-style, calm layout
FR-15: Portfolio Overview, Positions Detail, Watchlist/Valuation View
"""

from typing import Literal, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime, date

from config import config
from db import init_db, AssetStatus, AssetType
from analytics.portfolio import compute_portfolio
from analytics.risk import compute_risk_metrics
from analytics.valuation import run_valuation
from analytics.performance import get_nav_period_returns, get_nav_series
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
    pages = ["Overview", "Positions", "Watchlist", "Notes"]
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
            st.plotly_chart(fig, use_container_width=True)

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
        st.metric("Total Portfolio Value", format_currency(total_value))
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
            default_benchmarks = ["SPX"] if available_indices and any(idx.symbol == "SPX" for idx in available_indices) else []
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
            st.caption(f"Normalized Performance (First Day = 1.0) — {st.session_state.selected_horizon}")
            
            # Build DataFrame for portfolio NAV (normalized)
            nav_dates = [nav.date for nav in nav_series.daily]
            nav_values = [nav.nav for nav in nav_series.daily]
            
            # Normalize portfolio NAV (first day = 1.0)
            base_nav = nav_values[0] if nav_values and nav_values[0] > 0 else 1
            normalized_nav = [v / base_nav for v in nav_values]
            
            # Create DataFrame for plotting
            chart_data = pd.DataFrame({
                "Date": nav_dates,
                "Portfolio": normalized_nav,
            })
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
                        bench_df = pd.DataFrame([
                            {"Date": p.date, symbol: p.value}
                            for p in series.prices
                        ])
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
                    trace.line.width = 2
                    trace.line.dash = "dash"
            
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
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No NAV history available. Add trades or cash transactions to see chart.")
        
        # Benchmark selection (multiselect pills-style) - moved below chart
        st.markdown("")
        if available_indices:
            benchmark_options = {idx.symbol: f"{idx.name} ({idx.symbol})" for idx in available_indices}
            
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
        
        # Time horizon selection (pill-style buttons) - moved below chart
        st.caption("Time Horizon")
        horizon_cols = st.columns(len(time_horizons))
        
        for col, horizon_name in zip(horizon_cols, time_horizons.keys()):
            with col:
                is_selected = st.session_state.selected_horizon == horizon_name
                if st.button(
                    horizon_name,
                    key=f"horizon_{horizon_name}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_horizon = horizon_name
                    st.rerun()
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
        "close": st.column_config.TextColumn(
            "Current", help="Latest end-of-day price per share"
        ),
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

        # Add Asset helper (if ticker doesn't exist)
        with st.expander("➕ Add New Asset to Track"):
            st.caption("Add a ticker before trading if it doesn't exist")
            with st.form("add_asset_form", clear_on_submit=True):
                new_ticker_selection = st.text_input(
                    "Ticker Symbol",
                    placeholder="Enter new ticker (e.g. MSFT)",
                )
                new_ticker = new_ticker_selection.upper().strip() if new_ticker_selection else ""

                asset_status = st.selectbox(
                    "Status",
                    ["OWNED", "WATCHLIST"],
                    index=0,
                )

                asset_type = st.selectbox(
                    "Asset Type",
                    ["STOCK", "ETF", "CRYPTO", "BOND", "DERIVATIVE"],
                    index=0,
                )

                add_submitted = st.form_submit_button("Add Asset")

                if add_submitted:
                    if not new_ticker:
                        st.error("❌ Ticker symbol is required")
                    else:
                        with st.spinner(f"Adding {new_ticker}..."):
                            status = (
                                AssetStatus.OWNED
                                if asset_status == "OWNED"
                                else AssetStatus.WATCHLIST
                            )
                            result = create_asset_with_data(
                                new_ticker, status, asset_type=AssetType(asset_type)
                            )

                        if result.success:
                            st.success(
                                f"✅ Added {new_ticker}\n\n"
                                f"Prices fetched: {result.prices_fetched}\n\n"
                                f"{result.status_message}"
                            )
                            celebrate_and_rerun()
                        else:
                            st.error(f"❌ Failed: {', '.join(result.errors)}")

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
                            f"{int(trade.shares)} shares @ /${trade.price:.2f} / "
                            f"Fees: /${trade.fees or 0:.2f} / "
                            f"P&L: /${trade.realized_pnl or 0:.2f}"
                        )
                        st.caption(f"ID: {trade_id} · {date_str}")
                    
                    with col_actions:
                        # Action buttons in a compact layout
                        if is_editable:
                            col_edit, col_delete = st.columns(2)
                            with col_edit:
                                if st.button("✏️", key=f"edit_{trade_id}", help="Edit trade"):
                                    edit_trade_dialog({
                                        "trade_id": trade_id,
                                        "Date": trade.trade_at,
                                        "Ticker": ticker,
                                        "Action": trade.action.value,
                                        "Shares": int(trade.shares),
                                        "Price": trade.price,
                                        "Fees": trade.fees or 0.0,
                                    })
                            with col_delete:
                                if st.button("🗑️", key=f"delete_{trade_id}", help="Delete trade"):
                                    delete_trade_dialog({
                                        "trade_id": trade_id,
                                        "Date": trade.trade_at,
                                        "Ticker": ticker,
                                        "Action": trade.action.value,
                                        "Shares": int(trade.shares),
                                        "Price": trade.price,
                                        "Fees": trade.fees or 0.0,
                                    })
                        else:
                            st.markdown("<div style='text-align: center; opacity: 0.3;'>🔒</div>", unsafe_allow_html=True)
                    
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
                                st.success(f"Added {name} with {sync_result[0].records_added} price records")
                            else:
                                st.warning(f"Added {name} but price sync failed. Run update job to fetch prices.")
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
    - Yahoo-aligned Valuation Measures table
    - Yahoo-aligned Financial Highlights table
    - BUY/WAIT/AVOID signals
    - All fields are editable with manual overrides saved to DB
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

    # Show "As of" date from the most recent updated_at
    if "updated_at" in valuation_df.columns:
        latest_update = valuation_df["updated_at"].dropna().max()
        if pd.notna(latest_update):
            if hasattr(latest_update, "strftime"):
                as_of_date = latest_update.strftime("%m/%d/%Y")
            else:
                as_of_date = str(latest_update)[:10]
            st.caption(f"📅 Data as of: {as_of_date}")

    # Store original for comparison
    if "watchlist_original_df" not in st.session_state:
        st.session_state.watchlist_original_df = valuation_df.copy()

    # Check if data has refreshed (e.g., different asset_ids)
    current_ids = set(valuation_df["asset_id"].tolist())
    original_ids = set(st.session_state.watchlist_original_df["asset_id"].tolist())
    if current_ids != original_ids:
        st.session_state.watchlist_original_df = valuation_df.copy()

    # ====== VALUATION MEASURES TABLE ======
    st.subheader("📊 Valuation Measures")

    # Create editor dataframe for valuation measures
    vm_df = valuation_df[["ticker", "asset_id"]].copy()

    # Large number fields for VM
    vm_large_fields = ["market_cap", "enterprise_value"]

    # Add valuation measure columns with formatting helpers
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

    # Add override flags
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
        flag_col = f"{col}_overridden"
        if flag_col in valuation_df.columns:
            vm_df[flag_col] = valuation_df[flag_col]
        else:
            vm_df[flag_col] = False

    edited_vm_df = st.data_editor(
        vm_df,
        hide_index=True,
        disabled=["ticker", "asset_id", "valuation_action"]
        + [
            f"{c}_overridden"
            for c in [
                "market_cap",
                "enterprise_value",
                "pe_trailing",
                "pe_forward",
                "peg",
                "price_to_sales",
                "price_to_book",
                "ev_to_revenue",
                "ev_ebitda",
            ]
        ],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "asset_id": None,  # Hidden
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
            # Hide override flags
            **{
                f"{c}_overridden": None
                for c in [
                    "market_cap",
                    "enterprise_value",
                    "pe_trailing",
                    "pe_forward",
                    "peg",
                    "price_to_sales",
                    "price_to_book",
                    "ev_to_revenue",
                    "ev_ebitda",
                ]
            },
        },
        key="valuation_measures_editor",
    )

    st.divider()

    # ====== FINANCIAL HIGHLIGHTS TABLE ======
    st.subheader("💰 Financial Highlights")

    # Create editor dataframe for financial highlights
    fh_df = valuation_df[["ticker", "asset_id"]].copy()

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

    # Add override flags
    for col in [
        "profit_margin",
        "return_on_assets",
        "return_on_equity",
        "revenue_ttm",
        "net_income_ttm",
        "diluted_eps_ttm",
        "total_cash",
        "total_debt_to_equity",
        "levered_free_cash_flow",
    ]:
        flag_col = f"{col}_overridden"
        if flag_col in valuation_df.columns:
            fh_df[flag_col] = valuation_df[flag_col]
        else:
            fh_df[flag_col] = False

    edited_fh_df = st.data_editor(
        fh_df,
        hide_index=True,
        disabled=["ticker", "asset_id"]
        + [
            f"{c}_overridden"
            for c in [
                "profit_margin",
                "return_on_assets",
                "return_on_equity",
                "revenue_ttm",
                "net_income_ttm",
                "diluted_eps_ttm",
                "total_cash",
                "total_debt_to_equity",
                "levered_free_cash_flow",
            ]
        ],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "asset_id": None,  # Hidden
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
            # Hide override flags
            **{
                f"{c}_overridden": None
                for c in [
                    "profit_margin",
                    "return_on_assets",
                    "return_on_equity",
                    "revenue_ttm",
                    "net_income_ttm",
                    "diluted_eps_ttm",
                    "total_cash",
                    "total_debt_to_equity",
                    "levered_free_cash_flow",
                ]
            },
        },
        key="financial_highlights_editor",
    )

    # ====== DETECT AND SAVE CHANGES ======
    original_df = st.session_state.watchlist_original_df
    changes = _detect_valuation_changes(edited_vm_df, edited_fh_df, original_df, valuation_df)

    if changes:
        st.divider()
        st.subheader("📝 Pending Changes")

        for change in changes:
            field_label = change["field"].replace("_", " ").title()
            old_val = _format_change_value(change["old_value"], change["field"])
            new_val = _format_change_value(change["new_value"], change["field"])
            st.write(f"**{change['ticker']}** — {field_label}: {old_val} → {new_val}")

        if st.button("💾 Save Changes", type="primary"):
            _save_valuation_overrides(changes)
            st.success("✅ Overrides saved successfully!")
            # Clear the original to force refresh
            del st.session_state.watchlist_original_df
            time.sleep(0.5)
            st.rerun()

    # Legend
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.markdown("🟢 **BUY** — Multiple metrics suggest attractive valuation")
    col2.markdown("🟡 **WAIT** — Mixed signals, patience recommended")
    col3.markdown("🔴 **AVOID** — Multiple metrics suggest overvaluation")

    st.caption(
        "💡 Edit any numeric cell to override auto-fetched values. "
        "Overrides are saved and take precedence over yfinance data. "
        "Missing data is displayed as empty cells; you can fill them manually."
    )


def _format_change_value(value: float | None, field: str) -> str:
    """Format a change value for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    # Percentage fields
    if field in ["profit_margin", "return_on_assets", "return_on_equity"]:
        return f"{value:.2f}%"

    # Large number fields
    if field in [
        "market_cap",
        "enterprise_value",
        "revenue_ttm",
        "net_income_ttm",
        "total_cash",
        "levered_free_cash_flow",
    ]:
        return _format_large_number(value)

    # Other fields
    return f"{value:.2f}"


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


def _parse_large_number(value: str | float | None) -> float | None:
    """Parse string with B/T notation (e.g., '1.45T') back to float."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    value = value.strip().upper()
    if not value or value == "—":
        return None

    # Handle multipliers
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}

    value = value.replace(",", "")

    suffix = value[-1]
    if suffix in multipliers:
        try:
            return float(value[:-1]) * multipliers[suffix]
        except ValueError:
            return None

    try:
        return float(value)
    except ValueError:
        return None


def _detect_valuation_changes(
    edited_vm_df: pd.DataFrame,
    edited_fh_df: pd.DataFrame,
    original_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
) -> list[dict]:
    """Detect changes between edited and original dataframes."""
    changes = []

    # Valuation measures fields
    vm_fields = [
        "market_cap",
        "enterprise_value",
        "pe_trailing",
        "pe_forward",
        "peg",
        "price_to_sales",
        "price_to_book",
        "ev_to_revenue",
        "ev_ebitda",
    ]

    # Financial highlights fields (note: percentages were multiplied by 100 for display)
    fh_percentage_fields = ["profit_margin", "return_on_assets", "return_on_equity"]
    fh_other_fields = [
        "revenue_ttm",
        "net_income_ttm",
        "diluted_eps_ttm",
        "total_cash",
        "total_debt_to_equity",
        "levered_free_cash_flow",
    ]

    for idx, row in edited_vm_df.iterrows():
        asset_id = row["asset_id"]
        ticker = row["ticker"]

        # Find original values
        orig_row = original_df[original_df["asset_id"] == asset_id]
        if orig_row.empty:
            continue

        for field in vm_fields:
            if field not in edited_vm_df.columns or field not in original_df.columns:
                continue

            new_val = row[field]
            orig_val = orig_row.iloc[0][field]

            # Parse large number if needed
            if field in ["market_cap", "enterprise_value"]:
                new_val = _parse_large_number(new_val)

            if _values_differ(new_val, orig_val):
                changes.append(
                    {
                        "asset_id": asset_id,
                        "ticker": ticker,
                        "field": field,
                        "old_value": orig_val,
                        "new_value": new_val,
                    }
                )

    for idx, row in edited_fh_df.iterrows():
        asset_id = row["asset_id"]
        ticker = row["ticker"]

        orig_row = original_df[original_df["asset_id"] == asset_id]
        if orig_row.empty:
            continue

        # Percentage fields (need to convert back from display %)
        for field in fh_percentage_fields:
            if field not in edited_fh_df.columns or field not in original_df.columns:
                continue

            new_val_display = row[field]
            # Convert back to decimal
            new_val = new_val_display / 100 if pd.notna(new_val_display) else None
            orig_val = orig_row.iloc[0][field]

            if _values_differ(new_val, orig_val):
                changes.append(
                    {
                        "asset_id": asset_id,
                        "ticker": ticker,
                        "field": field,
                        "old_value": orig_val * 100 if pd.notna(orig_val) else None,
                        "new_value": new_val_display,
                    }
                )

        # Other fields
        for field in fh_other_fields:
            if field not in edited_fh_df.columns or field not in original_df.columns:
                continue

            new_val = row[field]
            orig_val = orig_row.iloc[0][field]

            # Parse large number if needed
            if field in ["revenue_ttm", "net_income_ttm", "total_cash", "levered_free_cash_flow"]:
                new_val = _parse_large_number(new_val)

            if _values_differ(new_val, orig_val):
                changes.append(
                    {
                        "asset_id": asset_id,
                        "ticker": ticker,
                        "field": field,
                        "old_value": orig_val,
                        "new_value": new_val,
                    }
                )

    return changes


def _values_differ(new_val, orig_val, tolerance: float = 0.001) -> bool:
    """Check if two values differ (handling NaN)."""
    if pd.isna(new_val) and pd.isna(orig_val):
        return False
    if pd.isna(new_val) != pd.isna(orig_val):
        return True
    if pd.notna(new_val) and pd.notna(orig_val):
        return abs(new_val - orig_val) >= tolerance
    return False


def _save_valuation_overrides(changes: list[dict]):
    """Save valuation override changes to database."""
    from db import get_db
    from db.repositories import ValuationOverrideRepository

    db = get_db()
    with db.session() as session:
        override_repo = ValuationOverrideRepository(session)

        # Group changes by asset_id
        changes_by_asset: dict[int, dict] = {}
        for change in changes:
            asset_id = change["asset_id"]
            if asset_id not in changes_by_asset:
                changes_by_asset[asset_id] = {}

            field = change["field"]
            new_value = change["new_value"]

            # Convert percentage display values back to decimals for storage
            if field in ["profit_margin", "return_on_assets", "return_on_equity"]:
                new_value = new_value / 100 if pd.notna(new_value) else None

            changes_by_asset[asset_id][f"{field}_override"] = (
                new_value if pd.notna(new_value) else None
            )

        for asset_id, field_updates in changes_by_asset.items():
            # Get existing override to preserve other fields
            existing = override_repo.get_by_asset_id(asset_id)

            # Build kwargs with existing values
            kwargs = {"asset_id": asset_id}

            override_fields = [
                "market_cap_override",
                "enterprise_value_override",
                "pe_trailing_override",
                "pe_forward_override",
                "peg_override",
                "price_to_sales_override",
                "price_to_book_override",
                "ev_to_revenue_override",
                "ev_ebitda_override",
                "profit_margin_override",
                "return_on_assets_override",
                "return_on_equity_override",
                "revenue_ttm_override",
                "net_income_ttm_override",
                "diluted_eps_ttm_override",
                "total_cash_override",
                "total_debt_to_equity_override",
                "levered_free_cash_flow_override",
            ]

            for field in override_fields:
                if field in field_updates:
                    kwargs[field] = field_updates[field]
                elif existing and hasattr(existing, field):
                    kwargs[field] = getattr(existing, field)
                else:
                    kwargs[field] = None

            override_repo.upsert(**kwargs)

        session.commit()


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
    elif page == "Notes":
        render_notes_page()
    elif page == "Admin":
        render_admin_page()


if __name__ == "__main__":
    main()
