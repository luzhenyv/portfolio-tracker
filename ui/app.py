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
from services.trade_service import get_recent_trades
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
            allocation_data.append({
                "Asset": row["ticker"], 
                "Type": row.get("asset_type", "STOCK"),
                "Value": market_value
            })

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
        "avg_cost": st.column_config.TextColumn(
            "Net Avg Cost",
            width="small",
            help="Average cost per share based on net invested capital (cost basis / net shares)"
        ),
        "close": st.column_config.TextColumn(
            "Current",
            width="small",
            help="Latest end-of-day price per share"
        ),
        "market_value": st.column_config.TextColumn(
            "Market Value",
            width="medium",
            help="Net market value = (long shares × price) - (short shares × price)"
        ),
        "display_pnl": st.column_config.TextColumn(
            "P&L",
            width="small",
            help="Unrealized profit/loss = net market value - net invested capital"
        ),
        "pnl_pct": st.column_config.TextColumn(
            "P&L %",
            width="small",
            help="Return on invested capital = (net market value - net invested) / net invested"
        ),
        "net_weight": st.column_config.TextColumn(
            "Exposure %",
            width="small",
            help="Share of absolute net exposure: $\\frac{|MV_{net}|}{\\sum |MV_{net}|}$ (positions only, excludes cash)"
        ),
        "action": st.column_config.TextColumn("Action", width="small"),
        "reasons": st.column_config.TextColumn("Reasons", width="large"),
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

            trade_date = st.date_input(
                "Trade Date",
                value=date.today(),
                help="Date of trade execution",
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
                                    trade_date=str(trade_date),
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
                                trade_date=str(trade_date),
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
                                new_ticker, 
                                status, 
                                asset_type=AssetType(asset_type)
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

            cash_date = st.date_input(
                "Transaction Date",
                value=date.today(),
                help="Date of cash transaction",
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
                        transaction_date=str(cash_date),
                        description=description or None,
                    )

                    new_balance = get_current_cash_balance()
                    st.success(
                        f"✅ **Deposited ${amount:,.2f}**\n\n"
                        f"Date: {cash_date}\n\n"
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
                            transaction_date=str(cash_date),
                            description=description or None,
                        )

                        new_balance = get_current_cash_balance()
                        st.success(
                            f"✅ **Withdrew ${amount:,.2f}**\n\n"
                            f"Date: {cash_date}\n\n"
                            f"💵 New balance: ${new_balance:,.2f}"
                        )
                    celebrate_and_rerun()
    # --- RECENT ACTIVITY ---
    st.subheader("📒 Recent Activity")

    activity_tabs = st.tabs(["Recent Trades", "Cash Ledger"])

    with activity_tabs[0]:
        # Show recent trades
        recent_trades = get_recent_trades(limit=10)

        if recent_trades:
            trade_data = []
            for trade in recent_trades:
                trade_data.append(
                    {
                        "Date": trade.trade_date,
                        "Ticker": trade.asset.ticker if trade.asset else "?",
                        "Action": trade.action.value,
                        "Shares": f"{trade.shares:,.2f}",
                        "Price": f"${trade.price:,.2f}",
                        "Fees": f"${trade.fees:,.2f}" if trade.fees else "-",
                        "Realized P&L": (
                            f"${trade.realized_pnl:+,.2f}" if trade.realized_pnl else "-"
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(trade_data),
                hide_index=True,
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


def render_watchlist_page():
    """
    Render Watchlist & Valuation page with editable metrics.

    Shows:
    - Valuation metrics for all assets
    - BUY/WAIT/AVOID signals
    - Editable PEG column for user overrides
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

    # Store original for comparison
    if "watchlist_original_df" not in st.session_state:
        st.session_state.watchlist_original_df = valuation_df.copy()
    
    # Check if data has refreshed (e.g., different asset_ids)
    current_ids = set(valuation_df["asset_id"].tolist())
    original_ids = set(st.session_state.watchlist_original_df["asset_id"].tolist())
    if current_ids != original_ids:
        st.session_state.watchlist_original_df = valuation_df.copy()

    # Prepare editable dataframe - keep numeric values for editing
    edit_df = valuation_df.copy()
    
    # Add display columns for override indicators
    edit_df["peg_display"] = edit_df.apply(
        lambda row: f"{row['peg']:.2f} ✏️" if pd.notna(row['peg']) and row.get('peg_overridden', False)
        else (f"{row['peg']:.2f}" if pd.notna(row['peg']) else None),
        axis=1
    )
    
    # Format non-editable numeric columns for display
    display_df = valuation_df.copy()
    
    for col in ["pe_forward", "ev_ebitda"]:
        if col in display_df.columns:
            display_df[f"{col}_fmt"] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    for col in ["revenue_growth", "eps_growth"]:
        if col in display_df.columns:
            display_df[f"{col}_fmt"] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    # Create the editable dataframe with only the columns we need
    # Multiply growth values by 100 for percentage display (0.116 -> 11.6)
    editor_df = pd.DataFrame({
        "ticker": valuation_df["ticker"],
        "asset_id": valuation_df["asset_id"],
        "pe_forward": valuation_df["pe_forward"],
        "peg": valuation_df["peg"],
        "ev_ebitda": valuation_df["ev_ebitda"],
        "revenue_growth": valuation_df["revenue_growth"].apply(lambda x: x * 100 if pd.notna(x) else None),
        "eps_growth": valuation_df["eps_growth"].apply(lambda x: x * 100 if pd.notna(x) else None),
        "valuation_action": valuation_df["valuation_action"],
        "peg_overridden": valuation_df.get("peg_overridden", False),
    })

    # Use data_editor with editable PEG column
    edited_df = st.data_editor(
        editor_df,
        hide_index=True,
        disabled=["ticker", "asset_id", "pe_forward", "ev_ebitda", "revenue_growth", "eps_growth", "valuation_action", "peg_overridden"],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "asset_id": None,  # Hide asset_id column
            "pe_forward": st.column_config.NumberColumn(
                "Fwd P/E",
                width="small",
                format="%.1f",
                help="Forward Price-to-Earnings ratio: market price / expected next-year EPS (lower = cheaper)"
            ),
            "peg": st.column_config.NumberColumn(
                "PEG",
                width="small",
                format="%.2f",
                help="Price/Earnings-to-Growth ratio: (P/E) / EPS growth % (< 1 = undervalued, > 2 = overvalued). Edit to override."
            ),
            "ev_ebitda": st.column_config.NumberColumn(
                "EV/EBITDA",
                width="small",
                format="%.1f",
                help="Enterprise Value / EBITDA multiple: measures valuation relative to operating earnings"
            ),
            "revenue_growth": st.column_config.NumberColumn(
                "Rev Growth %",
                width="small",
                format="%.1f%%",
                help="Year-over-year revenue growth rate (trailing or projected)"
            ),
            "eps_growth": st.column_config.NumberColumn(
                "EPS Growth %",
                width="small",
                format="%.1f%%",
                help="Year-over-year earnings per share growth rate"
            ),
            "valuation_action": st.column_config.TextColumn(
                "Signal",
                width="small",
                help="BUY = attractive valuation | WAIT = mixed signals | AVOID = overvalued"
            ),
            "peg_overridden": st.column_config.CheckboxColumn(
                "✏️",
                width="small",
                help="Checked if user override is applied",
            ),
        },
        key="watchlist_editor",
    )

    # Check for changes and show save button
    original_df = st.session_state.watchlist_original_df
    changes = []
    
    for idx, row in edited_df.iterrows():
        asset_id = row["asset_id"]
        new_peg = row["peg"]
        
        # Find original value
        orig_row = original_df[original_df["asset_id"] == asset_id]
        if not orig_row.empty:
            orig_peg = orig_row.iloc[0]["peg"]
            
            # Compare values (handle NaN)
            if pd.isna(new_peg) and pd.isna(orig_peg):
                continue
            if pd.notna(new_peg) and pd.notna(orig_peg) and abs(new_peg - orig_peg) < 0.001:
                continue
            if pd.isna(new_peg) != pd.isna(orig_peg) or (pd.notna(new_peg) and abs(new_peg - orig_peg) >= 0.001):
                changes.append({
                    "asset_id": asset_id,
                    "ticker": row["ticker"],
                    "old_peg": orig_peg,
                    "new_peg": new_peg,
                })

    if changes:
        st.divider()
        st.subheader("📝 Pending Changes")
        
        for change in changes:
            old_val = f"{change['old_peg']:.2f}" if pd.notna(change['old_peg']) else "—"
            new_val = f"{change['new_peg']:.2f}" if pd.notna(change['new_peg']) else "—"
            st.write(f"**{change['ticker']}**: PEG {old_val} → {new_val}")
        
        if st.button("💾 Save Changes", type="primary"):
            _save_peg_overrides(changes)
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
        "💡 Edit the PEG column to override auto-fetched values. "
        "Overrides are indicated by ✏️ checkbox. Changes require saving."
    )


def _save_peg_overrides(changes: list[dict]):
    """Save PEG override changes to database."""
    from db import get_db
    from db.repositories import ValuationOverrideRepository
    
    db = get_db()
    with db.session() as session:
        override_repo = ValuationOverrideRepository(session)
        
        for change in changes:
            asset_id = change["asset_id"]
            new_peg = change["new_peg"]
            
            # Get existing override to preserve other fields
            existing = override_repo.get_by_asset_id(asset_id)
            
            if existing:
                override_repo.upsert(
                    asset_id=asset_id,
                    peg_override=new_peg if pd.notna(new_peg) else None,
                    pe_forward_override=existing.pe_forward_override,
                    ev_ebitda_override=existing.ev_ebitda_override,
                    revenue_growth_override=existing.revenue_growth_override,
                    eps_growth_override=existing.eps_growth_override,
                )
            else:
                override_repo.upsert(
                    asset_id=asset_id,
                    peg_override=new_peg if pd.notna(new_peg) else None,
                )
        
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
