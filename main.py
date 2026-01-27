"""
Portfolio Analytics System - Main Entry Point.

A local-first personal investment analytics system for long-term portfolio
review and decision-making with support for long and short positions.

Usage:
    # Initialize database
    python main.py init

    # Add an asset
    python main.py add-asset TSLA --status OWNED --name "Tesla Inc."
    python main.py add-asset NVDA
    python main.py add-asset KO
    python main.py add-asset NFLX

    # Trading operations
    python main.py buy TSLA --shares 100 --price 150.00 --date 2025-01-15
    python main.py sell TSLA --shares 50 --price 170.00 --date 2025-06-15
    python main.py buy NVDA --shares 50 --price 123.00 --date 2025-09-15
    python main.py buy KO --shares 50 --price 64.30 --date 2025-10-23

    # Cash operations
    python main.py deposit 100000 --description "Initial capital" --date 2025-01-01
    python main.py withdraw 500 --description "Personal expense" --date 2025-12-25

    # Selling more than you own creates a short position
    python main.py sell TSLA --shares 120 --price 180.00  # Goes short 20 shares

    # Buying when short covers the short position first
    python main.py buy TSLA --shares 30 --price 175.00  # Covers 20 short, buys 10 long

    # View trades and P&L
    python main.py trades --limit 10
    python main.py pnl --since 2024-01-01

    # Run daily update (fetch prices + valuations)
    python main.py update

    # Launch dashboard
    python main.py dashboard

    # Show portfolio summary
    python main.py summary
"""

import argparse
import sys
from datetime import datetime

from db import init_db, AssetStatus, AssetType
from db.repositories import AssetRepository
from services.asset_service import (
    create_asset_with_data,
    print_asset_creation_result,
)
from services.position_service import (
    buy_position,
    print_trade_result,
    sell_position,
)


def cmd_init(args):
    """Initialize the database."""
    db = init_db(args.db_url, if_drop=args.if_drop)
    if args.if_drop:
        print("⚠️ Existing tables dropped.")
    print("✅ Database initialized successfully")
    print(f"   Location: {db.db_url}")


def cmd_add_asset(args):
    """Add a new asset to track."""
    status = AssetStatus.OWNED if args.status == "OWNED" else AssetStatus.WATCHLIST
    asset_type = AssetType(args.type.upper()) if args.type else AssetType.STOCK
    
    result = create_asset_with_data(
        ticker=args.ticker.upper(), 
        status=status,
        asset_type=asset_type
    )
    print_asset_creation_result(result)


def cmd_buy(args):
    """Buy shares (covers short position first if short, then goes long)."""
    result = buy_position(
        ticker=args.ticker,
        shares=args.shares,
        price=args.price,
        trade_date=args.date,
        fees=args.fees,
    )
    print_trade_result(result)


def cmd_sell(args):
    """Sell shares (reduces long position, or creates short if selling more than held)."""
    result = sell_position(
        ticker=args.ticker,
        shares=args.shares,
        price=args.price,
        trade_date=args.date,
        fees=args.fees,
    )
    print_trade_result(result)


def cmd_update(args):
    """Run daily data update."""
    from jobs.daily_update import run_daily_update

    return run_daily_update()


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
        ]
    )


def cmd_summary(args):
    """Show portfolio summary."""
    from db import init_db

    init_db()

    try:
        from analytics.portfolio import compute_portfolio
        from analytics.risk import compute_risk_metrics
        from decision.engine import decision_engine

        df, summary = compute_portfolio()
        risk = compute_risk_metrics()
        decisions = decision_engine()

        print("\n" + "=" * 50)
        print("📊 PORTFOLIO SUMMARY")
        print("=" * 50)

        print(f"\n💰 Value")
        print(f"   Long MV:         ${summary['long_mv']:>12,.0f}")
        print(f"   Short MV:        ${summary['short_mv']:>12,.0f}")
        print(f"   Gross Exposure:  ${summary['gross_exposure']:>12,.0f}")
        print(f"   Holdings (Net):  ${summary['holdings_market_value']:>12,.0f}")

        print(f"\n📈 P&L")
        print(f"   Unrealized:      ${summary['total_unrealized_pnl']:>+12,.0f}")
        print(f"   Realized:        ${summary['total_realized_pnl']:>+12,.0f}")
        print(f"   Total P&L:       ${summary['total_pnl']:>+12,.0f}")

        print(f"\n⚠️ Risk")
        print(f"   Portfolio Vol:   {risk['portfolio_volatility']:>12.1%}")
        print(f"   Max Drawdown:    {risk['portfolio_max_drawdown']:>12.1%}")

        print(f"\n📍 Positions ({len(df)})")
        for _, row in df.iterrows():
            ticker = row["ticker"]
            if row["long_shares"] > 0 and row["short_shares"] > 0:
                # Both long and short
                print(
                    f"   {ticker:<6} L:{row['long_shares']:>8.2f} S:{row['short_shares']:>8.2f} | Gross:{row['gross_weight']:>6.1%} P&L:{row['pnl']:>+10,.0f}"
                )
            elif row["long_shares"] > 0:
                # Long only
                print(
                    f"   {ticker:<6} Long {row['long_shares']:>8.2f} @ ${row['close']:>7.2f} | {row['gross_weight']:>6.1%} P&L:{row['pnl']:>+10,.0f}"
                )
            elif row["short_shares"] > 0:
                # Short only
                print(
                    f"   {ticker:<6} Short {row['short_shares']:>7.2f} @ ${row['close']:>7.2f} | {row['gross_weight']:>6.1%} P&L:{row['pnl']:>+10,.0f}"
                )

        if not decisions.empty:
            actions_needed = decisions[decisions["action"] != "HOLD"]
            if not actions_needed.empty:
                print(f"\n🧠 Actions Needed")
                for _, row in actions_needed.iterrows():
                    print(f"   {row['ticker']}: {row['action']} - {row['reasons']}")

        print("\n" + "=" * 50)

    except ValueError as e:
        print(f"⚠️ {e}")
        print("Add positions to see portfolio analytics.")


def cmd_list_assets(args):
    """List all tracked assets."""
    db = init_db()

    with db.session() as session:
        repo = AssetRepository(session)
        assets = repo.get_all()

        if not assets:
            print("No assets tracked yet.")
            return

        print("\n📋 Tracked Assets")
        print("-" * 50)
        print(f"{'Ticker':<8} {'Status':<10} {'Name':<30}")
        print("-" * 50)

        for asset in assets:
            name = asset.name or ""
            if len(name) > 28:
                name = name[:25] + "..."
            print(f"{asset.ticker:<8} {asset.status.value:<10} {name:<30}")


def cmd_trades(args):
    """List recent trades."""
    from db import init_db, TradeAction
    from db.repositories import TradeRepository

    db = init_db()

    with db.session() as session:
        trade_repo = TradeRepository(session)

        # Parse optional filters
        action = TradeAction[args.action.upper()] if args.action else None

        trades = trade_repo.get_all_trades(
            start_date=args.since,
            action=action,
            limit=args.limit,
        )

        if not trades:
            print("No trades found.")
            return

        print(f"\n💼 Trade History (last {len(trades)})")
        print("-" * 80)
        print(f"{'Date':<12} {'Ticker':<8} {'Action':<8} {'Shares':>10} {'Price':>10} {'P&L':>12}")
        print("-" * 80)

        for trade in trades:
            ticker = trade.asset.ticker if trade.asset else "?"
            pnl_str = f"${trade.realized_pnl:+,.2f}" if trade.realized_pnl != 0 else "-"
            print(
                f"{trade.trade_date:<12} {ticker:<8} {trade.action.value:<8} {trade.shares:>10.2f} ${trade.price:>9.2f} {pnl_str:>12}"
            )


def cmd_pnl(args):
    """Show realized P&L summary."""
    from db.repositories import TradeRepository

    db = init_db()

    with db.session() as session:
        trade_repo = TradeRepository(session)
        summary = trade_repo.get_realized_pnl_summary(
            start_date=args.since,
            end_date=args.until,
        )

    print("\n💰 Realized P&L Summary")
    print("-" * 40)
    print(f"   Total Realized P&L:  ${summary['total_realized_pnl']:>+12,.2f}")
    print(f"   Total Fees:          ${summary['total_fees']:>12,.2f}")
    print(f"   Net Realized P&L:    ${summary['net_realized_pnl']:>+12,.2f}")


def cmd_cash(args):
    """Show cash position and ledger."""
    from db.repositories import CashRepository

    db = init_db()

    with db.session() as session:
        cash_repo = CashRepository(session)

        summary = cash_repo.get_summary(
            start_date=args.since,
            end_date=args.until,
        )

        print("\n💵 Cash Summary")
        print("-" * 50)
        if args.since or args.until:
            print(f"   Starting Balance:    ${summary['starting_balance']:>12,.2f}")
        print(f"   Total Inflows:       ${summary['total_inflows']:>12,.2f}")
        print(f"   Total Outflows:      ${summary['total_outflows']:>12,.2f}")
        print(f"   Net Flow:            ${summary['net_flow']:>+12,.2f}")
        print(f"   Current Balance:     ${summary['ending_balance']:>12,.2f}")

        # Show breakdown by type
        by_type = cash_repo.get_balance_by_type(
            start_date=args.since,
            end_date=args.until,
        )

        if by_type:
            print("\n📊 Breakdown by Type")
            print("-" * 50)
            for tx_type, amount in sorted(by_type.items()):
                print(f"   {tx_type:<15}     ${amount:>+12,.2f}")

        # Show recent transactions
        if args.ledger:
            ledger = cash_repo.get_ledger(
                start_date=args.since,
                end_date=args.until,
                limit=args.limit,
            )

            if ledger:
                print(f"\n📒 Cash Ledger (last {min(len(ledger), args.limit)})")
                print("-" * 90)
                print(f"{'Date':<12} {'Type':<10} {'Amount':>12} {'Balance':>12}   Description")
                print("-" * 90)

                for entry in ledger[-args.limit:]:
                    print(
                        f"{entry['date']:<12} {entry['type']:<10} ${entry['amount']:>+11,.2f} ${entry['balance']:>11,.2f}   {entry['description'][:30]}"
                    )


def cmd_cash_deposit(args):
    """Deposit cash (add capital)."""
    from db.repositories import CashRepository

    db = init_db()

    with db.session() as session:
        cash_repo = CashRepository(session)

        tx = cash_repo.deposit(
            amount=args.amount,
            transaction_date=args.date,
            description=args.description,
        )

        balance = cash_repo.get_balance()

    print(f"\n✅ Deposited ${args.amount:,.2f}")
    print(f"   Date: {tx.transaction_date}")
    if args.description:
        print(f"   Description: {args.description}")
    print(f"   💵 New Balance: ${balance:,.2f}")


def cmd_cash_withdraw(args):
    """Withdraw cash."""
    from db import init_db
    from db.repositories import CashRepository

    db = init_db()

    with db.session() as session:
        cash_repo = CashRepository(session)

        # Check balance first
        current_balance = cash_repo.get_balance()
        if args.amount > current_balance:
            print(f"\n⚠️ Warning: Withdrawal ${args.amount:,.2f} exceeds balance ${current_balance:,.2f}")
            if not args.force:
                print("   Use --force to proceed anyway")
                return

        tx = cash_repo.withdraw(
            amount=args.amount,
            transaction_date=args.date,
            description=args.description,
        )

        balance = cash_repo.get_balance()

    print(f"\n✅ Withdrew ${args.amount:,.2f}")
    print(f"   Date: {tx.transaction_date}")
    if args.description:
        print(f"   Description: {args.description}")
    print(f"   💵 New Balance: ${balance:,.2f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Portfolio Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    init = subparsers.add_parser("init", help="Initialize database")
    init.add_argument("--db-url", help="Custom database URL", default=None)
    init.add_argument("--if-drop", action="store_true", help="Drop existing tables before creating")

    # add-asset command
    add_asset = subparsers.add_parser("add-asset", help="Add asset to track")
    add_asset.add_argument("ticker", help="Stock ticker symbol")
    add_asset.add_argument("--status", choices=["OWNED", "WATCHLIST"], default="WATCHLIST")
    add_asset.add_argument("--type", choices=["STOCK", "ETF", "CRYPTO", "BOND", "DERIVATIVE"], default="STOCK", help="Asset type")
    add_asset.add_argument("--name", help="Company name")
    add_asset.add_argument("--sector", help="Sector")
    add_asset.add_argument("--exchange", help="Exchange")

    # buy command
    buy = subparsers.add_parser("buy", help="Buy shares (long position)")
    buy.add_argument("ticker", help="Stock ticker symbol")
    buy.add_argument("--shares", type=float, required=True, help="Number of shares")
    buy.add_argument("--price", type=float, required=True, help="Price per share")
    buy.add_argument("--date", help="Trade date (YYYY-MM-DD, default: today)")
    buy.add_argument("--fees", type=float, default=0.0, help="Trading fees")

    # sell command
    sell = subparsers.add_parser("sell", help="Sell shares (reduces long, or creates short)")
    sell.add_argument("ticker", help="Stock ticker symbol")
    sell.add_argument("--shares", type=float, required=True, help="Number of shares")
    sell.add_argument("--price", type=float, required=True, help="Price per share")
    sell.add_argument("--date", help="Trade date (YYYY-MM-DD, default: today)")
    sell.add_argument("--fees", type=float, default=0.0, help="Trading fees")

    # update command
    subparsers.add_parser("update", help="Run daily data update")

    # dashboard command
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    # summary command
    subparsers.add_parser("summary", help="Show portfolio summary")

    # list command
    subparsers.add_parser("list", help="List tracked assets")

    # trades command
    trades = subparsers.add_parser("trades", help="List recent trades")
    trades.add_argument("--since", help="Start date (YYYY-MM-DD)")
    trades.add_argument(
        "--action", choices=["buy", "sell", "short", "cover"], help="Filter by action"
    )
    trades.add_argument("--limit", type=int, default=20, help="Max number of trades to show")

    # pnl command
    pnl = subparsers.add_parser("pnl", help="Show realized P&L summary")
    pnl.add_argument("--since", help="Start date (YYYY-MM-DD)")
    pnl.add_argument("--until", help="End date (YYYY-MM-DD)")

    # cash command
    cash = subparsers.add_parser("cash", help="Show cash position and ledger")
    cash.add_argument("--since", help="Start date (YYYY-MM-DD)")
    cash.add_argument("--until", help="End date (YYYY-MM-DD)")
    cash.add_argument("--ledger", action="store_true", help="Show transaction ledger")
    cash.add_argument("--limit", type=int, default=20, help="Max ledger entries to show")

    # deposit command
    deposit = subparsers.add_parser("deposit", help="Deposit cash (add capital)")
    deposit.add_argument("amount", type=float, help="Amount to deposit")
    deposit.add_argument("--date", help="Transaction date (YYYY-MM-DD, default: today)")
    deposit.add_argument("--description", help="Description of deposit")

    # withdraw command
    withdraw = subparsers.add_parser("withdraw", help="Withdraw cash")
    withdraw.add_argument("amount", type=float, help="Amount to withdraw")
    withdraw.add_argument("--date", help="Transaction date (YYYY-MM-DD, default: today)")
    withdraw.add_argument("--description", help="Description of withdrawal")
    withdraw.add_argument("--force", action="store_true", help="Allow negative balance")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init,
        "add-asset": cmd_add_asset,
        "buy": cmd_buy,
        "sell": cmd_sell,
        "update": cmd_update,
        "dashboard": cmd_dashboard,
        "summary": cmd_summary,
        "list": cmd_list_assets,
        "trades": cmd_trades,
        "pnl": cmd_pnl,
        "cash": cmd_cash,
        "deposit": cmd_cash_deposit,
        "withdraw": cmd_cash_withdraw,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
