"""
Portfolio Analytics System - Main Entry Point.

A local-first personal investment analytics system for long-term portfolio
review and decision-making.

Usage:
    # Initialize database
    python main.py init
    
    # Add an asset
    python main.py add-asset AAPL --status OWNED --name "Apple Inc."
    
    # Add a position
    python main.py add-position AAPL --shares 100 --price 150.00 --date 2024-01-15
    
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

from db import init_db, get_db, Asset, AssetStatus
from db.repositories import AssetRepository, PositionRepository
from services.asset_service import create_asset_with_data, print_asset_creation_result


def cmd_init(args):
    """Initialize the database."""
    db = init_db(args.db_url, if_drop=args.if_drop)
    print("✅ Database initialized successfully")
    print(f"   Location: {db.db_url}")


def cmd_add_asset(args):
    """Add a new asset to track."""
    status = AssetStatus.OWNED if args.status == "OWNED" else AssetStatus.WATCHLIST
    result = create_asset_with_data(args.ticker.upper(), status)
    print_asset_creation_result(result)


def cmd_add_position(args):
    """Add a position (buy lot) for an asset."""
    db = init_db()
    
    with db.session() as session:
        asset_repo = AssetRepository(session)
        position_repo = PositionRepository(session)
        
        asset = asset_repo.get_by_ticker(args.ticker.upper())
        
        if not asset:
            print(f"❌ Asset not found: {args.ticker}")
            print("   Use 'add-asset' to add it first.")
            return 1
        
        # Use today's date if not specified
        buy_date = args.date or datetime.now().strftime("%Y-%m-%d")
        
        position = position_repo.create(
            asset_id=asset.id,
            buy_date=buy_date,
            shares=args.shares,
            buy_price=args.price,
        )
        
        print(f"✅ Added position for {asset.ticker}")
        print(f"   Shares: {args.shares}")
        print(f"   Price: ${args.price:.2f}")
        print(f"   Date: {buy_date}")
        print(f"   Cost: ${args.shares * args.price:,.2f}")


def cmd_update(args):
    """Run daily data update."""
    from jobs.daily_update import run_daily_update
    return run_daily_update()


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/app.py",
        "--server.headless", "true",
    ])


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
        print(f"   Total Cost:      ${summary['total_cost']:>12,.0f}")
        print(f"   Market Value:    ${summary['total_market_value']:>12,.0f}")
        print(f"   Unrealized P&L:  ${summary['total_pnl']:>+12,.0f} ({summary['total_pnl_pct']:+.1%})")
        
        print(f"\n⚠️ Risk")
        print(f"   Portfolio Vol:   {risk['portfolio_volatility']:>12.1%}")
        print(f"   Max Drawdown:    {risk['portfolio_max_drawdown']:>12.1%}")
        
        print(f"\n📈 Positions ({len(df)})")
        for _, row in df.iterrows():
            pnl_pct = row['pnl_pct']
            print(f"   {row['ticker']:<6} {row['weight']:>6.1%}  P&L: {pnl_pct:>+6.1%}")
        
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
    add_asset.add_argument("--name", help="Company name")
    add_asset.add_argument("--sector", help="Sector")
    add_asset.add_argument("--exchange", help="Exchange")
    
    # add-position command
    add_pos = subparsers.add_parser("add-position", help="Add position (buy lot)")
    add_pos.add_argument("ticker", help="Stock ticker symbol")
    add_pos.add_argument("--shares", type=float, required=True, help="Number of shares")
    add_pos.add_argument("--price", type=float, required=True, help="Buy price per share")
    add_pos.add_argument("--date", help="Buy date (YYYY-MM-DD)")
    
    # update command
    subparsers.add_parser("update", help="Run daily data update")
    
    # dashboard command
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    
    # summary command
    subparsers.add_parser("summary", help="Show portfolio summary")
    
    # list command
    subparsers.add_parser("list", help="List tracked assets")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "init": cmd_init,
        "add-asset": cmd_add_asset,
        "add-position": cmd_add_position,
        "update": cmd_update,
        "dashboard": cmd_dashboard,
        "summary": cmd_summary,
        "list": cmd_list_assets,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
