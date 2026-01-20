"""
Seed Fake Data Script for Portfolio Analytics System.

Creates a fresh SQLite database with sample assets, trades, and cash transactions
for testing and development purposes.

Usage:
    python scripts/seed_fake_data.py --if-drop
    python scripts/seed_fake_data.py --db-url sqlite:///./test.db --if-drop
"""

import argparse
import sys

from db import init_db, AssetStatus
from db.repositories import CashRepository
from services.asset_service import create_asset_with_data
from services.position_service import buy_position, sell_position


def seed_fake_data(db_url=None, if_drop=False):
    """Seed the database with fake data."""
    print("🌱 Seeding fake data...")

    # Initialize database
    db = init_db(db_url, if_drop)
    print(f"✅ Database initialized at {db.db_url}")

    # Seed cash transactions
    print("\n💵 Seeding cash transactions...")
    with db.session() as session:
        cash_repo = CashRepository(session)
        cash_repo.deposit(100000.0, transaction_date="2025-01-01", description="Initial capital")
        cash_repo.withdraw(500.0, transaction_date="2025-12-25", description="Personal expense")
    print("✅ Cash transactions seeded")

    # Seed assets
    print("\n📈 Seeding assets...")
    assets = [
        ("TSLA", AssetStatus.OWNED),
        ("NVDA", AssetStatus.WATCHLIST),
        ("KO", AssetStatus.WATCHLIST),
        ("NFLX", AssetStatus.WATCHLIST),
    ]
    for ticker, status in assets:
        result = create_asset_with_data(ticker, status)
        if result.success:
            print(f"✅ Added {ticker}")
        else:
            print(f"⚠️ Failed to add {ticker}: {', '.join(result.errors)}")
    print("✅ Assets seeded")

    # Seed trades
    print("\n💼 Seeding trades...")
    trades = [
        ("buy", "TSLA", 100, 150.00, "2025-01-15"),
        ("sell", "TSLA", 50, 170.00, "2025-06-15"),
        ("buy", "NVDA", 50, 123.00, "2025-09-15"),
        ("buy", "KO", 50, 64.30, "2025-10-23"),
    ]
    for action, ticker, shares, price, date in trades:
        if action == "buy":
            result = buy_position(ticker=ticker, shares=shares, price=price, trade_date=date, fees=0.0)
        elif action == "sell":
            result = sell_position(ticker=ticker, shares=shares, price=price, trade_date=date, fees=0.0)
        if result.success:
            print(f"✅ {action.upper()} {shares} {ticker} @ ${price}")
        else:
            print(f"⚠️ Failed to {action} {ticker}: {', '.join(result.errors)}")
    print("✅ Trades seeded")

    print("\n🎉 Fake data seeding complete!")
    print("Run 'python main.py summary' to see the portfolio.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Seed fake data for Portfolio Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-url",
        help="Custom database URL (default: from config)",
        default=None,
    )
    parser.add_argument(
        "--if-drop",
        action="store_true",
        help="Drop existing tables before creating",
    )

    args = parser.parse_args()

    try:
        seed_fake_data(db_url=args.db_url, if_drop=args.if_drop)
        return 0
    except Exception as e:
        print(f"❌ Error seeding data: {e}")
        return 1


if __name__ == "__main__":
    # Seed default DB (overwrites existing)
    # python scripts/seed_fake_data.py --if-drop
    
    # Seed custom DB
    # python scripts/seed_fake_data.py --db-url sqlite:///./test.db --if-drop
    sys.exit(main())