"""
Database initialization script.

Creates all tables and optionally seeds with sample data.
Safe to run multiple times (idempotent).
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.session import init_db, get_db
from db.models import Asset, AssetStatus, AssetType
from services.position_service import execute_trade
from db import TradeAction


def create_sample_data():
    """
    Create sample data for testing/demo purposes using Trade ledger.
    
    This is optional and can be skipped in production.
    Uses the new Trade-based position tracking system.
    """
    db = get_db()
    
    sample_assets = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "status": AssetStatus.OWNED, "asset_type": AssetType.STOCK},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "status": AssetStatus.OWNED, "asset_type": AssetType.STOCK},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "status": AssetStatus.WATCHLIST, "asset_type": AssetType.STOCK},
    ]
    
    sample_trades = [
        {"ticker": "AAPL", "shares": 50, "price": 150.00, "date": "2024-01-15"},
        {"ticker": "MSFT", "shares": 30, "price": 375.00, "date": "2024-02-01"},
    ]
    
    with db.session() as session:
        # Add assets
        for asset_data in sample_assets:
            existing = session.query(Asset).filter_by(ticker=asset_data["ticker"]).first()
            if not existing:
                asset = Asset(
                    ticker=asset_data["ticker"],
                    name=asset_data["name"],
                    sector=asset_data["sector"],
                    status=asset_data["status"],
                    asset_type=asset_data["asset_type"],
                )
                session.add(asset)
                print(f"  Added asset: {asset_data['ticker']}")
        
        session.commit()
    
    # Execute trades to create positions (uses Trade + Position state)
    for trade_data in sample_trades:
        result = execute_trade(
            ticker=trade_data["ticker"],
            action=TradeAction.BUY,
            shares=trade_data["shares"],
            price=trade_data["price"],
            trade_date=trade_data["date"],
        )
        if result.success:
            print(f"  Executed trade: {trade_data['ticker']} BUY {trade_data['shares']} @ ${trade_data['price']}")


def main():
    """Initialize database and optionally create sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize portfolio database")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create sample data for testing",
    )
    args = parser.parse_args()
    
    # Initialize database with tables
    db = init_db()
    print("✅ Database initialized")
    print(f"   Location: {db.db_url}")
    
    if args.sample_data:
        print("\n📦 Creating sample data...")
        create_sample_data()
        print("✅ Sample data created")


if __name__ == "__main__":
    main()
