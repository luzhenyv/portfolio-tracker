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
from db.models import Asset, AssetStatus, Position


def create_sample_data():
    """
    Create sample data for testing/demo purposes.
    
    This is optional and can be skipped in production.
    """
    db = get_db()
    
    sample_assets = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "status": AssetStatus.OWNED},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "status": AssetStatus.OWNED},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "status": AssetStatus.WATCHLIST},
    ]
    
    sample_positions = [
        {"ticker": "AAPL", "shares": 50, "buy_price": 150.00, "buy_date": "2024-01-15"},
        {"ticker": "MSFT", "shares": 30, "buy_price": 375.00, "buy_date": "2024-02-01"},
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
                )
                session.add(asset)
                print(f"  Added asset: {asset_data['ticker']}")
        
        session.flush()
        
        # Add positions
        for pos_data in sample_positions:
            asset = session.query(Asset).filter_by(ticker=pos_data["ticker"]).first()
            if asset:
                # Check if position exists
                existing = session.query(Position).filter_by(
                    asset_id=asset.id,
                    buy_date=pos_data["buy_date"],
                ).first()
                
                if not existing:
                    position = Position(
                        asset_id=asset.id,
                        shares=pos_data["shares"],
                        buy_price=pos_data["buy_price"],
                        buy_date=pos_data["buy_date"],
                    )
                    session.add(position)
                    print(f"  Added position: {pos_data['ticker']} x {pos_data['shares']}")


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
