#!/usr/bin/env python3
"""
Migration script to add net_invested column and recalculate values from trade history.

Run this once to:
1. Add net_invested column to positions table if not exists
2. Recalculate net_invested for all existing positions from trade history

Usage:
    python scripts/migrate_net_invested.py
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db import get_db
from db.repositories import PositionRepository


def add_column_if_not_exists(db_path: str) -> bool:
    """Add net_invested column to positions table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(positions)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "net_invested" not in columns:
        print("Adding net_invested column to positions table...")
        cursor.execute("""
            ALTER TABLE positions 
            ADD COLUMN net_invested REAL NOT NULL DEFAULT 0
        """)
        conn.commit()
        print("✅ Column added successfully")
        added = True
    else:
        print("ℹ️ net_invested column already exists")
        added = False
    
    conn.close()
    return added


def recalculate_all_net_invested() -> dict[int, float]:
    """Recalculate net_invested for all positions from trade history."""
    print("\nRecalculating net_invested for all positions...")
    
    db = get_db()
    with db.session() as session:
        repo = PositionRepository(session)
        result = repo.recalculate_all_net_invested()
    
    return result


def main():
    """Main migration function."""
    print("=" * 60)
    print("Net Invested Migration Script")
    print("=" * 60)
    
    # Find database path
    from config import config
    db_path = config.database.path
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at: {db_path}")
        sys.exit(1)
    
    print(f"Database: {db_path}\n")
    
    # Step 1: Add column
    add_column_if_not_exists(db_path)
    
    # Step 2: Recalculate values
    result = recalculate_all_net_invested()
    
    if result:
        print(f"\n✅ Updated {len(result)} position(s):")
        
        # Get ticker names for display
        db = get_db()
        with db.session() as session:
            from db.repositories import AssetRepository, PositionRepository
            asset_repo = AssetRepository(session)
            pos_repo = PositionRepository(session)
            
            for asset_id, net_invested in result.items():
                asset = asset_repo.get_by_id(asset_id)
                position = pos_repo.get_by_asset_id(asset_id)
                
                if asset and position and position.long_shares > 0:
                    net_invested_avg_cost = net_invested / position.long_shares
                    print(f"   {asset.ticker}: net_invested=${net_invested:,.2f} "
                          f"({position.long_shares:.0f} shares @ ${net_invested_avg_cost:.2f} net avg cost)")
    else:
        print("\nℹ️ No positions to update")
    
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
