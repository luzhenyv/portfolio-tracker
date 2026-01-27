"""
Database migration script for Yahoo-aligned valuation fields.

This script adds new columns to valuation_metrics and valuation_metric_overrides
tables to support Yahoo Finance's Statistics page layout.

Run this script to migrate existing databases:
    python scripts/migrate_yahoo_valuation.py
"""

import sqlite3
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_db_path() -> str:
    """Get the database path from config or environment."""
    # Try environment variable first
    env_path = os.environ.get("PORTFOLIO_DB_PATH")
    if env_path:
        return env_path
    
    # Try to import from config
    try:
        from config import config
        return str(config.database.path)
    except ImportError:
        pass
    
    # Default fallback
    return "db/portfolio.db"


def column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate_valuation_metrics(cursor: sqlite3.Cursor):
    """Add new Yahoo-aligned columns to valuation_metrics table."""
    new_columns = [
        # Valuation Measures
        ("market_cap", "REAL"),
        ("enterprise_value", "REAL"),
        ("pe_trailing", "REAL"),
        ("price_to_sales", "REAL"),
        ("price_to_book", "REAL"),
        ("ev_to_revenue", "REAL"),
        # Financial Highlights - Profitability
        ("profit_margin", "REAL"),
        ("return_on_assets", "REAL"),
        ("return_on_equity", "REAL"),
        # Financial Highlights - Income Statement
        ("revenue_ttm", "REAL"),
        ("net_income_ttm", "REAL"),
        ("diluted_eps_ttm", "REAL"),
        # Financial Highlights - Balance Sheet & Cash Flow
        ("total_cash", "REAL"),
        ("total_debt_to_equity", "REAL"),
        ("levered_free_cash_flow", "REAL"),
    ]
    
    added = []
    for column, col_type in new_columns:
        if not column_exists(cursor, "valuation_metrics", column):
            cursor.execute(f"ALTER TABLE valuation_metrics ADD COLUMN {column} {col_type}")
            added.append(column)
    
    return added


def migrate_valuation_overrides(cursor: sqlite3.Cursor):
    """Add new Yahoo-aligned override columns to valuation_metric_overrides table."""
    new_columns = [
        # Valuation Measures overrides
        ("market_cap_override", "REAL"),
        ("enterprise_value_override", "REAL"),
        ("pe_trailing_override", "REAL"),
        ("price_to_sales_override", "REAL"),
        ("price_to_book_override", "REAL"),
        ("ev_to_revenue_override", "REAL"),
        # Financial Highlights - Profitability overrides
        ("profit_margin_override", "REAL"),
        ("return_on_assets_override", "REAL"),
        ("return_on_equity_override", "REAL"),
        # Financial Highlights - Income Statement overrides
        ("revenue_ttm_override", "REAL"),
        ("net_income_ttm_override", "REAL"),
        ("diluted_eps_ttm_override", "REAL"),
        # Financial Highlights - Balance Sheet & Cash Flow overrides
        ("total_cash_override", "REAL"),
        ("total_debt_to_equity_override", "REAL"),
        ("levered_free_cash_flow_override", "REAL"),
    ]
    
    added = []
    for column, col_type in new_columns:
        if not column_exists(cursor, "valuation_metric_overrides", column):
            cursor.execute(f"ALTER TABLE valuation_metric_overrides ADD COLUMN {column} {col_type}")
            added.append(column)
    
    return added


def main():
    """Run the migration."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("   The new schema will be created automatically when the app starts.")
        return
    
    print(f"📦 Migrating database: {db_path}")
    print(f"   Started at: {datetime.now().isoformat()}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Migrate valuation_metrics
        print("\n🔧 Migrating valuation_metrics table...")
        added_metrics = migrate_valuation_metrics(cursor)
        if added_metrics:
            print(f"   Added columns: {', '.join(added_metrics)}")
        else:
            print("   No new columns needed (already up to date)")
        
        # Migrate valuation_metric_overrides
        print("\n🔧 Migrating valuation_metric_overrides table...")
        added_overrides = migrate_valuation_overrides(cursor)
        if added_overrides:
            print(f"   Added columns: {', '.join(added_overrides)}")
        else:
            print("   No new columns needed (already up to date)")
        
        conn.commit()
        print("\n✅ Migration completed successfully!")
        
        # Reminder
        if added_metrics or added_overrides:
            print("\n💡 Next steps:")
            print("   1. Run 'python main.py fetch' to populate new valuation fields")
            print("   2. Restart the dashboard to see Yahoo-aligned tables")
    
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
