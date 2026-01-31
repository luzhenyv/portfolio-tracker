#!/usr/bin/env python3
"""
Migration script to add asset tagging tables (tags, asset_tags).

Run this once to add:
1. tags table for tag definitions
2. asset_tags join table for many-to-many Asset<->Tag relationships

For new databases, these tables are created automatically by SQLAlchemy create_all().
This script is only needed for existing databases.

Usage:
    python scripts/migrate_asset_tags.py
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def add_tags_table(cursor: sqlite3.Cursor) -> bool:
    """Create tags table if it doesn't exist."""
    if check_table_exists(cursor, "tags"):
        print("ℹ️  tags table already exists")
        return False
    
    print("Creating tags table...")
    cursor.execute("""
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL UNIQUE,
            description VARCHAR(255),
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on name for fast lookups
    cursor.execute("CREATE INDEX idx_tags_name ON tags(name)")
    
    print("✅ tags table created")
    return True


def add_asset_tags_table(cursor: sqlite3.Cursor) -> bool:
    """Create asset_tags join table if it doesn't exist."""
    if check_table_exists(cursor, "asset_tags"):
        print("ℹ️  asset_tags table already exists")
        return False
    
    print("Creating asset_tags table...")
    cursor.execute("""
        CREATE TABLE asset_tags (
            asset_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (asset_id, tag_id),
            FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
    """)
    
    # Create index on tag_id for "find assets by tag" queries
    cursor.execute("CREATE INDEX idx_asset_tags_tag ON asset_tags(tag_id)")
    
    print("✅ asset_tags table created")
    return True


def verify_tables(cursor: sqlite3.Cursor) -> None:
    """Verify that both tables exist and print their structure."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    # Check tags table
    if check_table_exists(cursor, "tags"):
        cursor.execute("PRAGMA table_info(tags)")
        columns = cursor.fetchall()
        print("\ntags table columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    else:
        print("❌ tags table not found!")
    
    # Check asset_tags table
    if check_table_exists(cursor, "asset_tags"):
        cursor.execute("PRAGMA table_info(asset_tags)")
        columns = cursor.fetchall()
        print("\nasset_tags table columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    else:
        print("❌ asset_tags table not found!")


def main():
    """Main migration function."""
    print("=" * 60)
    print("Asset Tags Migration Script")
    print("=" * 60)
    
    # Find database path
    from config import config
    db_path = config.database.path
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at: {db_path}")
        print("   If this is a new installation, tables will be created automatically.")
        sys.exit(1)
    
    print(f"Database: {db_path}\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Step 1: Add tags table
        tags_created = add_tags_table(cursor)
        
        # Step 2: Add asset_tags table
        asset_tags_created = add_asset_tags_table(cursor)
        
        conn.commit()
        
        # Verify
        verify_tables(cursor)
        
        print("\n" + "=" * 60)
        if tags_created or asset_tags_created:
            print("✅ Migration completed successfully!")
        else:
            print("ℹ️  No migration needed - tables already exist")
        print("=" * 60)
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
