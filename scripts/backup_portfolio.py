"""
Backup Script for Portfolio Analytics System.

Provides tools to:
1. Create a consistent physical snapshot of the SQLite database.
2. Export core ledgers (Assets, Trades, Cash Transactions) to CSV for audit.

Usage:
    python scripts/backup_portfolio.py --format sqlite --out backups/portfolio_backup.db
    python scripts/backup_portfolio.py --format csv --out backups/export_20250125/
"""

import argparse
import csv
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatabaseConfig
from db.session import DatabaseManager
from db.models import Asset, Trade, CashTransaction


def backup_sqlite(src_path: str, dest_path: str):
    """Create a consistent physical backup of the SQLite database."""
    print(f"📦 Creating physical backup: {src_path} -> {dest_path}")
    
    # Ensure destination directory exists
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    src_conn = sqlite3.connect(src_path)
    dest_conn = sqlite3.connect(dest_path)
    
    with dest_conn:
        src_conn.backup(dest_conn)
        
    dest_conn.close()
    src_conn.close()
    print(f"✅ Physical backup complete at {dest_path}")


def export_to_csv(out_dir: str):
    """Export core ledgers to CSV files."""
    print(f"📊 Exporting core ledgers to CSV in {out_dir}...")
    
    # Ensure output directory exists
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    db = DatabaseManager()
    
    with db.session() as session:
        # Export Assets
        assets = session.query(Asset).all()
        assets_file = out_path / "assets.csv"
        with open(assets_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "ticker", "name", "sector", "status", "created_at"])
            for a in assets:
                writer.writerow([a.id, a.ticker, a.name, a.sector, a.status, a.created_at])
        print(f"  - Written {len(assets)} assets to {assets_file}")

        # Export Trades
        trades = session.query(Trade).all()
        trades_file = out_path / "trades.csv"
        with open(trades_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "asset_id", "trade_date", "action", "shares", "price", "fees", "realized_pnl", "created_at"])
            for t in trades:
                writer.writerow([t.id, t.asset_id, t.trade_date, t.action, t.shares, t.price, t.fees, t.realized_pnl, t.created_at])
        print(f"  - Written {len(trades)} trades to {trades_file}")

        # Export Cash Transactions
        cash = session.query(CashTransaction).all()
        cash_file = out_path / "cash_transactions.csv"
        with open(cash_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "date", "type", "amount", "asset_id", "trade_id", "description", "created_at"])
            for c in cash:
                writer.writerow([c.id, c.transaction_date, c.transaction_type, c.amount, c.asset_id, c.trade_id, c.description, c.created_at])
        print(f"  - Written {len(cash)} cash transactions to {cash_file}")

    print(f"✅ Export complete.")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Backup Tool")
    parser.add_argument(
        "--format", 
        choices=["sqlite", "csv"], 
        default="sqlite",
        help="Backup format: 'sqlite' for full DB snapshot, 'csv' for ledger export"
    )
    parser.add_argument(
        "--out", 
        help="Output path (file for sqlite, directory for csv)"
    )
    parser.add_argument(
        "--db-path", 
        help="Override source database path"
    )

    args = parser.parse_args()
    
    # Resolve source path
    src_path = args.db_path if args.db_path else str(DatabaseConfig().path)
    if not os.path.exists(src_path):
        print(f"❌ Error: Database not found at {src_path}")
        sys.exit(1)

    # Resolve output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.format == "sqlite":
        out_path = args.out if args.out else f"backups/portfolio_{timestamp}.db"
        backup_sqlite(src_path, out_path)
    else:
        out_path = args.out if args.out else f"backups/export_{timestamp}/"
        export_to_csv(out_path)


if __name__ == "__main__":
    main()
