"""
Backup Script for Portfolio Analytics System.

Provides tools to:
1. Create a consistent physical snapshot of the SQLite database.
2. Export core ledgers (Assets, Trades, Cash Transactions) to CSV for audit.

Usage:
    python scripts/backup_portfolio.py backup --format sqlite --out backups/test_snapshot.db
    python scripts/backup_portfolio.py restore --format csv --in backups/january_report/ --db-path db/test_restore.db
    python scripts/backup_portfolio.py restore --format sqlite --in backups/test_snapshot.db --db-path db/test_restore_sqlite.db
    python scripts/backup_portfolio.py restore \
        --format csv \
        --in backups/january_report/ \
        --db-url "postgresql+psycopg://portfolio_user:portfolio_password@localhost:5432/portfolio_tracker"

    docker compose exec app python scripts/backup_portfolio.py restore \
        --format csv \
        --in backups/january_report/ \
        --db-url "postgresql+psycopg://portfolio_user:portfolio_password@db:5432/portfolio_tracker"
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
from db.session import DatabaseManager, init_db
from db.models import Asset, Trade, CashTransaction, Position, AssetStatus, TradeAction, CashTransactionType


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


def restore_sqlite(src_path: str, dest_path: str):
    """Restore a physical SQLite backup."""
    print(f"🔄 Restoring physical backup: {src_path} -> {dest_path}")
    
    if not os.path.exists(src_path):
        print(f"❌ Error: Backup file not found at {src_path}")
        return

    # Ensure destination directory exists
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    src_conn = sqlite3.connect(src_path)
    dest_conn = sqlite3.connect(dest_path)
    
    with dest_conn:
        src_conn.backup(dest_conn)
        
    dest_conn.close()
    src_conn.close()
    print(f"✅ Restore complete at {dest_path}")


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
            writer.writerow(["id", "ticker", "name", "sector", "asset_type", "status", "created_at"])
            for a in assets:
                writer.writerow([a.id, a.ticker, a.name, a.sector, a.asset_type.value, a.status.value, a.created_at])
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


def restore_from_csv(in_dir: str, db_url: str = None):
    """Restore core ledgers from CSV files and rebuild positions."""
    print(f"🔄 Restoring from CSV in {in_dir}...")
    
    in_path = Path(in_dir)
    if not in_path.exists():
        print(f"❌ Error: Directory not found: {in_path}")
        return

    # Initialize a fresh database
    db = init_db(db_url, if_drop=True)
    
    with db.session() as session:
        # 0. Import required Enums
        from db import AssetType
        
        # 1. Restore Assets
        assets_file = in_path / "assets.csv"
        if assets_file.exists():
            print(f"  - Restoring assets from {assets_file}")
            with open(assets_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    asset = Asset(
                        id=int(row["id"]),
                        ticker=row["ticker"],
                        name=row["name"] or None,
                        sector=row["sector"] or None,
                        asset_type=AssetType(row["asset_type"]) if "asset_type" in row else AssetType.STOCK,
                        status=AssetStatus(row["status"]),
                        created_at=datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S.%f")
                    )
                    session.add(asset)
            session.flush()
        
        # 2. Restore Trades
        trades_file = in_path / "trades.csv"
        if trades_file.exists():
            print(f"  - Restoring trades from {trades_file}")
            with open(trades_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trade = Trade(
                        id=int(row["id"]),
                        asset_id=int(row["asset_id"]),
                        trade_date=row["trade_date"],
                        action=TradeAction(row["action"]),
                        shares=float(row["shares"]),
                        price=float(row["price"]),
                        fees=float(row["fees"]),
                        realized_pnl=float(row["realized_pnl"]),
                        created_at=datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S.%f")
                    )
                    session.add(trade)
            session.flush()

        # 3. Restore Cash Transactions
        cash_file = in_path / "cash_transactions.csv"
        if cash_file.exists():
            print(f"  - Restoring cash transactions from {cash_file}")
            with open(cash_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cash = CashTransaction(
                        id=int(row["id"]),
                        transaction_date=row["date"],
                        transaction_type=CashTransactionType(row["type"]),
                        amount=float(row["amount"]),
                        asset_id=int(row["asset_id"]) if row["asset_id"] else None,
                        trade_id=int(row["trade_id"]) if row["trade_id"] else None,
                        description=row["description"] or None,
                        created_at=datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S.%f")
                    )
                    session.add(cash)
            session.flush()

        session.commit()
        print("✅ Core ledgers restored.")

    # 4. Rebuild Positions
    print("🏗️ Rebuilding positions from trade history...")
    rebuild_positions(db)
    print("✅ Restore complete.")


def rebuild_positions(db: DatabaseManager):
    """Rebuild the positions table from restored trades."""
    from db.repositories import PositionRepository
    
    with db.session() as session:
        # Get all distinct asset IDs from trades
        asset_ids = [r[0] for r in session.query(Trade.asset_id).distinct().all()]
        
        pos_repo = PositionRepository(session)
        
        for aid in asset_ids:
            # 1. Calculate long/short shares, avg cost, and realized_pnl by replaying trades
            trades = session.query(Trade).filter(Trade.asset_id == aid).order_by(Trade.trade_date, Trade.id).all()
            
            position = Position(
                asset_id=aid,
                long_shares=0.0,
                short_shares=0.0,
                realized_pnl=0.0,
                net_invested=0.0
            )
            session.add(position)
            
            for t in trades:
                # Same logic as execute_trade but simplified for replay
                if t.action == TradeAction.BUY:
                    if position.short_shares > 0:
                        shares_to_cover = min(t.shares, position.short_shares)
                        # realized_pnl is already stored in the trade record during restore
                        position.short_shares -= shares_to_cover
                        if position.short_shares == 0:
                            position.short_avg_price = None
                        
                        remaining_shares = t.shares - shares_to_cover
                        if remaining_shares > 0:
                            if position.long_shares > 0:
                                total_cost = position.long_shares * position.long_avg_cost + remaining_shares * t.price
                                position.long_shares += remaining_shares
                                position.long_avg_cost = total_cost / position.long_shares
                            else:
                                position.long_shares = remaining_shares
                                position.long_avg_cost = t.price
                    else:
                        if position.long_shares > 0:
                            total_cost = position.long_shares * position.long_avg_cost + t.shares * t.price
                            position.long_shares += t.shares
                            position.long_avg_cost = total_cost / position.long_shares
                        else:
                            position.long_shares = t.shares
                            position.long_avg_cost = t.price
                            
                elif t.action == TradeAction.SELL:
                    if position.long_shares > 0:
                        shares_to_sell = min(t.shares, position.long_shares)
                        position.long_shares -= shares_to_sell
                        if position.long_shares == 0:
                            position.long_avg_cost = None
                        
                        remaining_shares = t.shares - shares_to_sell
                        if remaining_shares > 0:
                            if position.short_shares > 0:
                                total_proceeds = position.short_shares * position.short_avg_price + remaining_shares * t.price
                                position.short_shares += remaining_shares
                                position.short_avg_price = total_proceeds / position.short_shares
                            else:
                                position.short_shares = remaining_shares
                                position.short_avg_price = t.price
                    else:
                        if position.short_shares > 0:
                            total_proceeds = position.short_shares * position.short_avg_price + t.shares * t.price
                            position.short_shares += t.shares
                            position.short_avg_price = total_proceeds / position.short_shares
                        else:
                            position.short_shares = t.shares
                            position.short_avg_price = t.price

                elif t.action == TradeAction.SHORT:
                    # Treat SHORT like SELL for position counting
                    if position.long_shares > 0:
                        shares_to_sell = min(t.shares, position.long_shares)
                        position.long_shares -= shares_to_sell
                        if position.long_shares == 0:
                            position.long_avg_cost = None
                        remaining_shares = t.shares - shares_to_sell
                    else:
                        remaining_shares = t.shares
                    
                    if remaining_shares > 0:
                        if position.short_shares > 0:
                            total_proceeds = position.short_shares * position.short_avg_price + remaining_shares * t.price
                            position.short_shares += remaining_shares
                            position.short_avg_price = total_proceeds / position.short_shares
                        else:
                            position.short_shares = remaining_shares
                            position.short_avg_price = t.price

                elif t.action == TradeAction.COVER:
                    # Treat COVER like BUY for position counting
                    if position.short_shares > 0:
                        shares_to_cover = min(t.shares, position.short_shares)
                        position.short_shares -= shares_to_cover
                        if position.short_shares == 0:
                            position.short_avg_price = None
                        remaining_shares = t.shares - shares_to_cover
                    else:
                        remaining_shares = t.shares
                    
                    if remaining_shares > 0:
                        if position.long_shares > 0:
                            total_cost = position.long_shares * position.long_avg_cost + remaining_shares * t.price
                            position.long_shares += remaining_shares
                            position.long_avg_cost = total_cost / position.long_shares
                        else:
                            position.long_shares = remaining_shares
                            position.long_avg_cost = t.price
                
                # Update total realized P&L for the position
                position.realized_pnl += t.realized_pnl

            session.flush()
            
            # 2. Recalculate net_invested (which uses the full trade history anyway)
            pos_repo.recalculate_net_invested(aid)
            
        session.commit()


def main():
    parser = argparse.ArgumentParser(description="Portfolio Backup and Restore Tool")
    parser.add_argument(
        "operation",
        choices=["backup", "restore"],
        default="backup",
        nargs="?",
        help="Operation to perform: 'backup' (default) or 'restore'"
    )
    parser.add_argument(
        "--format", 
        choices=["sqlite", "csv"], 
        default="sqlite",
        help="Format: 'sqlite' for full DB snapshot, 'csv' for ledger export/import"
    )
    parser.add_argument(
        "--out", 
        help="Output path for backup (file for sqlite, directory for csv)"
    )
    parser.add_argument(
        "--in", 
        dest="input_path",
        help="Input path for restore (file for sqlite, directory for csv)"
    )
    parser.add_argument(
        "--db-path", 
        help="Override database path (source for backup, destination for restore)"
    )
    parser.add_argument(
        "--db-url", 
        help="Database URL for restore (e.g., postgresql+psycopg://user:pass@host:port/dbname)"
    )

    args = parser.parse_args()
    
    # Resolve DB path
    db_path = args.db_path if args.db_path else str(DatabaseConfig().path)

    if args.operation == "backup":
        if not os.path.exists(db_path):
            print(f"❌ Error: Database not found at {db_path}")
            sys.exit(1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.format == "sqlite":
            out_path = args.out if args.out else f"backups/portfolio_{timestamp}.db"
            backup_sqlite(db_path, out_path)
        else:
            out_path = args.out if args.out else f"backups/export_{timestamp}/"
            export_to_csv(out_path)
            
    elif args.operation == "restore":
        if not args.input_path:
            print("❌ Error: --in <path> is required for restore")
            sys.exit(1)
            
        if args.format == "sqlite":
            restore_sqlite(args.input_path, db_path)
        else:
            # Use provided db_url or construct sqlite URL
            if args.db_url:
                db_url = args.db_url
            else:
                db_url = f"sqlite:///{db_path}"
            restore_from_csv(args.input_path, db_url)


if __name__ == "__main__":
    main()
