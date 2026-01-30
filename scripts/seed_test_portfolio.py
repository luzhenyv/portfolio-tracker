"""
Seed Test Portfolio Data Script.

Creates a test portfolio with:
- Initial cash deposit on 2025-01-01
- Trades from backups/january_report/trades.csv with randomized dates in 2025
- Market index data for the same period

Usage:
    python -m scripts.seed_test_portfolio
    python -m scripts.seed_test_portfolio --if-drop
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta, timezone

from db import init_db, get_db, AssetStatus
from db.repositories import (
    CashRepository,
    MarketIndexRepository,
    IndexPriceRepository,
)
from services.asset_service import create_asset_with_data
from services.position_service import buy_position


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Portfolio configuration based on trades.csv and assets.csv
INITIAL_DEPOSIT = 23489.68
DEPOSIT_DATE = datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)

# Trades from trades.csv (asset_id, ticker, shares, price)
TRADES_DATA = [
    (1, "TSLA", 12.0, 190.986),
    (2, "NVDA", 25.0, 179.366),
    (3, "AVGO", 12.0, 342.588),
    (4, "META", 4.0, 679.475),
    (5, "GOOGL", 5.0, 91.37),
    (6, "HIMS", 43.0, 42.977),
    (7, "MRVL", 8.0, 37.88),
]

# Market indices to seed with fake data
DEFAULT_INDICES = [
    {
        "symbol": "SPX",
        "name": "S&P 500",
        "description": "Large-cap US equity market benchmark tracking 500 leading companies",
        "category": "EQUITY",
        "base_price": 5900.0,
        "volatility": 0.012,  # Daily volatility
    },
    {
        "symbol": "RUT",
        "name": "Russell 2000",
        "description": "Small-cap US equity market benchmark tracking 2000 small companies",
        "category": "EQUITY",
        "base_price": 2250.0,
        "volatility": 0.015,
    },
    {
        "symbol": "VIX",
        "name": "CBOE Volatility Index",
        "description": "Market volatility expectation derived from S&P 500 options",
        "category": "VOLATILITY",
        "base_price": 16.0,
        "volatility": 0.05,
    },
    {
        "symbol": "DJI",
        "name": "Dow Jones Industrial Average",
        "description": "Price-weighted index of 30 large-cap US blue-chip companies",
        "category": "EQUITY",
        "base_price": 43000.0,
        "volatility": 0.010,
    },
    {
        "symbol": "IXIC",
        "name": "NASDAQ Composite",
        "description": "Tech-heavy index tracking all NASDAQ-listed stocks",
        "category": "EQUITY",
        "base_price": 19500.0,
        "volatility": 0.014,
    },
]


def generate_random_trade_datetime(start_date: str = "2025-01-02", end_date: str = "2025-12-31") -> datetime:
    """
    Generate a random datetime within the specified date range.
    
    Returns datetime object with timezone.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=13, minute=59, second=59, tzinfo=timezone.utc)
    
    # Random seconds between start and end
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    random_dt = start + timedelta(seconds=random_seconds)
    
    # Clamp time to trading hours (9:30 AM - 4:00 PM EST-ish)
    if random_dt.hour < 9:
        random_dt = random_dt.replace(hour=9, minute=30)
    elif random_dt.hour >= 16:
        random_dt = random_dt.replace(hour=15, minute=59)
    
    return random_dt


def generate_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate list of trading days (weekdays) between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    trading_days = []
    current = start
    while current <= end:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            trading_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return trading_days


def generate_fake_price_series(
    base_price: float,
    volatility: float,
    trading_days: list[str],
    trend: float = 0.0002,  # Slight upward drift
) -> list[dict]:
    """
    Generate a fake price series using geometric Brownian motion.
    
    Args:
        base_price: Starting price
        volatility: Daily volatility (e.g., 0.01 = 1%)
        trading_days: List of date strings
        trend: Daily drift rate
    
    Returns:
        List of dicts with {date, open, high, low, close, volume}
    """
    prices = []
    current_price = base_price
    
    for date_str in trading_days:
        # Random return using normal distribution
        daily_return = random.gauss(trend, volatility)
        
        # Calculate OHLC
        open_price = current_price
        
        # Intraday range
        intraday_vol = volatility * 0.5
        high_return = abs(random.gauss(0, intraday_vol))
        low_return = abs(random.gauss(0, intraday_vol))
        
        high_price = open_price * (1 + high_return + max(0, daily_return))
        low_price = open_price * (1 - low_return + min(0, daily_return))
        close_price = open_price * (1 + daily_return)
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Random volume
        base_volume = int(base_price * 1000000 / 50)  # Scale volume with price
        volume = int(base_volume * random.uniform(0.5, 2.0))
        
        prices.append({
            "date": date_str,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume,
        })
        
        current_price = close_price
    
    return prices


def seed_test_portfolio(if_drop: bool = False) -> dict:
    """
    Seed the database with test portfolio data.
    
    Args:
        if_drop: Whether to drop existing tables first.
    
    Returns:
        Dict with seeding results.
    """
    # Initialize database
    db = init_db(if_drop=if_drop)
    logger.info(f"✅ Database initialized at {db.db_url}")
    
    results = {
        "cash_deposited": 0,
        "assets_created": [],
        "trades_executed": [],
        "indices_created": [],
        "index_prices_seeded": [],
        "errors": [],
    }
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Seed initial cash deposit
    logger.info("\n💵 Seeding initial cash deposit...")
    with db.session() as session:
        cash_repo = CashRepository(session)
        cash_repo.deposit(
            INITIAL_DEPOSIT,
            transaction_at=DEPOSIT_DATE,
            description="Initial capital deposit",
        )
        session.commit()
    results["cash_deposited"] = INITIAL_DEPOSIT
    logger.info(f"✅ Deposited ${INITIAL_DEPOSIT:,.2f} on {DEPOSIT_DATE.strftime('%Y-%m-%d')}")
    
    # Step 2: Create assets
    logger.info("\n📈 Creating assets...")
    for _, ticker, _, _ in TRADES_DATA:
        result = create_asset_with_data(ticker, AssetStatus.OWNED)
        if result.success:
            logger.info(f"✅ Created asset: {ticker}")
            results["assets_created"].append(ticker)
        else:
            logger.warning(f"⚠️ Failed to create {ticker}: {', '.join(result.errors)}")
            results["errors"].append(f"Asset {ticker}: {', '.join(result.errors)}")
    
    # Step 3: Execute trades with random dates
    logger.info("\n💼 Executing trades with random dates...")
    
    # Generate random dates for each trade, then sort them chronologically
    trade_dates = []
    for _ in TRADES_DATA:
        trade_dt = generate_random_trade_datetime("2025-01-02", "2025-12-31")
        trade_dates.append(trade_dt)
    
    # Sort trades by date
    trades_with_dates = list(zip(TRADES_DATA, trade_dates))
    trades_with_dates.sort(key=lambda x: x[1])
    
    for (asset_id, ticker, shares, price), trade_at in trades_with_dates:
        result = buy_position(
            ticker=ticker,
            shares=shares,
            price=price,
            trade_at=trade_at,
            fees=0.0,
        )
        if result.success:
            logger.info(f"✅ BUY {shares} {ticker} @ ${price:.2f} on {trade_at.strftime('%Y-%m-%d %H:%M:%S')}")
            results["trades_executed"].append({
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "trade_at": trade_at.strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            logger.warning(f"⚠️ Failed to buy {ticker}: {', '.join(result.errors)}")
            results["errors"].append(f"Trade {ticker}: {', '.join(result.errors)}")
    
    # Step 4: Seed market indices
    logger.info("\n📊 Seeding market indices...")
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        
        for idx_data in DEFAULT_INDICES:
            index, created = index_repo.get_or_create(
                symbol=idx_data["symbol"],
                name=idx_data["name"],
                description=idx_data.get("description"),
                category=idx_data.get("category", "EQUITY"),
            )
            
            status = "created" if created else "exists"
            logger.info(f"✅ Index {idx_data['symbol']} ({idx_data['name']}): {status}")
            results["indices_created"].append({
                "symbol": idx_data["symbol"],
                "created": created,
            })
        
        session.commit()
    
    # Step 5: Generate fake index price data
    logger.info("\n📈 Generating fake index price data...")
    trading_days = generate_trading_days("2025-01-02", "2025-12-31")
    logger.info(f"   Generated {len(trading_days)} trading days for 2025")
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        price_repo = IndexPriceRepository(session)
        
        for idx_data in DEFAULT_INDICES:
            index = index_repo.get_by_symbol(idx_data["symbol"])
            if not index:
                continue
            
            # Generate fake prices
            prices = generate_fake_price_series(
                base_price=idx_data["base_price"],
                volatility=idx_data["volatility"],
                trading_days=trading_days,
            )
            
            # Bulk insert prices
            count = price_repo.bulk_upsert_prices(index.id, prices)
            logger.info(f"   📊 {idx_data['symbol']}: {count} price records seeded")
            results["index_prices_seeded"].append({
                "symbol": idx_data["symbol"],
                "records": count,
            })
        
        session.commit()
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Seed test portfolio data for development and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.seed_test_portfolio
    python -m scripts.seed_test_portfolio --if-drop
        """,
    )
    parser.add_argument(
        "--if-drop",
        action="store_true",
        help="Drop existing tables before creating (fresh start)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🌱 Seeding Test Portfolio Data")
    logger.info("=" * 60)
    
    try:
        results = seed_test_portfolio(if_drop=args.if_drop)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 Seeding Summary")
        logger.info("=" * 60)
        logger.info(f"   💵 Cash Deposited: ${results['cash_deposited']:,.2f}")
        logger.info(f"   📈 Assets Created: {len(results['assets_created'])}")
        logger.info(f"   💼 Trades Executed: {len(results['trades_executed'])}")
        logger.info(f"   📊 Indices Created: {len(results['indices_created'])}")
        
        total_index_records = sum(r["records"] for r in results["index_prices_seeded"])
        logger.info(f"   📈 Index Price Records: {total_index_records}")
        
        if results["errors"]:
            logger.info(f"\n   ❌ Errors: {len(results['errors'])}")
            for error in results["errors"]:
                logger.info(f"      - {error}")
        
        logger.info("\n🎉 Test portfolio seeding complete!")
        logger.info("   Run 'streamlit run ui/app.py' to view the dashboard.")
        
        return 0 if not results["errors"] else 1
        
    except Exception as e:
        logger.error(f"❌ Error seeding data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
