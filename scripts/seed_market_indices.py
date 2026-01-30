"""
Seed Market Indices Script.

Populates the market_indices table with default benchmark indices
for portfolio comparison and correlation analysis.

Usage:
    python -m scripts.seed_market_indices
"""

import logging
import sys
from datetime import datetime, timedelta

from config import config
from db import init_db, get_db
from db.repositories import MarketIndexRepository, IndexPriceRepository
from data.yfinance_fetcher import IndexPriceFetcher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default indices to seed (used if config.market_indices is not available)
DEFAULT_INDICES = [
    {
        "symbol": "SPX",
        "name": "S&P 500",
        "description": "Large-cap US equity market benchmark tracking 500 leading companies",
        "category": "EQUITY",
        "yahoo_symbol": "^GSPC",
    },
    {
        "symbol": "RUT",
        "name": "Russell 2000",
        "description": "Small-cap US equity market benchmark tracking 2000 small companies",
        "category": "EQUITY",
        "yahoo_symbol": "^RUT",
    },
    {
        "symbol": "VIX",
        "name": "CBOE Volatility Index",
        "description": "Market volatility expectation derived from S&P 500 options",
        "category": "VOLATILITY",
        "yahoo_symbol": "^VIX",
    },
    {
        "symbol": "DJI",
        "name": "Dow Jones Industrial Average",
        "description": "Price-weighted index of 30 large-cap US blue-chip companies",
        "category": "EQUITY",
        "yahoo_symbol": "^DJI",
    },
    {
        "symbol": "IXIC",
        "name": "NASDAQ Composite",
        "description": "Tech-heavy index tracking all NASDAQ-listed stocks",
        "category": "EQUITY",
        "yahoo_symbol": "^IXIC",
    },
    {
        "symbol": "GOLD",
        "name": "Gold Futures",
        "description": "Gold commodity price benchmark",
        "category": "COMMODITY",
        "yahoo_symbol": "GC=F",
    },
]


def get_indices_to_seed() -> list[dict]:
    """
    Get the list of indices to seed.
    
    Uses config.market_indices if available, otherwise falls back to DEFAULT_INDICES.
    """
    try:
        if hasattr(config, 'market_indices') and config.market_indices.tracked_indices:
            indices = []
            for idx_config in config.market_indices.tracked_indices:
                indices.append({
                    "symbol": idx_config.symbol,
                    "name": idx_config.name,
                    "description": idx_config.description,
                    "category": idx_config.category,
                    "yahoo_symbol": idx_config.sources.get("yahoo", idx_config.symbol),
                })
            return indices
    except AttributeError:
        pass
    
    return DEFAULT_INDICES


def seed_indices(fetch_prices: bool = True, lookback_days: int | None = None) -> dict:
    """
    Seed market indices into the database.
    
    Args:
        fetch_prices: Whether to fetch historical prices after seeding.
        lookback_days: Optional number of days to fetch prices for (overrides config).
    
    Returns:
        Dict with seeding results.
    """
    init_db()
    db = get_db()
    
    indices_to_seed = get_indices_to_seed()
    
    results = {
        "created": [],
        "existing": [],
        "errors": [],
        "price_fetch_results": [],
    }
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        
        for idx_data in indices_to_seed:
            try:
                index, created = index_repo.get_or_create(
                    symbol=idx_data["symbol"],
                    name=idx_data["name"],
                    description=idx_data.get("description"),
                    category=idx_data.get("category", "EQUITY"),
                )
                
                if created:
                    logger.info(f"✅ Created index: {idx_data['symbol']} ({idx_data['name']})")
                    results["created"].append(idx_data["symbol"])
                else:
                    logger.info(f"⏭️  Index already exists: {idx_data['symbol']}")
                    results["existing"].append(idx_data["symbol"])
                    
            except Exception as e:
                logger.error(f"❌ Failed to seed {idx_data['symbol']}: {e}")
                results["errors"].append(f"{idx_data['symbol']}: {e}")
        
        session.commit()
    
    # Fetch historical prices if requested
    if fetch_prices:
        logger.info("\n📈 Fetching historical prices for seeded indices...")
        price_fetcher = IndexPriceFetcher()
        
        with db.session() as session:
            index_repo = MarketIndexRepository(session)
            price_repo = IndexPriceRepository(session)
            
            for idx_data in indices_to_seed:
                index = index_repo.get_by_symbol(idx_data["symbol"])
                if not index:
                    continue
                
                # Determine start date
                if lookback_days:
                    start_dt = datetime.now() - timedelta(days=lookback_days)
                else:
                    start_dt = datetime.now() - timedelta(days=config.data_fetcher.default_lookback_days)
                start_date = start_dt.strftime("%Y-%m-%d")
                
                # Fetch using Yahoo Finance symbol
                yahoo_symbol = idx_data.get("yahoo_symbol", idx_data["symbol"])
                fetch_result = price_fetcher.fetch_for_index(yahoo_symbol, start_date)
                
                if fetch_result.success and fetch_result.records:
                    count = price_repo.bulk_upsert_prices(index.id, fetch_result.records)
                    logger.info(f"   📊 {idx_data['symbol']}: {count} price records stored")
                    results["price_fetch_results"].append({
                        "symbol": idx_data["symbol"],
                        "success": True,
                        "records": count,
                    })
                else:
                    logger.warning(f"   ⚠️ {idx_data['symbol']}: {fetch_result.message}")
                    results["price_fetch_results"].append({
                        "symbol": idx_data["symbol"],
                        "success": False,
                        "message": fetch_result.message,
                    })
            
            session.commit()
    
    return results


def main():
    """Main entry point for seeding script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed market indices into the database")
    parser.add_argument(
        "--no-prices",
        action="store_true",
        help="Skip fetching historical prices after seeding",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Number of days of historical prices to fetch (default: config.data_fetcher.default_lookback_days)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("🌱 Seeding Market Indices")
    logger.info("=" * 50)
    
    results = seed_indices(
        fetch_prices=not args.no_prices,
        lookback_days=args.lookback,
    )
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 Seeding Summary")
    logger.info("=" * 50)
    logger.info(f"   Created: {len(results['created'])} indices")
    logger.info(f"   Existing: {len(results['existing'])} indices")
    logger.info(f"   Errors: {len(results['errors'])}")
    
    if results["price_fetch_results"]:
        successful_fetches = sum(1 for r in results["price_fetch_results"] if r["success"])
        total_records = sum(r.get("records", 0) for r in results["price_fetch_results"])
        logger.info(f"   Price Fetches: {successful_fetches}/{len(results['price_fetch_results'])} successful")
        logger.info(f"   Total Records: {total_records}")
    
    if results["errors"]:
        logger.info("\n❌ Errors:")
        for error in results["errors"]:
            logger.info(f"   - {error}")
    
    return 0 if not results["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
