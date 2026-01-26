#!/usr/bin/env python3
"""
Manual Valuation Fetch Script

Manually fetch and store valuation data to fix missing/stale valuations.
Supports both:
- Watchlist valuation metrics (Forward P/E, PEG, EV/EBITDA, growth rates)
- Portfolio market value computation (positions × latest prices)

Usage:
    # Fetch valuation metrics for all assets
    python scripts/manual_valuation.py --mode multiples

    # Fetch for specific tickers only
    python scripts/manual_valuation.py --mode multiples --symbols AAPL,MSFT,GOOGL

    # Compute market value using latest prices
    python scripts/manual_valuation.py --mode market

    # Compute market value as of a specific date
    python scripts/manual_valuation.py --mode market --as-of 2026-01-20

    # Do both
    python scripts/manual_valuation.py --mode both

    # Force refetch even if recently updated
    python scripts/manual_valuation.py --mode multiples --force

    # Override specific prices manually
    python scripts/manual_valuation.py --mode market --override-price AAPL=192.10,MSFT=450.25
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

from config import config
from data.fetch_prices import PriceFetcher, ValuationFetcher
from db import get_db, init_db
from db.repositories import AssetRepository, PriceRepository, PositionRepository


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ManualValuationRunner:
    """Orchestrates manual valuation fetching and computation."""
    
    def __init__(self, args):
        self.args = args
        self.db = get_db()
    
    def run(self):
        """Execute the requested valuation operations."""
        mode = self.args.mode.lower()
        
        if mode in ["multiples", "both"]:
            logger.info("📊 Starting valuation metrics fetch...")
            self.fetch_valuation_multiples()
        
        if mode in ["market", "both"]:
            logger.info("💰 Starting market value computation...")
            self.compute_market_value()
        
        logger.info("✅ Manual valuation complete!")
    
    def fetch_valuation_multiples(self):
        """
        Fetch Forward P/E, PEG, EV/EBITDA, and growth metrics.
        
        Stores data in the valuation_metrics table.
        """
        with self.db.session() as session:
            asset_repo = AssetRepository(session)
            
            # Determine which assets to fetch
            if self.args.symbols:
                tickers = [s.strip().upper() for s in self.args.symbols.split(',')]
                assets = []
                for ticker in tickers:
                    asset = asset_repo.get_by_ticker(ticker)
                    if asset:
                        assets.append(asset)
                    else:
                        logger.warning(f"⚠️  Ticker {ticker} not found in database")
            else:
                assets = asset_repo.get_all()
            
            if not assets:
                logger.error("❌ No assets to process")
                return
            
            logger.info(f"Fetching valuation metrics for {len(assets)} assets...")
            
            # Fetch valuation metrics
            fetcher = ValuationFetcher()
            success_count = 0
            error_count = 0
            
            for asset in assets:
                try:
                    logger.info(f"  📥 {asset.ticker}...")
                    metrics = fetcher.fetch_for_ticker(asset.ticker)
                    
                    # Store in database
                    from db.repositories import ValuationRepository
                    valuation_repo = ValuationRepository(session)
                    valuation_repo.upsert(
                        asset_id=asset.id,
                        pe_forward=metrics.get("pe_forward"),
                        peg=metrics.get("peg"),
                        ev_ebitda=metrics.get("ev_ebitda"),
                        revenue_growth=metrics.get("revenue_growth"),
                        eps_growth=metrics.get("eps_growth"),
                    )
                    
                    # Display fetched metrics
                    metrics_str = []
                    if metrics.get("pe_forward") is not None:
                        metrics_str.append(f"P/E={metrics['pe_forward']:.2f}")
                    if metrics.get("peg") is not None:
                        metrics_str.append(f"PEG={metrics['peg']:.2f}")
                    if metrics.get("ev_ebitda") is not None:
                        metrics_str.append(f"EV/EBITDA={metrics['ev_ebitda']:.2f}")
                    
                    if metrics_str:
                        logger.info(f"     ✅ {', '.join(metrics_str)}")
                    else:
                        logger.warning(f"     ⚠️  No metrics available")
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"     ❌ Failed: {e}")
                    error_count += 1
                
                # Rate limiting
                import time
                time.sleep(config.data_fetcher.request_delay)
            
            session.commit()
            
            logger.info(f"\n📊 Valuation Metrics Summary:")
            logger.info(f"   Success: {success_count}")
            logger.info(f"   Errors:  {error_count}")
    
    def compute_market_value(self):
        """
        Compute portfolio market value using positions × prices.
        
        Optionally uses a specific as-of date or the latest available prices.
        """
        as_of_date = self.args.as_of
        price_overrides = self._parse_price_overrides()
        
        with self.db.session() as session:
            asset_repo = AssetRepository(session)
            position_repo = PositionRepository(session)
            price_repo = PriceRepository(session)
            
            # Get all positions
            positions = position_repo.get_all_active_positions()
            
            if not positions:
                logger.warning("⚠️  No positions found in portfolio")
                return
            
            logger.info(f"Computing market value for {len(positions)} positions...")
            if as_of_date:
                logger.info(f"Using as-of date: {as_of_date}")
            else:
                logger.info("Using latest available prices")
            
            total_market_value = 0.0
            success_count = 0
            missing_price_count = 0
            
            for position in positions:
                asset = position.asset
                if not asset:
                    logger.warning(f"⚠️  Asset ID {position.asset_id} not found")
                    continue
                
                # Use long_shares (net_shares would be long - short)
                shares = position.long_shares or 0.0
                if shares <= 0:
                    continue
                
                # Get price (with override support)
                price = None
                if asset.ticker in price_overrides:
                    price = price_overrides[asset.ticker]
                    logger.info(f"  {asset.ticker}: using override price ${price:.2f}")
                elif as_of_date:
                    price_record = price_repo.get_price_on_date(position.asset_id, as_of_date)
                    if price_record and price_record.close is not None:
                        price = price_record.close
                else:
                    price_record = price_repo.get_latest_price(position.asset_id)
                    if price_record and price_record.close is not None:
                        price = price_record.close
                
                if price is None:
                    logger.warning(f"  ⚠️  {asset.ticker}: No price available (shares: {shares:.4f})")
                    missing_price_count += 1
                    continue
                
                market_value = shares * price
                total_market_value += market_value
                
                logger.info(
                    f"  ✅ {asset.ticker}: {shares:.4f} shares × ${price:.2f} = ${market_value:,.2f}"
                )
                success_count += 1
            
            logger.info(f"\n💰 Market Value Summary:")
            logger.info(f"   Total Market Value: ${total_market_value:,.2f}")
            logger.info(f"   Positions Valued:   {success_count}/{len(positions)}")
            if missing_price_count > 0:
                logger.warning(f"   Missing Prices:     {missing_price_count}")
                logger.warning(f"\n   💡 Tip: Run 'python scripts/manual_valuation.py --mode prices' first")
                logger.warning(f"           or use --override-price to set manual prices")
    
    def _parse_price_overrides(self) -> dict[str, float]:
        """Parse --override-price argument into dict."""
        if not self.args.override_price:
            return {}
        
        overrides = {}
        try:
            pairs = self.args.override_price.split(',')
            for pair in pairs:
                ticker, price = pair.split('=')
                overrides[ticker.strip().upper()] = float(price.strip())
        except ValueError as e:
            logger.error(f"❌ Invalid --override-price format: {e}")
            logger.error("   Expected format: AAPL=192.10,MSFT=450.25")
            sys.exit(1)
        
        return overrides


def fetch_missing_prices(symbols: Optional[str] = None):
    """
    Helper to fetch missing price data.
    
    Args:
        symbols: Comma-separated list of tickers (optional)
    """
    logger.info("📈 Fetching missing price data...")
    
    with get_db().session() as session:
        asset_repo = AssetRepository(session)
        price_repo = PriceRepository(session)
        
        if symbols:
            tickers = [s.strip().upper() for s in symbols.split(',')]
            assets = []
            for ticker in tickers:
                asset = asset_repo.get_by_ticker(ticker)
                if asset:
                    assets.append(asset)
                else:
                    logger.warning(f"⚠️  Ticker {ticker} not found")
        else:
            assets = asset_repo.get_all()
        
        if not assets:
            logger.error("❌ No assets to fetch prices for")
            return
        
        fetcher = PriceFetcher()
        success_count = 0
        
        for asset in assets:
            # Check if we have recent prices
            latest_price = price_repo.get_latest_price(asset.id)
            
            if latest_price:
                logger.info(f"  {asset.ticker}: Latest price from {latest_price.date}")
            else:
                logger.info(f"  {asset.ticker}: No prices found, fetching initial data...")
            
            # Fetch
            from datetime import timedelta
            if latest_price:
                start_dt = datetime.strptime(latest_price.date, "%Y-%m-%d") + timedelta(days=1)
                # Don't fetch if start date is in the future
                if start_dt.date() > datetime.now().date():
                    logger.info(f"     ℹ️  Already up to date (latest: {latest_price.date})")
                    continue
                start_date = start_dt.strftime("%Y-%m-%d")
            else:
                start_dt = datetime.now() - timedelta(days=config.data_fetcher.default_lookback_days)
                start_date = start_dt.strftime("%Y-%m-%d")
            
            result = fetcher.fetch_for_asset(asset, start_date)
            
            if result.success and result.records:
                count = price_repo.bulk_upsert_prices(asset.id, result.records)
                logger.info(f"     ✅ Stored {count} new price records")
                success_count += 1
            elif result.success:
                logger.info(f"     ℹ️  No new data available")
            else:
                logger.error(f"     ❌ Failed: {result.message}")
            
            import time
            time.sleep(config.data_fetcher.request_delay)
        
        session.commit()
        logger.info(f"\n✅ Price fetch complete: {success_count} assets updated")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manual valuation fetch and computation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch valuation metrics for all assets
  %(prog)s --mode multiples

  # Fetch for specific tickers
  %(prog)s --mode multiples --symbols AAPL,MSFT

  # Compute current market value
  %(prog)s --mode market

  # Compute market value as of Jan 20, 2026
  %(prog)s --mode market --as-of 2026-01-20

  # Fetch both metrics and compute value
  %(prog)s --mode both

  # Fetch missing price data first
  %(prog)s --mode prices
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['multiples', 'market', 'both', 'prices'],
        default='both',
        help='What to fetch/compute (default: both)'
    )
    
    parser.add_argument(
        '--symbols',
        help='Comma-separated list of tickers (default: all assets)'
    )
    
    parser.add_argument(
        '--as-of',
        metavar='YYYY-MM-DD',
        help='As-of date for market value computation (default: latest)'
    )
    
    parser.add_argument(
        '--override-price',
        metavar='TICKER=PRICE,...',
        help='Manual price overrides (e.g., AAPL=192.10,MSFT=450.25)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refetch even if recently updated'
    )
    
    parser.add_argument(
        '--db-path',
        help='Override database path (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Override DB path if provided
    if args.db_path:
        import os
        os.environ['PORTFOLIO_DB_PATH'] = args.db_path
    
    # Initialize database
    init_db()
    
    # Execute
    if args.mode == 'prices':
        fetch_missing_prices(args.symbols)
    else:
        runner = ManualValuationRunner(args)
        runner.run()


if __name__ == "__main__":
    main()
