"""
Data fetching services for market data and valuations.

FR-1: Fetch daily EOD prices from Yahoo Finance
FR-2: Missing data must not crash; store raw values only
FR-8: Auto-fetch valuation metrics via Yahoo Finance
FR-9: Missing fields stored as NULL
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from config import config
from db import Asset, get_db
from db.repositories import AssetRepository, PriceRepository, ValuationRepository


logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    ticker: str
    success: bool
    records_count: int = 0
    message: str = ""


class PriceFetcher:
    """
    Fetches EOD price data from Yahoo Finance.
    
    FR-1: Idempotent execution (safe to run daily)
    FR-2: Store raw fetched values only, no interpolation
    """
    
    def __init__(self):
        self.config = config.data_fetcher
    
    def fetch_for_asset(
        self,
        asset: Asset,
        start_date: str | None = None,
    ) -> FetchResult:
        """
        Fetch prices for a single asset.
        
        Args:
            asset: Asset to fetch prices for.
            start_date: Optional start date (YYYY-MM-DD). 
                       If None, fetches from day after last stored price.
                       
        Returns:
            FetchResult with operation status.
        """
        logger.info(f"📥 Fetching prices for {asset.ticker}")
        
        try:
            # Fetch data from Yahoo Finance
            df = yf.download(
                asset.ticker,
                start=start_date,
                progress=False,
                auto_adjust=False,
                timeout=self.config.yfinance_timeout,
            )
            
            if df.empty:
                logger.warning(f"⚠️ No data returned for {asset.ticker}")
                return FetchResult(
                    ticker=asset.ticker,
                    success=True,
                    records_count=0,
                    message="No data available",
                )
            
            # Convert to records
            records = self._dataframe_to_records(df)
            
            # Store in database
            db = get_db()
            with db.session() as session:
                price_repo = PriceRepository(session)
                count = price_repo.bulk_upsert_prices(asset.id, records)
            
            logger.info(f"✅ Stored {count} new records for {asset.ticker}")
            return FetchResult(
                ticker=asset.ticker,
                success=True,
                records_count=count,
                message=f"Stored {count} new records",
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch {asset.ticker}: {e}")
            return FetchResult(
                ticker=asset.ticker,
                success=False,
                message=str(e),
            )
    
    def _dataframe_to_records(self, df: pd.DataFrame) -> list[dict]:
        """Convert yfinance DataFrame to list of dicts."""
        df = df.reset_index()
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        records = []
        for _, row in df.iterrows():
            # Handle date format
            date_val = row.get("Date") or row.get("index")
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)[:10]
            
            records.append({
                "date": date_str,
                "open": self._safe_float(row.get("Open")),
                "high": self._safe_float(row.get("High")),
                "low": self._safe_float(row.get("Low")),
                "close": self._safe_float(row.get("Close")),
                "adjusted_close": self._safe_float(row.get("Adj Close")),
                "volume": self._safe_int(row.get("Volume")),
            })
        
        return records
    
    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Safely convert to float, returning None for invalid values."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_int(value: Any) -> int | None:
        """Safely convert to int, returning None for invalid values."""
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def fetch_all_assets(self) -> list[FetchResult]:
        """
        Fetch prices for all assets.
        
        Implements incremental fetching from last known date.
        
        Returns:
            List of FetchResult for each asset.
        """
        db = get_db()
        results = []
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            assets = list(asset_repo.get_all())
        
        # Fetch outside session to avoid long transactions
        for asset in assets:
            # Get incremental start date
            with db.session() as session:
                price_repo = PriceRepository(session)
                latest_date = price_repo.get_latest_date(asset.id)
            
            if latest_date:
                start_dt = datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)
                start_date = start_dt.strftime("%Y-%m-%d")
            else:
                start_dt = datetime.now() - timedelta(days=self.config.default_lookback_days)
                start_date = start_dt.strftime("%Y-%m-%d")
            
            result = self.fetch_for_asset(asset, start_date)
            results.append(result)
            
            # Rate limiting
            time.sleep(self.config.request_delay)
        
        return results


class ValuationFetcher:
    """
    Fetches valuation metrics from Yahoo Finance.
    
    FR-8: Auto-fetch Forward P/E, PEG, EV/EBITDA, growth metrics
    FR-9: Missing fields stored as NULL (no synthetic values)
    """
    
    def fetch_for_ticker(self, ticker: str) -> dict[str, float | None]:
        """
        Fetch valuation metrics for a single ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Dict of valuation metrics (with NULL for missing data).
        """
        try:
            info = yf.Ticker(ticker).info
            
            return {
                "pe_forward": self._safe_metric(info.get("forwardPE")),
                "peg": self._safe_metric(info.get("pegRatio")),
                "ev_ebitda": self._safe_metric(info.get("enterpriseToEbitda")),
                "revenue_growth": self._safe_metric(info.get("revenueGrowth")),
                "eps_growth": self._safe_metric(info.get("earningsGrowth")),
            }
        except Exception as e:
            logger.error(f"Failed to fetch valuation for {ticker}: {e}")
            return {
                "pe_forward": None,
                "peg": None,
                "ev_ebitda": None,
                "revenue_growth": None,
                "eps_growth": None,
            }
    
    @staticmethod
    def _safe_metric(value: Any) -> float | None:
        """
        Safely extract metric, returning None for missing/invalid data.
        
        FR-9: No synthetic values allowed.
        """
        if value is None:
            return None
        try:
            val = float(value)
            # Filter out clearly invalid values
            if pd.isna(val) or val == float("inf") or val == float("-inf"):
                return None
            return val
        except (ValueError, TypeError):
            return None
    
    def fetch_and_store_for_asset(self, asset: Asset) -> FetchResult:
        """
        Fetch and store valuation metrics for an asset.
        
        Args:
            asset: Asset to fetch valuation for.
            
        Returns:
            FetchResult with operation status.
        """
        logger.info(f"📊 Fetching valuation for {asset.ticker}")
        
        metrics = self.fetch_for_ticker(asset.ticker)
        
        # Store in database
        db = get_db()
        with db.session() as session:
            valuation_repo = ValuationRepository(session)
            valuation_repo.upsert(
                asset_id=asset.id,
                pe_forward=metrics["pe_forward"],
                peg=metrics["peg"],
                ev_ebitda=metrics["ev_ebitda"],
                revenue_growth=metrics["revenue_growth"],
                eps_growth=metrics["eps_growth"],
            )
        
        logger.info(f"✅ Updated valuation for {asset.ticker}")
        return FetchResult(
            ticker=asset.ticker,
            success=True,
            records_count=1,
            message="Valuation updated",
        )
    
    def fetch_all_assets(self) -> list[FetchResult]:
        """
        Fetch valuation metrics for all assets.
        
        Returns:
            List of FetchResult for each asset.
        """
        db = get_db()
        results = []
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            assets = list(asset_repo.get_all())
        
        for asset in assets:
            result = self.fetch_and_store_for_asset(asset)
            results.append(result)
            time.sleep(config.data_fetcher.request_delay)
        
        return results


def fetch_prices_main():
    """Entry point for price fetching."""
    from db import init_db
    
    logging.basicConfig(level=logging.INFO)
    init_db()
    
    fetcher = PriceFetcher()
    results = fetcher.fetch_all_assets()
    
    success_count = sum(1 for r in results if r.success)
    total_records = sum(r.records_count for r in results)
    
    print(f"\n📈 Price Fetch Complete")
    print(f"   Assets: {success_count}/{len(results)} successful")
    print(f"   Records: {total_records} new prices stored")


def fetch_valuations_main():
    """Entry point for valuation fetching."""
    from db import init_db
    
    logging.basicConfig(level=logging.INFO)
    init_db()
    
    fetcher = ValuationFetcher()
    results = fetcher.fetch_all_assets()
    
    success_count = sum(1 for r in results if r.success)
    
    print(f"\n📊 Valuation Fetch Complete")
    print(f"   Assets: {success_count}/{len(results)} successful")


if __name__ == "__main__":
    fetch_prices_main()
