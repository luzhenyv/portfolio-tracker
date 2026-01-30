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
from db import Asset
from db.repositories import AssetRepository, PriceRepository, ValuationRepository


logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    ticker: str
    success: bool
    records_count: int = 0
    message: str = ""
    records: list[dict] = None  # Price records fetched (not yet saved)


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
            
            logger.info(f"✅ Fetched {len(records)} records for {asset.ticker}")
            return FetchResult(
                ticker=asset.ticker,
                success=True,
                records_count=len(records),
                records=records,
                message=f"Fetched {len(records)} records",
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
        Fetch prices for all assets in the database.
        
        Returns:
            List of FetchResult for each asset.
        """
        from db import get_db
        
        db = get_db()
        results = []
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            price_repo = PriceRepository(session)
            
            assets = asset_repo.get_all()
            
            for asset in assets:
                # Determine start date
                latest_price = price_repo.get_latest_price(asset.id)
                
                if latest_price:
                    # Fetch from day after last price
                    start_dt = datetime.strptime(latest_price.date, "%Y-%m-%d") + timedelta(days=1)
                    start_date = start_dt.strftime("%Y-%m-%d")
                else:
                    # Initial fetch - get last 90 days
                    start_dt = datetime.now() - timedelta(days=config.data_fetcher.default_lookback_days)
                    start_date = start_dt.strftime("%Y-%m-%d")
                
                # Fetch prices
                fetch_result = self.fetch_for_asset(asset, start_date)
                
                # Save records if successful
                if fetch_result.success and fetch_result.records:
                    count = price_repo.bulk_upsert_prices(asset.id, fetch_result.records)
                    fetch_result.records_count = count
                
                results.append(fetch_result)
                
                # Rate limiting
                time.sleep(self.config.request_delay)
        
        return results


class ValuationFetcher:
    """
    Fetches valuation metrics from Yahoo Finance.
    
    Yahoo-aligned fields matching "Statistics" page:
    - Valuation Measures: Market Cap, EV, P/E, PEG, P/S, P/B, EV/Rev, EV/EBITDA
    - Financial Highlights: Margins, ROA, ROE, Revenue, Net Income, EPS, Cash, Debt, FCF
    
    FR-8: Auto-fetch via yfinance Ticker.info
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
                # Valuation Measures
                "market_cap": self._safe_metric(info.get("marketCap")),
                "enterprise_value": self._safe_metric(info.get("enterpriseValue")),
                "pe_trailing": self._safe_metric(info.get("trailingPE")),
                "pe_forward": self._safe_metric(info.get("forwardPE")),
                "peg": self._safe_metric(info.get("pegRatio")),
                "price_to_sales": self._safe_metric(info.get("priceToSalesTrailing12Months")),
                "price_to_book": self._safe_metric(info.get("priceToBook")),
                "ev_to_revenue": self._safe_metric(info.get("enterpriseToRevenue")),
                "ev_ebitda": self._safe_metric(info.get("enterpriseToEbitda")),
                # Financial Highlights - Profitability
                "profit_margin": self._safe_metric(info.get("profitMargins")),
                "return_on_assets": self._safe_metric(info.get("returnOnAssets")),
                "return_on_equity": self._safe_metric(info.get("returnOnEquity")),
                # Financial Highlights - Income Statement
                "revenue_ttm": self._safe_metric(info.get("totalRevenue")),
                "net_income_ttm": self._safe_metric(info.get("netIncomeToCommon")),
                "diluted_eps_ttm": self._safe_metric(info.get("trailingEps")),
                # Financial Highlights - Balance Sheet & Cash Flow
                "total_cash": self._safe_metric(info.get("totalCash")),
                "total_debt_to_equity": self._safe_metric(info.get("debtToEquity")),
                "levered_free_cash_flow": self._safe_metric(info.get("freeCashflow")),
                # Legacy fields
                "revenue_growth": self._safe_metric(info.get("revenueGrowth")),
                "eps_growth": self._safe_metric(info.get("earningsGrowth")),
            }
        except Exception as e:
            logger.error(f"Failed to fetch valuation for {ticker}: {e}")
            return {
                "market_cap": None,
                "enterprise_value": None,
                "pe_trailing": None,
                "pe_forward": None,
                "peg": None,
                "price_to_sales": None,
                "price_to_book": None,
                "ev_to_revenue": None,
                "ev_ebitda": None,
                "profit_margin": None,
                "return_on_assets": None,
                "return_on_equity": None,
                "revenue_ttm": None,
                "net_income_ttm": None,
                "diluted_eps_ttm": None,
                "total_cash": None,
                "total_debt_to_equity": None,
                "levered_free_cash_flow": None,
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
    
    def fetch_all_assets(self) -> list[FetchResult]:
        """
        Fetch valuation metrics for all assets in the database.
        
        Returns:
            List of FetchResult for each asset.
        """
        from db import get_db
        
        db = get_db()
        results = []
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            valuation_repo = ValuationRepository(session)
            
            assets = asset_repo.get_all()
            
            for asset in assets:
                try:
                    # Fetch valuation data
                    valuation_data = self.fetch_for_ticker(asset.ticker)
                    
                    # Save to database with all Yahoo-aligned fields
                    valuation_repo.upsert(
                        asset_id=asset.id,
                        # Valuation Measures
                        market_cap=valuation_data.get("market_cap"),
                        enterprise_value=valuation_data.get("enterprise_value"),
                        pe_trailing=valuation_data.get("pe_trailing"),
                        pe_forward=valuation_data.get("pe_forward"),
                        peg=valuation_data.get("peg"),
                        price_to_sales=valuation_data.get("price_to_sales"),
                        price_to_book=valuation_data.get("price_to_book"),
                        ev_to_revenue=valuation_data.get("ev_to_revenue"),
                        ev_ebitda=valuation_data.get("ev_ebitda"),
                        # Financial Highlights - Profitability
                        profit_margin=valuation_data.get("profit_margin"),
                        return_on_assets=valuation_data.get("return_on_assets"),
                        return_on_equity=valuation_data.get("return_on_equity"),
                        # Financial Highlights - Income Statement
                        revenue_ttm=valuation_data.get("revenue_ttm"),
                        net_income_ttm=valuation_data.get("net_income_ttm"),
                        diluted_eps_ttm=valuation_data.get("diluted_eps_ttm"),
                        # Financial Highlights - Balance Sheet & Cash Flow
                        total_cash=valuation_data.get("total_cash"),
                        total_debt_to_equity=valuation_data.get("total_debt_to_equity"),
                        levered_free_cash_flow=valuation_data.get("levered_free_cash_flow"),
                        # Legacy fields
                        revenue_growth=valuation_data.get("revenue_growth"),
                        eps_growth=valuation_data.get("eps_growth"),
                    )
                    
                    results.append(FetchResult(
                        ticker=asset.ticker,
                        success=True,
                        records_count=1,
                        message="Valuation metrics updated",
                    ))
                except Exception as e:
                    logger.error(f"Failed to fetch valuation for {asset.ticker}: {e}")
                    results.append(FetchResult(
                        ticker=asset.ticker,
                        success=False,
                        message=str(e),
                    ))
                
                # Rate limiting
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
