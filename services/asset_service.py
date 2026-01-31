"""
Asset Service - Orchestrates asset creation with yfinance auto-fetch.

This service layer provides a reusable interface for creating assets with
automatic metadata, price, and valuation fetching from Yahoo Finance.
Designed to be consumed by CLI, Streamlit, FastAPI, etc.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import yfinance as yf

from config import config
from db import Asset, AssetStatus, AssetType, get_db
from db.repositories import AssetRepository, PriceRepository, ValuationRepository
from data.yfinance_fetcher import PriceFetcher, ValuationFetcher


logger = logging.getLogger(__name__)


@dataclass
class AssetCreationResult:
    """
    Result object for asset creation operations.
    
    Captures both successful and partial outcomes to provide clear
    feedback to users about what succeeded and what failed.
    """
    asset: Asset | None
    created: bool
    prices_fetched: int
    valuation_fetched: bool
    errors: list[str] = field(default_factory=list)
    status_message: str = ""
    
    @property
    def success(self) -> bool:
        """Whether the operation was successful (asset created/found)."""
        return self.asset is not None
    
    @property
    def full_success(self) -> bool:
        """Whether the operation was fully successful (all data fetched)."""
        return self.success and self.prices_fetched > 0 and self.valuation_fetched


def create_asset_with_data(
    ticker: str,
    status: AssetStatus = AssetStatus.OWNED,
    asset_type: AssetType = AssetType.STOCK,
    lookback_days: int | None = None,
) -> AssetCreationResult:
    """
    Create an asset with automatic yfinance data fetching.
    
    This is the main entry point for asset creation across all interfaces
    (CLI, Streamlit, FastAPI). It orchestrates:
    
    1. Asset creation/retrieval
    2. Metadata enrichment from yfinance (name, sector, industry, exchange)
    3. Historical price data fetching
    4. Valuation metrics fetching
    
    The function handles partial failures gracefully, ensuring that if
    prices are fetched but valuations fail, the user still gets useful data.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        status: Asset status (OWNED or WATCHLIST)
        asset_type: Type of asset (STOCK, ETF, CRYPTO, etc.)
        lookback_days: Number of days to fetch price history (default from config)
        
    Returns:
        AssetCreationResult with detailed outcome information
        
    Example:
        >>> result = create_asset_with_data("AAPL", AssetStatus.OWNED, AssetType.STOCK)
        >>> print(result.status_message)
        "✅ Asset AAPL created with 365 prices and valuation metrics"
    """
    ticker = ticker.upper()
    errors = []
    
    if lookback_days is None:
        lookback_days = config.data_fetcher.default_lookback_days
    
    db = get_db()
    
    with db.session() as session:
        asset_repo = AssetRepository(session)
        price_repo = PriceRepository(session)
        valuation_repo = ValuationRepository(session)
        
        # Step 1: Get or create asset
        asset, created = asset_repo.get_or_create(ticker=ticker, status=status, asset_type=asset_type)
        
        if not created and asset.status != status:
            asset_repo.update_status(asset.id, status)
            logger.info(f"Updated {ticker} status to {status.value}")
        
        # Step 2: Fetch metadata from yfinance
        metadata = _fetch_ticker_metadata(ticker)
        if metadata:
            if not asset.name:
                asset.name = metadata.get("name")
            if not asset.sector:
                asset.sector = metadata.get("sector")
            if not asset.industry:
                asset.industry = metadata.get("industry")
            if not asset.exchange:
                asset.exchange = metadata.get("exchange")
            session.flush()
        else:
            errors.append("Could not fetch metadata from yfinance")
        
        # Step 3: Fetch historical prices
        prices_fetched = 0
        latest_date = price_repo.get_latest_date(asset.id)
        
        if latest_date:
            # Incremental fetch from last known date
            start_dt = datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)
            start_date = start_dt.strftime("%Y-%m-%d")
        else:
            # Initial fetch
            start_dt = datetime.now() - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        try:
            fetcher = PriceFetcher()
            fetch_result = fetcher.fetch_for_asset(asset, start_date)
            
            if not fetch_result.success:
                errors.append(f"Price fetch failed: {fetch_result.message}")
            elif fetch_result.records:
                # Save fetched records using existing session
                count = price_repo.bulk_upsert_prices(asset.id, fetch_result.records)
                prices_fetched = count
                logger.info(f"✅ Stored {count} price records for {ticker}")
            else:
                prices_fetched = 0
        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
            errors.append(f"Price fetch error: {str(e)}")
        
        # Small delay for rate limiting
        time.sleep(config.data_fetcher.request_delay)
        
        # Step 4: Fetch valuation metrics
        valuation_fetched = False
        try:
            val_fetcher = ValuationFetcher()
            valuation_data = val_fetcher.fetch_for_ticker(ticker)
            
            valuation_repo.upsert(
                asset_id=asset.id,
                pe_forward=valuation_data.get("pe_forward"),
                peg=valuation_data.get("peg"),
                ev_ebitda=valuation_data.get("ev_ebitda"),
                revenue_growth=valuation_data.get("revenue_growth"),
                eps_growth=valuation_data.get("eps_growth"),
            )
            valuation_fetched = True
        except Exception as e:
            logger.error(f"Failed to fetch valuation for {ticker}: {e}")
            errors.append(f"Valuation fetch error: {str(e)}")
        
        # Build status message
        status_message = _build_status_message(
            ticker=ticker,
            created=created,
            prices_fetched=prices_fetched,
            valuation_fetched=valuation_fetched,
            errors=errors,
        )
        
        return AssetCreationResult(
            asset=asset,
            created=created,
            prices_fetched=prices_fetched,
            valuation_fetched=valuation_fetched,
            errors=errors,
            status_message=status_message,
        )


def _fetch_ticker_metadata(ticker: str) -> dict[str, str | None] | None:
    """
    Fetch basic metadata for a ticker from yfinance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with name, sector, industry, exchange or None if fetch fails
    """
    try:
        info = yf.Ticker(ticker).info
        
        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "exchange": info.get("exchange"),
        }
    except Exception as e:
        logger.warning(f"Could not fetch metadata for {ticker}: {e}")
        return None


def _build_status_message(
    ticker: str,
    created: bool,
    prices_fetched: int,
    valuation_fetched: bool,
    errors: list[str],
) -> str:
    """
    Build a user-friendly status message based on operation outcome.
    
    Args:
        ticker: Stock ticker
        created: Whether asset was newly created
        prices_fetched: Number of price records fetched
        valuation_fetched: Whether valuation was fetched
        errors: List of error messages
        
    Returns:
        Formatted status message
    """
    if not created and prices_fetched == 0 and not valuation_fetched:
        return f"ℹ️ Asset {ticker} already exists (no new data available)"
    
    if created:
        action = "created"
    else:
        action = "updated"
    
    parts = [f"✅ Asset {ticker} {action}"]
    
    if prices_fetched > 0:
        parts.append(f"{prices_fetched} prices")
    
    if valuation_fetched:
        parts.append("valuation metrics")
    
    result = " with ".join([parts[0], ", ".join(parts[1:])])
    
    if errors:
        result += f"\n⚠️ Warnings: {'; '.join(errors)}"
    
    return result


def print_asset_creation_result(result: AssetCreationResult) -> None:
    """
    Print a formatted asset creation result to console.
    
    Helper function for CLI usage to display results in a user-friendly way.
    
    Args:
        result: AssetCreationResult to display
    """
    print(result.status_message)
    
    if result.asset:
        if result.asset.name:
            print(f"   Name: {result.asset.name}")
        if result.asset.sector:
            print(f"   Sector: {result.asset.sector}")
        if result.asset.industry:
            print(f"   Industry: {result.asset.industry}")
        if result.asset.exchange:
            print(f"   Exchange: {result.asset.exchange}")


def get_all_tickers() -> list[str]:
    """Get sorted list of all existing tickers."""
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        return sorted([a.ticker for a in asset_repo.get_all()])


def get_watchlist_tickers() -> list[str]:
    """Get sorted list of watchlist tickers only."""
    db = get_db()
    with db.session() as session:
        asset_repo = AssetRepository(session)
        return sorted([a.ticker for a in asset_repo.get_watchlist_assets()])


@dataclass
class DeleteResult:
    """Result of a batch delete operation."""
    deleted: list[str]
    blocked: list[dict]  # {"ticker": str, "reason": str}
    not_found: list[str]
    errors: list[dict]  # {"ticker": str, "error": str}
    
    @property
    def success_count(self) -> int:
        return len(self.deleted)
    
    @property
    def blocked_count(self) -> int:
        return len(self.blocked)
    
    @property
    def total_requested(self) -> int:
        return len(self.deleted) + len(self.blocked) + len(self.not_found) + len(self.errors)


def delete_assets(
    tickers: list[str],
    allow_owned: bool = False,
    allow_with_trades: bool = False,
    allow_with_active_position: bool = False,
) -> DeleteResult:
    """
    Delete one or more assets and all their related data with configurable safety checks.
    
    By default (all 'allow_*' parameters False), this function safely deletes WATCHLIST 
    assets, blocking deletion of:
    - OWNED assets (unless allow_owned=True)
    - Assets with active positions (unless allow_with_active_position=True)
    - Assets with trade history (unless allow_with_trades=True)
    
    The deletion cascades to remove all related data:
    - prices_daily
    - fundamentals_quarterly
    - valuation_metrics
    - valuation_metric_overrides
    - watchlist_targets
    - investment_thesis
    - trades (cascades if allowed)
    - positions (cascades if allowed)
    
    Args:
        tickers: List of ticker symbols to delete
        allow_owned: If True, allows deleting assets with AssetStatus.OWNED 
            (useful for closed positions that were never moved back to watchlist)
        allow_with_trades: If True, allows deleting assets even if they have historical trade records
        allow_with_active_position: If True, allows deleting assets even if they have current holdings
        
    Returns:
        DeleteResult with deleted, blocked, not_found, and errors lists
    """
    deleted = []
    blocked = []
    not_found = []
    errors = []
    
    if not tickers:
        return DeleteResult(deleted=deleted, blocked=blocked, not_found=not_found, errors=errors)
    
    db = get_db()
    
    with db.session() as session:
        asset_repo = AssetRepository(session)
        
        # Fetch all requested assets in one query
        assets = asset_repo.get_by_tickers(tickers)
        found_tickers = {a.ticker: a for a in assets}
        
        for ticker in tickers:
            ticker_upper = ticker.upper()
            
            if ticker_upper not in found_tickers:
                not_found.append(ticker_upper)
                continue
            
            asset = found_tickers[ticker_upper]
            
            # Block deletion of OWNED assets unless explicitly allowed
            if asset.status == AssetStatus.OWNED and not allow_owned:
                blocked.append({
                    "ticker": ticker_upper,
                    "reason": "Asset is OWNED (set allow_owned=True to delete)"
                })
                continue
            
            # Block if asset has active position (non-zero shares) unless explicitly allowed
            if asset_repo.has_position(asset.id) and not allow_with_active_position:
                blocked.append({
                    "ticker": ticker_upper,
                    "reason": "Asset has active position (set allow_with_active_position=True to delete)"
                })
                continue
            
            # Block if asset has trade history (extra safety) unless explicitly allowed
            if asset_repo.has_trades(asset.id) and not allow_with_trades:
                blocked.append({
                    "ticker": ticker_upper,
                    "reason": "Asset has trade history (set allow_with_trades=True to delete)"
                })
                continue
            
            # Safe to delete
            try:
                if asset_repo.delete_by_id(asset.id):
                    deleted.append(ticker_upper)
                    logger.info(f"Deleted asset: {ticker_upper} (allow_owned={allow_owned}, trades={allow_with_trades})")
                else:
                    errors.append({
                        "ticker": ticker_upper,
                        "error": "Delete operation returned False"
                    })
            except Exception as e:
                logger.error(f"Failed to delete {ticker_upper}: {e}")
                errors.append({
                    "ticker": ticker_upper,
                    "error": str(e)
                })
        
        # Commit all successful deletes
        session.commit()
    
    return DeleteResult(deleted=deleted, blocked=blocked, not_found=not_found, errors=errors)