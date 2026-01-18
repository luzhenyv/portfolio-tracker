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
from db import Asset, AssetStatus, Position, get_db
from db.repositories import AssetRepository, PositionRepository, PriceRepository, ValuationRepository
from data.fetch_prices import PriceFetcher, ValuationFetcher


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


@dataclass
class PositionCreationResult:
    """
    Result object for position creation operations.
    
    Captures the outcome of position creation including any automatic
    status changes from WATCHLIST to OWNED.
    """
    position: Position | None
    asset: Asset | None
    created: bool
    status_changed: bool  # WATCHLIST -> OWNED
    errors: list[str] = field(default_factory=list)
    status_message: str = ""
    
    @property
    def success(self) -> bool:
        """Whether the operation was successful (position created)."""
        return self.position is not None


def create_asset_with_data(
    ticker: str,
    status: AssetStatus = AssetStatus.OWNED,
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
        lookback_days: Number of days to fetch price history (default from config)
        
    Returns:
        AssetCreationResult with detailed outcome information
        
    Example:
        >>> result = create_asset_with_data("AAPL", AssetStatus.OWNED)
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
        asset, created = asset_repo.get_or_create(ticker=ticker, status=status)
        
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


def create_position(
    ticker: str,
    shares: float,
    buy_price: float,
    buy_date: str | None = None,
) -> PositionCreationResult:
    """
    Create a position for an asset with automatic status management.
    
    This function orchestrates position creation and automatically updates
    the asset status from WATCHLIST to OWNED when a position is added.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        shares: Number of shares purchased
        buy_price: Price per share at purchase
        buy_date: Buy date in YYYY-MM-DD format (defaults to today)
        
    Returns:
        PositionCreationResult with detailed outcome information
        
    Example:
        >>> result = create_position("NVDA", 10, 180.00, "2026-01-18")
        >>> print(result.status_message)
        "✅ Added position for NVDA"
    """
    ticker = ticker.upper()
    errors = []
    status_changed = False
    
    # Default to today's date if not specified
    if buy_date is None:
        buy_date = datetime.now().strftime("%Y-%m-%d")
    
    db = get_db()
    
    with db.session() as session:
        asset_repo = AssetRepository(session)
        position_repo = PositionRepository(session)
        
        # Step 1: Validate asset exists
        asset = asset_repo.get_by_ticker(ticker)
        
        if not asset:
            error_msg = f"Asset not found: {ticker}"
            return PositionCreationResult(
                position=None,
                asset=None,
                created=False,
                status_changed=False,
                errors=[error_msg],
                status_message=f"❌ {error_msg}\n   Use 'add-asset' to add it first.",
            )
        
        # Step 2: Update asset status to OWNED if currently WATCHLIST
        if asset.status == AssetStatus.WATCHLIST:
            asset_repo.update_status(asset.id, AssetStatus.OWNED)
            asset.status = AssetStatus.OWNED  # Update in-memory object
            status_changed = True
            logger.info(f"Updated {ticker} status from WATCHLIST to OWNED")
        
        # Step 3: Create position
        try:
            position = position_repo.create(
                asset_id=asset.id,
                buy_date=buy_date,
                shares=shares,
                buy_price=buy_price,
            )
            
            # Build status message
            cost = shares * buy_price
            status_message = f"✅ Added position for {ticker}"
            
            if status_changed:
                status_message += f"\n   📍 Status changed: WATCHLIST → OWNED"
            
            return PositionCreationResult(
                position=position,
                asset=asset,
                created=True,
                status_changed=status_changed,
                errors=errors,
                status_message=status_message,
            )
            
        except Exception as e:
            logger.error(f"Failed to create position for {ticker}: {e}")
            error_msg = f"Failed to create position: {str(e)}"
            return PositionCreationResult(
                position=None,
                asset=asset,
                created=False,
                status_changed=status_changed,
                errors=[error_msg],
                status_message=f"❌ {error_msg}",
            )


def print_position_creation_result(result: PositionCreationResult) -> None:
    """
    Print a formatted position creation result to console.
    
    Helper function for CLI usage to display results in a user-friendly way.
    Shows position details including shares, price, date, and cost basis.
    
    Args:
        result: PositionCreationResult to display
    """
    print(result.status_message)
    
    if result.position:
        print(f"   Shares: {result.position.shares}")
        print(f"   Price: ${result.position.buy_price:.2f}")
        print(f"   Date: {result.position.buy_date}")
        cost = result.position.shares * result.position.buy_price
        print(f"   Cost: ${cost:,.2f}")
