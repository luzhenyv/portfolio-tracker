"""
Market Index Service.

Provides a clean API for managing market benchmark indices and their price data.
Supports portfolio comparison, correlation analysis, and benchmark tracking.

Usage:
    from services.market_index_service import (
        get_all_indices,
        get_index_by_symbol,
        get_index_prices,
        get_latest_index_prices,
        sync_index_prices,
    )
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Sequence

from config import config
from db import get_db, MarketIndex, IndexPriceDaily, IndexCategory
from db.repositories import MarketIndexRepository, IndexPriceRepository


@dataclass
class IndexSummary:
    """Summary of a market index with latest price."""
    id: int
    symbol: str
    name: str
    description: str | None
    category: str
    latest_close: float | None
    latest_date: str | None
    change_pct: float | None  # 1-day change percentage


@dataclass
class IndexPricePoint:
    """Single price point for an index."""
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None


@dataclass
class IndexPriceHistory:
    """Price history for an index."""
    symbol: str
    name: str
    prices: list[IndexPricePoint]
    start_date: str | None
    end_date: str | None


@dataclass
class SyncResult:
    """Result of syncing index prices."""
    symbol: str
    records_added: int
    latest_date: str | None
    success: bool
    error: str | None = None


# ============================================================================
# Query Functions
# ============================================================================

def get_all_indices() -> list[IndexSummary]:
    """
    Get all tracked market indices with their latest prices.
    
    Returns:
        List of IndexSummary with latest price data.
    """
    db = get_db()
    results = []
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        price_repo = IndexPriceRepository(session)
        
        indices = index_repo.get_all()
        
        # Get latest prices for all indices at once
        index_ids = [idx.id for idx in indices]
        latest_prices = price_repo.get_latest_prices_for_indices(index_ids) if index_ids else {}
        
        for idx in indices:
            latest_price = latest_prices.get(idx.id)
            
            # Calculate 1-day change if we have enough data
            change_pct = None
            if latest_price and latest_price.close:
                # Get previous day's price
                prices = price_repo.get_price_history(idx.id)
                if len(prices) >= 2:
                    prev_close = prices[-2].close if prices[-2].close else None
                    if prev_close and prev_close > 0:
                        change_pct = (latest_price.close - prev_close) / prev_close
            
            results.append(IndexSummary(
                id=idx.id,
                symbol=idx.symbol,
                name=idx.name,
                description=idx.description,
                category=idx.category.value if isinstance(idx.category, IndexCategory) else idx.category,
                latest_close=latest_price.close if latest_price else None,
                latest_date=latest_price.date if latest_price else None,
                change_pct=change_pct,
            ))
    
    return results


def get_index_by_symbol(symbol: str) -> MarketIndex | None:
    """
    Get a market index by its symbol.
    
    Args:
        symbol: The canonical symbol (e.g., "SPX", "VIX").
        
    Returns:
        MarketIndex model or None if not found.
    """
    db = get_db()
    with db.session() as session:
        repo = MarketIndexRepository(session)
        return repo.get_by_symbol(symbol)


def get_index_by_id(index_id: int) -> MarketIndex | None:
    """
    Get a market index by its ID.
    
    Args:
        index_id: The database ID.
        
    Returns:
        MarketIndex model or None if not found.
    """
    db = get_db()
    with db.session() as session:
        repo = MarketIndexRepository(session)
        return repo.get_by_id(index_id)


def get_indices_by_category(category: str) -> Sequence[MarketIndex]:
    """
    Get all indices in a category.
    
    Args:
        category: Category name (EQUITY, VOLATILITY, COMMODITY, BOND, CURRENCY).
        
    Returns:
        List of MarketIndex models.
    """
    db = get_db()
    with db.session() as session:
        repo = MarketIndexRepository(session)
        return repo.get_by_category(category)


def get_equity_indices() -> Sequence[MarketIndex]:
    """Get all equity indices (S&P 500, Russell 2000, etc.)."""
    return get_indices_by_category("EQUITY")


def get_index_prices(
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    lookback_days: int | None = None,
) -> IndexPriceHistory | None:
    """
    Get price history for an index.
    
    Args:
        symbol: Index symbol (e.g., "SPX").
        start_date: Start date (YYYY-MM-DD or date object).
        end_date: End date (YYYY-MM-DD or date object).
        lookback_days: Alternative to start_date - number of days to look back.
        
    Returns:
        IndexPriceHistory with price data, or None if index not found.
    """
    db = get_db()
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        price_repo = IndexPriceRepository(session)
        
        idx = index_repo.get_by_symbol(symbol)
        if not idx:
            return None
        
        # Convert dates to strings
        start_str = None
        end_str = None
        
        if lookback_days:
            end_str = date.today().isoformat()
            start_str = (date.today() - timedelta(days=lookback_days)).isoformat()
        else:
            if start_date:
                start_str = start_date.isoformat() if isinstance(start_date, date) else start_date
            if end_date:
                end_str = end_date.isoformat() if isinstance(end_date, date) else end_date
        
        prices = price_repo.get_price_history(idx.id, start_str, end_str)
        
        price_points = [
            IndexPricePoint(
                date=p.date,
                open=p.open,
                high=p.high,
                low=p.low,
                close=p.close,
                volume=p.volume,
            )
            for p in prices
        ]
        
        return IndexPriceHistory(
            symbol=idx.symbol,
            name=idx.name,
            prices=price_points,
            start_date=price_points[0].date if price_points else None,
            end_date=price_points[-1].date if price_points else None,
        )


def get_latest_index_prices() -> dict[str, tuple[float, str]]:
    """
    Get latest prices for all tracked indices.
    
    Returns:
        Dict mapping symbol -> (close_price, date).
    """
    db = get_db()
    result = {}
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        price_repo = IndexPriceRepository(session)
        
        indices = index_repo.get_all()
        index_ids = [idx.id for idx in indices]
        
        latest_prices = price_repo.get_latest_prices_for_indices(index_ids) if index_ids else {}
        
        for idx in indices:
            price = latest_prices.get(idx.id)
            if price and price.close:
                result[idx.symbol] = (price.close, price.date)
    
    return result


def get_index_returns(
    symbol: str,
    lookback_days: int = 365,
) -> dict[str, float | None]:
    """
    Calculate period returns for an index.
    
    Args:
        symbol: Index symbol.
        lookback_days: Days of history to consider.
        
    Returns:
        Dict with period returns: {"1m": 0.05, "3m": 0.12, "6m": 0.18, "1y": 0.25}
    """
    history = get_index_prices(symbol, lookback_days=lookback_days)
    if not history or not history.prices:
        return {"1m": None, "3m": None, "6m": None, "1y": None}
    
    prices_by_date = {p.date: p.close for p in history.prices if p.close}
    if not prices_by_date:
        return {"1m": None, "3m": None, "6m": None, "1y": None}
    
    sorted_dates = sorted(prices_by_date.keys())
    latest_date = sorted_dates[-1]
    latest_price = prices_by_date[latest_date]
    
    periods = {
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365,
    }
    
    results = {}
    for period_name, days in periods.items():
        target_date = (date.fromisoformat(latest_date) - timedelta(days=days)).isoformat()
        
        # Find closest date on or before target
        prior_price = None
        for d in reversed(sorted_dates):
            if d <= target_date:
                prior_price = prices_by_date.get(d)
                break
        
        if prior_price and prior_price > 0:
            results[period_name] = (latest_price - prior_price) / prior_price
        else:
            results[period_name] = None
    
    return results


# ============================================================================
# Mutation Functions
# ============================================================================

def create_index(
    symbol: str,
    name: str,
    description: str | None = None,
    category: str = "EQUITY",
) -> MarketIndex:
    """
    Create a new market index.
    
    Args:
        symbol: Unique symbol for the index.
        name: Human-readable name.
        description: Optional description.
        category: Index category (EQUITY, VOLATILITY, etc.).
        
    Returns:
        Created MarketIndex model.
    """
    db = get_db()
    with db.session() as session:
        repo = MarketIndexRepository(session)
        idx = repo.create(
            symbol=symbol,
            name=name,
            description=description,
            category=category,
        )
        session.commit()
        return idx


def get_or_create_index(
    symbol: str,
    name: str,
    **kwargs,
) -> tuple[MarketIndex, bool]:
    """
    Get existing index or create a new one.
    
    Returns:
        Tuple of (index, created) where created is True if newly created.
    """
    db = get_db()
    with db.session() as session:
        repo = MarketIndexRepository(session)
        idx, created = repo.get_or_create(symbol=symbol, name=name, **kwargs)
        session.commit()
        return idx, created


def sync_index_prices(
    symbol: str | None = None,
    lookback_days: int | None = None,
) -> list[SyncResult]:
    """
    Sync index prices from Yahoo Finance.
    
    If symbol is None, syncs all tracked indices.
    
    Args:
        symbol: Optional specific index to sync.
        lookback_days: Days of history to fetch (default from config).
        
    Returns:
        List of SyncResult for each index synced.
    """
    from data.yfinance_fetcher import IndexPriceFetcher
    
    db = get_db()
    results = []
    lookback = lookback_days or config.data_fetcher.default_lookback_days
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        price_repo = IndexPriceRepository(session)
        
        if symbol:
            indices = [index_repo.get_by_symbol(symbol)]
            indices = [idx for idx in indices if idx]
        else:
            indices = list(index_repo.get_all())
        
        if not indices:
            return results
        
        fetcher = IndexPriceFetcher()
        
        for idx in indices:
            try:
                # Get Yahoo symbol from config
                yahoo_symbol = config.market_indices.get_yahoo_symbol(idx.symbol)
                
                # Fetch price data
                prices_df = fetcher.fetch_history(yahoo_symbol, lookback_days=lookback)
                
                if prices_df.empty:
                    results.append(SyncResult(
                        symbol=idx.symbol,
                        records_added=0,
                        latest_date=None,
                        success=False,
                        error="No price data returned",
                    ))
                    continue
                
                # Convert DataFrame to list of dicts
                price_records = []
                for dt, row in prices_df.iterrows():
                    price_records.append({
                        "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10],
                        "open": row.get("Open"),
                        "high": row.get("High"),
                        "low": row.get("Low"),
                        "close": row.get("Close"),
                        "volume": int(row.get("Volume", 0)) if row.get("Volume") else None,
                    })
                
                # Bulk upsert
                count = price_repo.bulk_upsert_prices(idx.id, price_records)
                session.commit()
                
                # Get latest date
                latest_date = price_repo.get_latest_date(idx.id)
                
                results.append(SyncResult(
                    symbol=idx.symbol,
                    records_added=count,
                    latest_date=latest_date,
                    success=True,
                ))
                
            except Exception as e:
                session.rollback()
                results.append(SyncResult(
                    symbol=idx.symbol,
                    records_added=0,
                    latest_date=None,
                    success=False,
                    error=str(e),
                ))
    
    return results


def add_index_prices(
    symbol: str,
    prices: list[dict],
) -> int:
    """
    Manually add price records for an index.
    
    Args:
        symbol: Index symbol.
        prices: List of price dicts with date, open, high, low, close, volume.
        
    Returns:
        Number of records added.
        
    Raises:
        ValueError: If index not found.
    """
    db = get_db()
    
    with db.session() as session:
        index_repo = MarketIndexRepository(session)
        price_repo = IndexPriceRepository(session)
        
        idx = index_repo.get_by_symbol(symbol)
        if not idx:
            raise ValueError(f"Index not found: {symbol}")
        
        count = price_repo.bulk_upsert_prices(idx.id, prices)
        session.commit()
        return count


# ============================================================================
# Utility Functions
# ============================================================================

def get_configured_indices() -> list[dict]:
    """
    Get list of indices configured in config.py.
    
    Returns:
        List of dicts with symbol, name, category, yahoo_symbol.
    """
    try:
        if hasattr(config, 'market_indices') and config.market_indices.tracked_indices:
            return [
                {
                    "symbol": idx.symbol,
                    "name": idx.name,
                    "description": idx.description,
                    "category": idx.category,
                    "yahoo_symbol": idx.sources.get("yahoo", idx.symbol),
                }
                for idx in config.market_indices.tracked_indices
            ]
    except AttributeError:
        pass
    
    return []


def ensure_indices_seeded() -> list[SyncResult]:
    """
    Ensure all configured indices exist in the database.
    
    Creates missing indices but does NOT fetch prices.
    Use sync_index_prices() to populate price data.
    
    Returns:
        List of SyncResult indicating which indices were created.
    """
    configured = get_configured_indices()
    results = []
    
    for idx_config in configured:
        idx, created = get_or_create_index(
            symbol=idx_config["symbol"],
            name=idx_config["name"],
            description=idx_config.get("description"),
            category=idx_config.get("category", "EQUITY"),
        )
        results.append(SyncResult(
            symbol=idx.symbol,
            records_added=0,
            latest_date=None,
            success=True,
            error="Created" if created else "Already exists",
        ))
    
    return results


@dataclass
class NormalizedPricePoint:
    """Single normalized price point (base = 1.0)."""
    date: str
    value: float  # Normalized value (first day = 1.0)


@dataclass
class NormalizedPriceSeries:
    """Normalized price series for an index or portfolio."""
    symbol: str
    name: str
    prices: list[NormalizedPricePoint]
    start_date: str | None
    end_date: str | None
    total_return: float | None  # Total return over period


def get_normalized_index_prices(
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    lookback_days: int | None = None,
) -> NormalizedPriceSeries | None:
    """
    Get normalized price series for an index (first day = 1.0).
    
    This is useful for comparing performance across different assets
    on the same chart.
    
    Args:
        symbol: Index symbol (e.g., "SPX").
        start_date: Start date (YYYY-MM-DD or date object).
        end_date: End date (YYYY-MM-DD or date object).
        lookback_days: Alternative to start_date - number of days to look back.
        
    Returns:
        NormalizedPriceSeries with normalized values, or None if not found.
    """
    history = get_index_prices(
        symbol,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
    )
    
    if not history or not history.prices:
        return None
    
    # Filter to prices with valid close values
    valid_prices = [p for p in history.prices if p.close is not None and p.close > 0]
    if not valid_prices:
        return None
    
    # Normalize to first day = 1.0
    base_price = valid_prices[0].close
    normalized_points = [
        NormalizedPricePoint(
            date=p.date,
            value=p.close / base_price,
        )
        for p in valid_prices
    ]
    
    # Calculate total return
    if len(normalized_points) >= 2:
        total_return = normalized_points[-1].value - 1.0
    else:
        total_return = None
    
    return NormalizedPriceSeries(
        symbol=history.symbol,
        name=history.name,
        prices=normalized_points,
        start_date=normalized_points[0].date if normalized_points else None,
        end_date=normalized_points[-1].date if normalized_points else None,
        total_return=total_return,
    )


def get_benchmark_comparison_data(
    benchmark_symbols: list[str],
    lookback_days: int = 365,
) -> dict[str, NormalizedPriceSeries]:
    """
    Get normalized price series for multiple benchmarks.
    
    Args:
        benchmark_symbols: List of index symbols to fetch.
        lookback_days: Days of history to fetch.
        
    Returns:
        Dict mapping symbol -> NormalizedPriceSeries
    """
    result = {}
    
    for symbol in benchmark_symbols:
        series = get_normalized_index_prices(symbol, lookback_days=lookback_days)
        if series:
            result[symbol] = series
    
    return result


def delete_index(symbol: str) -> bool:
    """
    Delete a market index and all its price data.
    
    Args:
        symbol: Index symbol to delete.
        
    Returns:
        True if deleted, False if not found.
    """
    db = get_db()
    with db.session() as session:
        repo = MarketIndexRepository(session)
        idx = repo.get_by_symbol(symbol)
        if not idx:
            return False
        repo.delete(idx.id)
        session.commit()
        return True
