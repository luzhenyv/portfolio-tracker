"""
Position Service - Handles position creation with automatic status management.

This service layer provides functionality for creating positions and managing
asset status transitions (e.g., WATCHLIST to OWNED when a position is added).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from db import Asset, AssetStatus, Position, get_db
from db.repositories import AssetRepository, PositionRepository


logger = logging.getLogger(__name__)


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
