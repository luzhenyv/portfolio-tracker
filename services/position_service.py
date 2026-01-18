"""
Position Service - Handles position creation and trading with automatic status management.

This service layer provides functionality for:
- Buy/Sell operations for long positions
- Short/Cover operations for short positions
- Average cost basis tracking
- Realized P&L calculation
- Asset status transitions (WATCHLIST to OWNED)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from db import Asset, AssetStatus, Position, Trade, TradeAction, get_db
from db.repositories import AssetRepository, PositionRepository, TradeRepository


logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """
    Result object for trade operations (buy/sell/short/cover).
    
    Captures the outcome of a trade including position state changes,
    realized P&L, and any status transitions.
    """
    trade: Trade | None
    asset: Asset | None
    position: Position | None
    success: bool
    action: TradeAction | None
    realized_pnl: float = 0.0
    status_changed: bool = False  # WATCHLIST -> OWNED
    errors: list[str] = field(default_factory=list)
    status_message: str = ""


def execute_trade(
    ticker: str,
    action: TradeAction,
    shares: float,
    price: float,
    trade_date: str | None = None,
    fees: float = 0.0,
) -> TradeResult:
    """
    Execute a trade (BUY/SELL/SHORT/COVER) with average cost accounting.
    
    Args:
        ticker: Stock ticker symbol
        action: Trade action (BUY/SELL/SHORT/COVER)
        shares: Number of shares (positive)
        price: Price per share
        trade_date: Trade date in YYYY-MM-DD format (defaults to today)
        fees: Trading fees (optional)
        
    Returns:
        TradeResult with detailed outcome information
        
    Raises:
        ValueError: If shares <= 0 or price <= 0
    """
    if shares <= 0:
        return TradeResult(
            trade=None,
            asset=None,
            position=None,
            success=False,
            action=action,
            errors=["Shares must be positive"],
            status_message="❌ Shares must be positive",
        )
    
    if price <= 0:
        return TradeResult(
            trade=None,
            asset=None,
            position=None,
            success=False,
            action=action,
            errors=["Price must be positive"],
            status_message="❌ Price must be positive",
        )
    
    ticker = ticker.upper()
    status_changed = False
    
    # Default to today's date if not specified
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    db = get_db()
    
    with db.session() as session:
        asset_repo = AssetRepository(session)
        position_repo = PositionRepository(session)
        trade_repo = TradeRepository(session)
        
        # Validate asset exists
        asset = asset_repo.get_by_ticker(ticker)
        
        if not asset:
            return TradeResult(
                trade=None,
                asset=None,
                position=None,
                success=False,
                action=action,
                errors=[f"Asset not found: {ticker}"],
                status_message=f"❌ Asset not found: {ticker}\n   Use 'add-asset' to add it first.",
            )
        
        # Update asset status to OWNED if currently WATCHLIST (for BUY/SHORT)
        if action in (TradeAction.BUY, TradeAction.SHORT) and asset.status == AssetStatus.WATCHLIST:
            asset_repo.update_status(asset.id, AssetStatus.OWNED)
            asset.status = AssetStatus.OWNED
            status_changed = True
            logger.info(f"Updated {ticker} status from WATCHLIST to OWNED")
        
        # Get or create position state
        position, _ = position_repo.get_or_create(asset.id)
        
        # Calculate realized P&L and update state based on action
        realized_pnl = 0.0
        
        try:
            if action == TradeAction.BUY:
                # Add to long position with average cost
                if position.long_shares > 0:
                    total_cost = position.long_shares * position.long_avg_cost + shares * price
                    position.long_shares += shares
                    position.long_avg_cost = total_cost / position.long_shares
                else:
                    position.long_shares = shares
                    position.long_avg_cost = price
            
            elif action == TradeAction.SELL:
                # Reduce long position and calculate realized P&L
                if shares > position.long_shares:
                    return TradeResult(
                        trade=None,
                        asset=asset,
                        position=position,
                        success=False,
                        action=action,
                        errors=[f"Cannot sell {shares} shares; only {position.long_shares} shares held"],
                        status_message=f"❌ Cannot sell {shares} shares; only {position.long_shares:.2f} shares held",
                    )
                
                # Calculate realized P&L (avg cost method)
                realized_pnl = shares * (price - position.long_avg_cost) - fees
                position.long_shares -= shares
                position.realized_pnl += realized_pnl
                
                # Clear avg cost if fully closed
                if position.long_shares == 0:
                    position.long_avg_cost = None
            
            elif action == TradeAction.SHORT:
                # Add to short position with average price
                if position.short_shares > 0:
                    total_proceeds = position.short_shares * position.short_avg_price + shares * price
                    position.short_shares += shares
                    position.short_avg_price = total_proceeds / position.short_shares
                else:
                    position.short_shares = shares
                    position.short_avg_price = price
            
            elif action == TradeAction.COVER:
                # Reduce short position and calculate realized P&L
                if shares > position.short_shares:
                    return TradeResult(
                        trade=None,
                        asset=asset,
                        position=position,
                        success=False,
                        action=action,
                        errors=[f"Cannot cover {shares} shares; only {position.short_shares} shares short"],
                        status_message=f"❌ Cannot cover {shares} shares; only {position.short_shares:.2f} shares short",
                    )
                
                # Calculate realized P&L for short cover
                realized_pnl = shares * (position.short_avg_price - price) - fees
                position.short_shares -= shares
                position.realized_pnl += realized_pnl
                
                # Clear avg price if fully closed
                if position.short_shares == 0:
                    position.short_avg_price = None
            
            # Create trade record
            trade = trade_repo.create_trade(
                asset_id=asset.id,
                trade_date=trade_date,
                action=action,
                shares=shares,
                price=price,
                fees=fees,
                realized_pnl=realized_pnl,
            )
            
            # Build status message
            action_verb = {
                TradeAction.BUY: "Bought",
                TradeAction.SELL: "Sold",
                TradeAction.SHORT: "Shorted",
                TradeAction.COVER: "Covered",
            }
            
            status_message = f"✅ {action_verb[action]} {shares} shares of {ticker} @ ${price:.2f}"
            
            if realized_pnl != 0:
                status_message += f"\n   💰 Realized P&L: ${realized_pnl:+,.2f}"
            
            if status_changed:
                status_message += f"\n   📍 Status changed: WATCHLIST → OWNED"
            
            return TradeResult(
                trade=trade,
                asset=asset,
                position=position,
                success=True,
                action=action,
                realized_pnl=realized_pnl,
                status_changed=status_changed,
                errors=[],
                status_message=status_message,
            )
            
        except Exception as e:
            logger.error(f"Failed to execute {action} for {ticker}: {e}")
            return TradeResult(
                trade=None,
                asset=asset,
                position=position,
                success=False,
                action=action,
                errors=[str(e)],
                status_message=f"❌ Failed to execute {action}: {str(e)}",
            )


def buy_position(
    ticker: str,
    shares: float,
    price: float,
    trade_date: str | None = None,
    fees: float = 0.0,
) -> TradeResult:
    """
    Buy shares (open or add to long position).
    
    Example:
        >>> result = buy_position("AAPL", 100, 150.00, "2026-01-18")
    """
    return execute_trade(ticker, TradeAction.BUY, shares, price, trade_date, fees)


def sell_position(
    ticker: str,
    shares: float,
    price: float,
    trade_date: str | None = None,
    fees: float = 0.0,
) -> TradeResult:
    """
    Sell shares (reduce or close long position).
    
    Validates that sufficient shares are held. Calculates realized P&L using
    average cost method.
    
    Example:
        >>> result = sell_position("NVDA", 10, 180.00, "2026-01-18")
    """
    return execute_trade(ticker, TradeAction.SELL, shares, price, trade_date, fees)


def short_position(
    ticker: str,
    shares: float,
    price: float,
    trade_date: str | None = None,
    fees: float = 0.0,
) -> TradeResult:
    """
    Short shares (open or add to short position).
    
    Example:
        >>> result = short_position("TSLA", 50, 200.00, "2026-01-18")
    """
    return execute_trade(ticker, TradeAction.SHORT, shares, price, trade_date, fees)


def cover_position(
    ticker: str,
    shares: float,
    price: float,
    trade_date: str | None = None,
    fees: float = 0.0,
) -> TradeResult:
    """
    Cover short shares (reduce or close short position).
    
    Validates that sufficient short shares exist. Calculates realized P&L.
    
    Example:
        >>> result = cover_position("TSLA", 25, 190.00, "2026-01-18")
    """
    return execute_trade(ticker, TradeAction.COVER, shares, price, trade_date, fees)


def print_trade_result(result: TradeResult) -> None:
    """
    Print a formatted trade result to console.
    
    Helper function for CLI usage to display results in a user-friendly way.
    """
    print(result.status_message)
    
    if result.success and result.trade:
        print(f"   Date: {result.trade.trade_date}")
        if result.trade.fees > 0:
            print(f"   Fees: ${result.trade.fees:.2f}")
        
        if result.position:
            state = result.position
            if state.long_shares > 0:
                print(f"   Long Position: {state.long_shares:.2f} shares @ ${state.long_avg_cost:.2f} avg")
            if state.short_shares > 0:
                print(f"   Short Position: {state.short_shares:.2f} shares @ ${state.short_avg_price:.2f} avg")
            if state.realized_pnl != 0:
                print(f"   Total Realized P&L: ${state.realized_pnl:+,.2f}")

