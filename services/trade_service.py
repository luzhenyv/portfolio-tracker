"""
Trade Service - Handles trade history, editing, and reconciliation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from db import get_db, Trade, TradeAction
from db.repositories import (
    AssetRepository,
    CashRepository,
    PositionRepository,
    TradeRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeEditResult:
    """Result of a trade edit or delete operation."""
    success: bool
    trade_id: int | None = None
    asset_id: int | None = None
    ticker: str | None = None
    errors: list[str] = field(default_factory=list)
    message: str = ""


def get_recent_trades(limit: int = 10) -> Sequence[Trade]:
    """Get recent trade activity."""
    db = get_db()
    with db.session() as session:
        trade_repo = TradeRepository(session)
        return trade_repo.get_all_trades(limit=limit)


def get_latest_trade_ids_by_ticker() -> dict[str, int]:
    """
    Get the latest trade ID for each ticker.
    
    Used to determine which trades are editable (only latest per ticker).
    
    Returns:
        Dict mapping ticker to latest trade_id
    """
    db = get_db()
    with db.session() as session:
        trade_repo = TradeRepository(session)
        return trade_repo.get_latest_trade_ids_by_ticker()


def reconcile_position_for_asset(session, asset_id: int) -> None:
    """
    Rebuild position state for an asset by replaying all its trades.
    
    This recalculates:
    - long_shares, long_avg_cost
    - short_shares, short_avg_price
    - realized_pnl
    - net_invested
    
    Args:
        session: Active database session
        asset_id: Asset ID to reconcile
    """
    trade_repo = TradeRepository(session)
    position_repo = PositionRepository(session)
    
    # Get or create position
    position, _ = position_repo.get_or_create(asset_id)
    
    # Reset position state
    position.long_shares = 0.0
    position.long_avg_cost = None
    position.short_shares = 0.0
    position.short_avg_price = None
    position.realized_pnl = 0.0
    position.net_invested = 0.0
    
    # Get all trades for this asset in chronological order
    trades = trade_repo.list_trades_for_asset_chronological(asset_id)
    
    # Replay each trade
    for trade in trades:
        shares = trade.shares
        price = trade.price
        fees = trade.fees
        
        if trade.action == TradeAction.BUY:
            # If short shares exist, cover them first
            realized_pnl = 0.0
            if position.short_shares > 0:
                shares_to_cover = min(shares, position.short_shares)
                realized_pnl = shares_to_cover * (position.short_avg_price - price) - (
                    fees * shares_to_cover / trade.shares if trade.shares > 0 else 0
                )
                position.short_shares -= shares_to_cover
                position.realized_pnl += realized_pnl
                
                if position.short_shares == 0:
                    position.short_avg_price = None
                
                shares -= shares_to_cover
            
            # Add remaining shares to long position
            if shares > 0:
                if position.long_shares > 0 and position.long_avg_cost:
                    total_cost = position.long_shares * position.long_avg_cost + shares * price
                    position.long_shares += shares
                    position.long_avg_cost = total_cost / position.long_shares
                else:
                    position.long_shares = shares
                    position.long_avg_cost = price
            
            # Update trade's realized_pnl
            trade.realized_pnl = realized_pnl
        
        elif trade.action == TradeAction.SELL:
            # If long shares exist, sell them first
            realized_pnl = 0.0
            if position.long_shares > 0:
                shares_to_sell = min(shares, position.long_shares)
                realized_pnl = shares_to_sell * (price - (position.long_avg_cost or 0)) - (
                    fees * shares_to_sell / trade.shares if trade.shares > 0 else 0
                )
                position.long_shares -= shares_to_sell
                position.realized_pnl += realized_pnl
                
                if position.long_shares == 0:
                    position.long_avg_cost = None
                
                shares -= shares_to_sell
            
            # Remaining shares open or add to short position
            if shares > 0:
                if position.short_shares > 0 and position.short_avg_price:
                    total_proceeds = position.short_shares * position.short_avg_price + shares * price
                    position.short_shares += shares
                    position.short_avg_price = total_proceeds / position.short_shares
                else:
                    position.short_shares = shares
                    position.short_avg_price = price
            
            # Update trade's realized_pnl
            trade.realized_pnl = realized_pnl
    
    # Recalculate net_invested from trade history
    position = position_repo.recalculate_net_invested(asset_id)
    
    logger.info(
        f"Reconciled position for asset_id={asset_id}: "
        f"long={position.long_shares}, short={position.short_shares}, "
        f"realized_pnl={position.realized_pnl:.2f}, net_invested={position.net_invested:.2f}"
    )


def regenerate_cash_transactions_for_trade(session, trade: Trade) -> None:
    """
    Delete and recreate cash transactions for a specific trade.
    
    This handles the BUY/SELL cash flow and separate FEE transaction.
    
    Args:
        session: Active database session
        trade: Trade object to regenerate cash transactions for
    """
    cash_repo = CashRepository(session)
    
    # Delete existing cash transactions for this trade
    deleted_count = cash_repo.delete_transactions_by_trade_id(trade.id)
    logger.debug(f"Deleted {deleted_count} cash transactions for trade_id={trade.id}")
    
    # Recreate cash transactions
    cash_repo.record_trade_cash_flow(trade, fees=trade.fees)
    logger.debug(f"Recreated cash transactions for trade_id={trade.id}")


def update_trade(
    trade_id: int,
    shares: float | None = None,
    price: float | None = None,
    fees: float | None = None,
    trade_at: str | datetime | None = None,
) -> TradeEditResult:
    """
    Update an existing trade and reconcile dependent state.
    
    Only the latest trade per ticker can be edited.
    After update, regenerates cash transactions and rebuilds the position.
    
    Args:
        trade_id: ID of trade to update
        shares: New share count (if provided)
        price: New price per share (if provided)
        fees: New fee amount (if provided)
        trade_at: New trade date as YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, or datetime
        
    Returns:
        TradeEditResult with operation status
    """
    db = get_db()
    
    with db.session() as session:
        trade_repo = TradeRepository(session)
        
        # Get the trade
        trade = trade_repo.get_by_id(trade_id)
        if not trade:
            return TradeEditResult(
                success=False,
                trade_id=trade_id,
                errors=["Trade not found"],
                message=f"❌ Trade #{trade_id} not found",
            )
        
        ticker = trade.asset.ticker if trade.asset else "?"
        asset_id = trade.asset_id
        
        # Verify this is the latest trade for this ticker
        latest_by_ticker = trade_repo.get_latest_trade_ids_by_ticker()
        if latest_by_ticker.get(ticker) != trade_id:
            return TradeEditResult(
                success=False,
                trade_id=trade_id,
                asset_id=asset_id,
                ticker=ticker,
                errors=["Only the latest trade per ticker can be edited"],
                message=f"❌ Cannot edit trade #{trade_id} - only the latest trade for {ticker} can be edited",
            )
        
        # Update the trade
        trade = trade_repo.update_trade(
            trade_id=trade_id,
            shares=shares,
            price=price,
            fees=fees,
            trade_at=trade_at,
        )
        
        # Regenerate cash transactions for this trade
        regenerate_cash_transactions_for_trade(session, trade)
        
        # Reconcile the position
        reconcile_position_for_asset(session, asset_id)
        
        logger.info(f"Updated trade #{trade_id} for {ticker}")
        
        return TradeEditResult(
            success=True,
            trade_id=trade_id,
            asset_id=asset_id,
            ticker=ticker,
            message=f"✅ Updated trade #{trade_id} for {ticker}",
        )


def delete_trade(trade_id: int) -> TradeEditResult:
    """
    Delete a trade and reconcile dependent state.
    
    Only the latest trade per ticker can be deleted.
    After deletion, removes linked cash transactions and rebuilds the position.
    
    Args:
        trade_id: ID of trade to delete
        
    Returns:
        TradeEditResult with operation status
    """
    db = get_db()
    
    with db.session() as session:
        trade_repo = TradeRepository(session)
        cash_repo = CashRepository(session)
        
        # Get the trade
        trade = trade_repo.get_by_id(trade_id)
        if not trade:
            return TradeEditResult(
                success=False,
                trade_id=trade_id,
                errors=["Trade not found"],
                message=f"❌ Trade #{trade_id} not found",
            )
        
        ticker = trade.asset.ticker if trade.asset else "?"
        asset_id = trade.asset_id
        
        # Verify this is the latest trade for this ticker
        latest_by_ticker = trade_repo.get_latest_trade_ids_by_ticker()
        if latest_by_ticker.get(ticker) != trade_id:
            return TradeEditResult(
                success=False,
                trade_id=trade_id,
                asset_id=asset_id,
                ticker=ticker,
                errors=["Only the latest trade per ticker can be deleted"],
                message=f"❌ Cannot delete trade #{trade_id} - only the latest trade for {ticker} can be deleted",
            )
        
        # Delete cash transactions linked to this trade
        deleted_cash_count = cash_repo.delete_transactions_by_trade_id(trade_id)
        logger.debug(f"Deleted {deleted_cash_count} cash transactions for trade_id={trade_id}")
        
        # Delete the trade
        trade_repo.delete_trade(trade_id)
        
        # Reconcile the position
        reconcile_position_for_asset(session, asset_id)
        
        logger.info(f"Deleted trade #{trade_id} for {ticker}")
        
        return TradeEditResult(
            success=True,
            trade_id=trade_id,
            asset_id=asset_id,
            ticker=ticker,
            message=f"✅ Deleted trade #{trade_id} for {ticker}",
        )
