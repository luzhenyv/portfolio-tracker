"""
Repository pattern for data access operations.

Provides a clean abstraction layer between business logic and database operations.
Supports future extensions (caching, different backends, etc.).
"""

from datetime import datetime
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from db.models import (
    Asset,
    AssetStatus,
    Position,
    PositionState,
    PriceDaily,
    Trade,
    TradeAction,
    ValuationMetric,
)


class AssetRepository:
    """Repository for Asset-related operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_id(self, asset_id: int) -> Asset | None:
        """Get asset by ID."""
        return self.session.get(Asset, asset_id)
    
    def get_by_ticker(self, ticker: str) -> Asset | None:
        """Get asset by ticker symbol."""
        stmt = select(Asset).where(Asset.ticker == ticker.upper())
        return self.session.scalar(stmt)
    
    def get_all(self) -> Sequence[Asset]:
        """Get all assets."""
        stmt = select(Asset).order_by(Asset.ticker)
        return self.session.scalars(stmt).all()
    
    def get_by_status(self, status: AssetStatus) -> Sequence[Asset]:
        """Get assets filtered by status."""
        stmt = select(Asset).where(Asset.status == status).order_by(Asset.ticker)
        return self.session.scalars(stmt).all()
    
    def get_owned_assets(self) -> Sequence[Asset]:
        """Get all owned assets (convenience method)."""
        return self.get_by_status(AssetStatus.OWNED)
    
    def get_watchlist_assets(self) -> Sequence[Asset]:
        """Get all watchlist assets (convenience method)."""
        return self.get_by_status(AssetStatus.WATCHLIST)
    
    def get_with_positions(self, asset_id: int) -> Asset | None:
        """Get asset with positions eagerly loaded."""
        stmt = (
            select(Asset)
            .options(joinedload(Asset.positions))
            .where(Asset.id == asset_id)
        )
        return self.session.scalar(stmt)
    
    def create(
        self,
        ticker: str,
        status: AssetStatus,
        name: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        exchange: str | None = None,
    ) -> Asset:
        """Create a new asset."""
        asset = Asset(
            ticker=ticker.upper(),
            status=status,
            name=name,
            sector=sector,
            industry=industry,
            exchange=exchange,
        )
        self.session.add(asset)
        self.session.flush()  # Get the ID
        return asset
    
    def get_or_create(
        self,
        ticker: str,
        status: AssetStatus,
        **kwargs,
    ) -> tuple[Asset, bool]:
        """
        Get existing asset or create new one.
        
        Returns:
            Tuple of (asset, created) where created is True if new.
        """
        asset = self.get_by_ticker(ticker)
        if asset:
            return asset, False
        return self.create(ticker=ticker, status=status, **kwargs), True
    
    def update_status(self, asset_id: int, status: AssetStatus) -> Asset | None:
        """Update asset status."""
        asset = self.get_by_id(asset_id)
        if asset:
            asset.status = status
            self.session.flush()
        return asset


class PriceRepository:
    """Repository for price data operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_latest_date(self, asset_id: int) -> str | None:
        """Get the most recent price date for an asset."""
        stmt = (
            select(func.max(PriceDaily.date))
            .where(PriceDaily.asset_id == asset_id)
        )
        return self.session.scalar(stmt)
    
    def get_latest_price(self, asset_id: int) -> PriceDaily | None:
        """Get the most recent price record for an asset."""
        latest_date = self.get_latest_date(asset_id)
        if not latest_date:
            return None
        
        stmt = (
            select(PriceDaily)
            .where(PriceDaily.asset_id == asset_id, PriceDaily.date == latest_date)
        )
        return self.session.scalar(stmt)
    
    def get_price_history(
        self,
        asset_id: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Sequence[PriceDaily]:
        """Get price history for an asset within date range."""
        stmt = select(PriceDaily).where(PriceDaily.asset_id == asset_id)
        
        if start_date:
            stmt = stmt.where(PriceDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(PriceDaily.date <= end_date)
        
        stmt = stmt.order_by(PriceDaily.date)
        return self.session.scalars(stmt).all()
    
    def get_latest_prices_for_assets(
        self,
        asset_ids: list[int] | None = None,
    ) -> dict[int, PriceDaily]:
        """
        Get latest price for multiple assets.
        
        Returns:
            Dict mapping asset_id to latest PriceDaily.
        """
        # Subquery to get max date per asset
        subq = (
            select(
                PriceDaily.asset_id,
                func.max(PriceDaily.date).label("max_date"),
            )
            .group_by(PriceDaily.asset_id)
        )
        
        if asset_ids:
            subq = subq.where(PriceDaily.asset_id.in_(asset_ids))
        
        subq = subq.subquery()
        
        # Main query joining with subquery
        stmt = (
            select(PriceDaily)
            .join(
                subq,
                (PriceDaily.asset_id == subq.c.asset_id) &
                (PriceDaily.date == subq.c.max_date)
            )
        )
        
        prices = self.session.scalars(stmt).all()
        return {p.asset_id: p for p in prices}
    
    def bulk_upsert_prices(
        self,
        asset_id: int,
        price_records: list[dict],
    ) -> int:
        """
        Bulk insert/update price records (idempotent).
        
        Args:
            asset_id: Asset ID for the prices.
            price_records: List of dicts with date, open, high, low, close, etc.
            
        Returns:
            Number of records inserted.
        """
        count = 0
        for record in price_records:
            # Check if exists
            stmt = select(PriceDaily).where(
                PriceDaily.asset_id == asset_id,
                PriceDaily.date == record["date"],
            )
            existing = self.session.scalar(stmt)
            
            if not existing:
                price = PriceDaily(
                    asset_id=asset_id,
                    date=record["date"],
                    open=record.get("open"),
                    high=record.get("high"),
                    low=record.get("low"),
                    close=record.get("close"),
                    adjusted_close=record.get("adjusted_close"),
                    volume=record.get("volume"),
                )
                self.session.add(price)
                count += 1
        
        self.session.flush()
        return count


class PositionRepository:
    """Repository for position operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_id(self, position_id: int) -> Position | None:
        """Get position by ID."""
        return self.session.get(Position, position_id)
    
    def get_positions_for_asset(self, asset_id: int) -> Sequence[Position]:
        """Get all positions for an asset."""
        stmt = (
            select(Position)
            .where(Position.asset_id == asset_id)
            .order_by(Position.buy_date)
        )
        return self.session.scalars(stmt).all()
    
    def get_all_owned_positions(self) -> Sequence[Position]:
        """Get all positions for owned assets."""
        stmt = (
            select(Position)
            .join(Asset)
            .where(Asset.status == AssetStatus.OWNED)
            .options(joinedload(Position.asset))
        )
        return self.session.scalars(stmt).all()
    
    def get_position_summary(self) -> list[dict]:
        """
        Get aggregated position summary per asset.
        
        Returns:
            List of dicts with ticker, total_shares, avg_buy_price, total_cost.
        """
        stmt = (
            select(
                Asset.ticker,
                Asset.id.label("asset_id"),
                func.sum(Position.shares).label("total_shares"),
                func.sum(Position.shares * Position.buy_price).label("total_cost"),
            )
            .join(Asset)
            .where(Asset.status == AssetStatus.OWNED)
            .group_by(Asset.id, Asset.ticker)
        )
        
        results = self.session.execute(stmt).all()
        return [
            {
                "asset_id": r.asset_id,
                "ticker": r.ticker,
                "total_shares": r.total_shares,
                "total_cost": r.total_cost,
                "avg_buy_price": r.total_cost / r.total_shares if r.total_shares else 0,
            }
            for r in results
        ]
    
    def create(
        self,
        asset_id: int,
        buy_date: str,
        shares: float,
        buy_price: float,
    ) -> Position:
        """Create a new position."""
        position = Position(
            asset_id=asset_id,
            buy_date=buy_date,
            shares=shares,
            buy_price=buy_price,
        )
        self.session.add(position)
        self.session.flush()
        return position
    
    def delete(self, position_id: int) -> bool:
        """Delete a position by ID."""
        position = self.get_by_id(position_id)
        if position:
            self.session.delete(position)
            self.session.flush()
            return True
        return False


class ValuationRepository:
    """Repository for valuation metrics operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_asset_id(self, asset_id: int) -> ValuationMetric | None:
        """Get valuation metrics for an asset."""
        return self.session.get(ValuationMetric, asset_id)
    
    def get_all_with_assets(self) -> Sequence[ValuationMetric]:
        """Get all valuation metrics with asset data."""
        stmt = (
            select(ValuationMetric)
            .options(joinedload(ValuationMetric.asset))
        )
        return self.session.scalars(stmt).all()
    
    def get_for_watchlist(self) -> Sequence[ValuationMetric]:
        """Get valuation metrics for watchlist assets only."""
        stmt = (
            select(ValuationMetric)
            .join(Asset)
            .where(Asset.status == AssetStatus.WATCHLIST)
            .options(joinedload(ValuationMetric.asset))
        )
        return self.session.scalars(stmt).all()
    
    def upsert(
        self,
        asset_id: int,
        pe_forward: float | None = None,
        peg: float | None = None,
        ev_ebitda: float | None = None,
        revenue_growth: float | None = None,
        eps_growth: float | None = None,
    ) -> ValuationMetric:
        """
        Insert or update valuation metrics (FR-8, FR-9).
        
        Missing values are stored as NULL, no synthetic values allowed.
        """
        existing = self.get_by_asset_id(asset_id)
        
        if existing:
            existing.pe_forward = pe_forward
            existing.peg = peg
            existing.ev_ebitda = ev_ebitda
            existing.revenue_growth = revenue_growth
            existing.eps_growth = eps_growth
            existing.updated_at = datetime.utcnow()
            self.session.flush()
            return existing
        
        valuation = ValuationMetric(
            asset_id=asset_id,
            pe_forward=pe_forward,
            peg=peg,
            ev_ebitda=ev_ebitda,
            revenue_growth=revenue_growth,
            eps_growth=eps_growth,
        )
        self.session.add(valuation)
        self.session.flush()
        return valuation


class TradeRepository:
    """Repository for trade operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_trade(
        self,
        asset_id: int,
        trade_date: str,
        action: TradeAction,
        shares: float,
        price: float,
        fees: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> Trade:
        """Create a new trade record."""
        trade = Trade(
            asset_id=asset_id,
            trade_date=trade_date,
            action=action,
            shares=shares,
            price=price,
            fees=fees,
            realized_pnl=realized_pnl,
        )
        self.session.add(trade)
        self.session.flush()
        return trade
    
    def get_trades_for_asset(
        self,
        asset_id: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Sequence[Trade]:
        """Get trades for an asset within optional date range."""
        stmt = (
            select(Trade)
            .where(Trade.asset_id == asset_id)
        )
        
        if start_date:
            stmt = stmt.where(Trade.trade_date >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_date <= end_date)
        
        stmt = stmt.order_by(Trade.trade_date.desc(), Trade.id.desc())
        return self.session.scalars(stmt).all()
    
    def get_all_trades(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        action: TradeAction | None = None,
        limit: int | None = None,
    ) -> Sequence[Trade]:
        """Get all trades with optional filters."""
        stmt = select(Trade).options(joinedload(Trade.asset))
        
        if start_date:
            stmt = stmt.where(Trade.trade_date >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_date <= end_date)
        if action:
            stmt = stmt.where(Trade.action == action)
        
        stmt = stmt.order_by(Trade.trade_date.desc(), Trade.id.desc())
        
        if limit:
            stmt = stmt.limit(limit)
        
        return self.session.scalars(stmt).all()
    
    def get_realized_pnl_summary(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, float]:
        """Get realized P&L summary across all trades."""
        stmt = select(
            func.sum(Trade.realized_pnl).label("total_realized"),
            func.sum(func.abs(Trade.fees)).label("total_fees"),
        )
        
        if start_date:
            stmt = stmt.where(Trade.trade_date >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_date <= end_date)
        
        result = self.session.execute(stmt).first()
        
        return {
            "total_realized_pnl": result.total_realized or 0.0,
            "total_fees": result.total_fees or 0.0,
            "net_realized_pnl": (result.total_realized or 0.0) - (result.total_fees or 0.0),
        }


class PositionStateRepository:
    """Repository for position state operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_asset_id(self, asset_id: int) -> PositionState | None:
        """Get position state for an asset."""
        return self.session.get(PositionState, asset_id)
    
    def get_or_create(self, asset_id: int) -> tuple[PositionState, bool]:
        """Get existing position state or create a new one."""
        state = self.get_by_asset_id(asset_id)
        if state:
            return state, False
        
        state = PositionState(asset_id=asset_id)
        self.session.add(state)
        self.session.flush()
        return state, True
    
    def get_all_active_positions(self) -> Sequence[PositionState]:
        """Get all position states with non-zero long or short shares."""
        stmt = (
            select(PositionState)
            .options(joinedload(PositionState.asset))
            .where(
                (PositionState.long_shares > 0) | (PositionState.short_shares > 0)
            )
            .order_by(PositionState.asset_id)
        )
        return self.session.scalars(stmt).all()
    
    def get_position_summary(self) -> list[dict]:
        """
        Get position summary for analytics (replaces old position aggregation).
        
        Returns list of dicts with ticker, long/short shares, avg costs, etc.
        """
        stmt = (
            select(
                Asset.ticker,
                Asset.id.label("asset_id"),
                PositionState.long_shares,
                PositionState.long_avg_cost,
                PositionState.short_shares,
                PositionState.short_avg_price,
                PositionState.realized_pnl,
            )
            .select_from(PositionState)
            .join(Asset, Asset.id == PositionState.asset_id)
            .where(
                Asset.status == AssetStatus.OWNED,
                (PositionState.long_shares > 0) | (PositionState.short_shares > 0)
            )
        )
        
        results = self.session.execute(stmt).all()
        return [
            {
                "asset_id": r.asset_id,
                "ticker": r.ticker,
                "long_shares": r.long_shares,
                "long_avg_cost": r.long_avg_cost or 0.0,
                "short_shares": r.short_shares,
                "short_avg_price": r.short_avg_price or 0.0,
                "realized_pnl": r.realized_pnl,
            }
            for r in results
        ]
    
    def update_state(
        self,
        asset_id: int,
        long_shares: float | None = None,
        long_avg_cost: float | None = None,
        short_shares: float | None = None,
        short_avg_price: float | None = None,
        realized_pnl: float | None = None,
    ) -> PositionState:
        """Update position state fields."""
        state, _ = self.get_or_create(asset_id)
        
        if long_shares is not None:
            state.long_shares = long_shares
        if long_avg_cost is not None:
            state.long_avg_cost = long_avg_cost
        if short_shares is not None:
            state.short_shares = short_shares
        if short_avg_price is not None:
            state.short_avg_price = short_avg_price
        if realized_pnl is not None:
            state.realized_pnl = realized_pnl
        
        self.session.flush()
        return state

