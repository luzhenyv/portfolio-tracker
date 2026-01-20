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
    CashTransaction,
    CashTransactionType,
    Position,
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
    
    def get_with_position(self, asset_id: int) -> Asset | None:
        """Get asset with position state eagerly loaded."""
        stmt = (
            select(Asset)
            .options(joinedload(Asset.position))
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

    def get_price_history_for_assets(
        self,
        asset_ids: list[int] | None = None,
        status: AssetStatus | None = None,
    ) -> list[dict]:
        """
        Get price history for multiple assets (FR-5).

        Uses raw close prices for NAV/returns calculation.
        Dividends are tracked separately in the cash ledger.

        Returns:
            List of dicts with [date, ticker, close].
        """
        stmt = (
            select(
                PriceDaily.date,
                Asset.ticker,
                PriceDaily.close,
            )
            .join(Asset, PriceDaily.asset_id == Asset.id)
            .where(PriceDaily.close.is_not(None))
        )

        if asset_ids:
            stmt = stmt.where(Asset.id.in_(asset_ids))
        if status:
            stmt = stmt.where(Asset.status == status)

        stmt = stmt.order_by(PriceDaily.date)

        results = self.session.execute(stmt).all()
        return [
            {
                "date": r.date,
                "ticker": r.ticker,
                "close": r.close,
            }
            for r in results
        ]


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

    def list_all_chronological(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Sequence[Trade]:
        """
        Get all trades in chronological order for position reconstruction.
        
        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)
            
        Returns:
            Trades sorted by date ASC, id ASC
        """
        stmt = select(Trade).options(joinedload(Trade.asset))
        
        if start_date:
            stmt = stmt.where(Trade.trade_date >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_date <= end_date)
        
        stmt = stmt.order_by(Trade.trade_date.asc(), Trade.id.asc())
        return self.session.scalars(stmt).all()


class PositionRepository:
    """
    Repository for position state operations.
    
    Manages current position inventory (long/short) and average costs.
    Works with Trade ledger for complete transaction history.
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_asset_id(self, asset_id: int) -> Position | None:
        """Get position state for an asset."""
        return self.session.get(Position, asset_id)
    
    def get_or_create(self, asset_id: int) -> tuple[Position, bool]:
        """Get existing position state or create a new one."""
        position = self.get_by_asset_id(asset_id)
        if position:
            return position, False
        
        position = Position(asset_id=asset_id)
        self.session.add(position)
        self.session.flush()
        return position, True
    
    def get_all_active_positions(self) -> Sequence[Position]:
        """Get all position states with non-zero long or short shares."""
        stmt = (
            select(Position)
            .options(joinedload(Position.asset))
            .where(
                (Position.long_shares > 0) | (Position.short_shares > 0)
            )
            .order_by(Position.asset_id)
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
                Position.long_shares,
                Position.long_avg_cost,
                Position.net_invested,
                Position.short_shares,
                Position.short_avg_price,
                Position.realized_pnl,
            )
            .select_from(Position)
            .join(Asset, Asset.id == Position.asset_id)
            .where(
                Asset.status == AssetStatus.OWNED,
                (Position.long_shares > 0) | (Position.short_shares > 0)
            )
        )
        
        results = self.session.execute(stmt).all()
        return [
            {
                "asset_id": r.asset_id,
                "ticker": r.ticker,
                "long_shares": r.long_shares,
                "long_avg_cost": r.long_avg_cost or 0.0,
                "net_invested": r.net_invested or 0.0,
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
    ) -> Position:
        """Update position state fields."""
        position, _ = self.get_or_create(asset_id)
        
        if long_shares is not None:
            position.long_shares = long_shares
        if long_avg_cost is not None:
            position.long_avg_cost = long_avg_cost
        if short_shares is not None:
            position.short_shares = short_shares
        if short_avg_price is not None:
            position.short_avg_price = short_avg_price
        if realized_pnl is not None:
            position.realized_pnl = realized_pnl
        
        self.session.flush()
        return position
    
    
    def get_net_invested_by_asset(self, asset_id: int) -> float:
        """
        Calculate net invested amount for a single asset from trade history.
        
        Net invested = Σ(buy_shares * buy_price + buy_fees) - Σ(sell_shares * sell_price - sell_fees)
        
        Returns:
            Net invested amount (cash still at risk after taking profits).
        """
        # Get all trades for this asset
        stmt = (
            select(Trade)
            .where(Trade.asset_id == asset_id)
            .order_by(Trade.trade_date, Trade.id)
        )
        trades = self.session.scalars(stmt).all()
        
        net_invested = 0.0
        for trade in trades:
            if trade.action in (TradeAction.BUY, TradeAction.COVER):
                # Money going out (buying)
                net_invested += trade.shares * trade.price + trade.fees
            elif trade.action in (TradeAction.SELL, TradeAction.SHORT):
                # Money coming in (selling)
                net_invested -= trade.shares * trade.price - trade.fees
        
        return net_invested
    
    def recalculate_net_invested(self, asset_id: int) -> Position:
        """
        Recalculate and store net_invested for an asset from trade history.
        
        This should be called after each trade to keep the position's
        net_invested field in sync with the trade ledger.
        
        Net Invested Avg Cost = net_invested / long_shares
        
        Returns:
            Updated Position with recalculated net_invested.
        """
        position, _ = self.get_or_create(asset_id)
        
        # Calculate net invested from trade history
        net_invested = self.get_net_invested_by_asset(asset_id)
        
        # Store in position record
        position.net_invested = net_invested
        self.session.flush()
        
        return position
    
    def recalculate_all_net_invested(self) -> dict[int, float]:
        """
        Recalculate net_invested for all assets with positions.
        
        Useful for data migration or fixing historical records.
        
        Returns:
            Dict mapping asset_id to updated net_invested values.
        """
        # Get all assets with active positions
        stmt = (
            select(Position.asset_id)
            .where((Position.long_shares > 0) | (Position.short_shares > 0))
        )
        asset_ids = self.session.scalars(stmt).all()
        
        result = {}
        for asset_id in asset_ids:
            position = self.recalculate_net_invested(asset_id)
            result[asset_id] = position.net_invested
        
        return result
    
    def get_net_invested_summary(self) -> dict[int, float]:
        """
        Get net invested amount for all assets with positions.
        
        Returns stored net_invested values from positions table.
        
        Returns:
            Dict mapping asset_id to net invested amount.
        """
        stmt = (
            select(Position.asset_id, Position.net_invested)
            .where((Position.long_shares > 0) | (Position.short_shares > 0))
        )
        results = self.session.execute(stmt).all()
        
        return {r.asset_id: r.net_invested for r in results}


class CashRepository:
    """
    Repository for cash transaction operations.
    
    Manages cash position tracking including:
    - Deposits/withdrawals
    - Trade-related cash flows
    - Cash balance calculations
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_transaction(
        self,
        transaction_date: str,
        transaction_type: CashTransactionType,
        amount: float,
        trade_id: int | None = None,
        asset_id: int | None = None,
        description: str | None = None,
    ) -> CashTransaction:
        """
        Create a new cash transaction.
        
        Args:
            transaction_date: Date in YYYY-MM-DD format
            transaction_type: Type of transaction
            amount: Signed amount (+ inflow, - outflow)
            trade_id: Optional reference to related trade
            asset_id: Optional reference to related asset
            description: Optional description
            
        Returns:
            Created CashTransaction
        """
        transaction = CashTransaction(
            transaction_date=transaction_date,
            transaction_type=transaction_type,
            amount=amount,
            trade_id=trade_id,
            asset_id=asset_id,
            description=description,
        )
        self.session.add(transaction)
        self.session.flush()
        return transaction
    
    def deposit(
        self,
        amount: float,
        transaction_date: str | None = None,
        description: str | None = None,
    ) -> CashTransaction:
        """
        Record a cash deposit (capital injection).
        
        Args:
            amount: Deposit amount (positive)
            transaction_date: Date (defaults to today)
            description: Optional description
            
        Returns:
            Created CashTransaction
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        if transaction_date is None:
            transaction_date = datetime.now().strftime("%Y-%m-%d")
        
        return self.create_transaction(
            transaction_date=transaction_date,
            transaction_type=CashTransactionType.DEPOSIT,
            amount=amount,
            description=description or "Cash deposit",
        )
    
    def withdraw(
        self,
        amount: float,
        transaction_date: str | None = None,
        description: str | None = None,
    ) -> CashTransaction:
        """
        Record a cash withdrawal.
        
        Args:
            amount: Withdrawal amount (positive, will be stored as negative)
            transaction_date: Date (defaults to today)
            description: Optional description
            
        Returns:
            Created CashTransaction
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if transaction_date is None:
            transaction_date = datetime.now().strftime("%Y-%m-%d")
        
        return self.create_transaction(
            transaction_date=transaction_date,
            transaction_type=CashTransactionType.WITHDRAW,
            amount=-amount,  # Store as negative (outflow)
            description=description or "Cash withdrawal",
        )
    
    def record_dividend(
        self,
        asset_id: int,
        amount: float,
        transaction_date: str,
        description: str | None = None,
    ) -> tuple[CashTransaction, bool]:
        """
        Record a dividend payment with idempotency.
        
        Uses deduplication strategy: asset_id + transaction_date + amount
        to prevent duplicate dividend entries on reruns.
        
        Args:
            asset_id: Asset ID for dividend attribution
            amount: Dividend amount (positive, cash inflow)
            transaction_date: Ex-dividend or payment date (YYYY-MM-DD)
            description: Optional description
            
        Returns:
            Tuple of (CashTransaction, created) where created=False for duplicates
            
        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError("Dividend amount must be positive")
        
        # Check for duplicate
        existing = self.find_dividend(asset_id, amount, transaction_date)
        if existing:
            return existing, False
        
        # Create new dividend transaction
        transaction = CashTransaction(
            transaction_date=transaction_date,
            transaction_type=CashTransactionType.DIVIDEND,
            amount=amount,  # Positive for inflow
            asset_id=asset_id,
            description=description or "Dividend",
        )
        self.session.add(transaction)
        self.session.flush()
        return transaction, True
    
    def find_dividend(
        self,
        asset_id: int,
        amount: float,
        transaction_date: str,
    ) -> CashTransaction | None:
        """
        Find existing dividend matching the dedup key.
        
        Dedup key: asset_id + transaction_date + amount
        
        Args:
            asset_id: Asset ID
            amount: Dividend amount
            transaction_date: Transaction date
            
        Returns:
            Existing CashTransaction if found, else None
        """
        stmt = (
            select(CashTransaction)
            .where(
                CashTransaction.asset_id == asset_id,
                CashTransaction.transaction_date == transaction_date,
                CashTransaction.transaction_type == CashTransactionType.DIVIDEND,
                CashTransaction.amount == amount,
            )
        )
        return self.session.scalar(stmt)
    
    def get_dividends(
        self,
        asset_id: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Sequence[CashTransaction]:
        """
        Get dividend transactions with optional filters.
        
        Args:
            asset_id: Filter by specific asset
            start_date: Filter from this date (inclusive)
            end_date: Filter to this date (inclusive)
            
        Returns:
            Sequence of dividend CashTransaction records
        """
        stmt = (
            select(CashTransaction)
            .where(CashTransaction.transaction_type == CashTransactionType.DIVIDEND)
        )
        
        if asset_id is not None:
            stmt = stmt.where(CashTransaction.asset_id == asset_id)
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        stmt = stmt.order_by(CashTransaction.transaction_date.desc())
        return self.session.scalars(stmt).all()
    
    def get_dividend_summary_by_asset(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[int, float]:
        """
        Get total dividends grouped by asset.
        
        Args:
            start_date: Filter from this date
            end_date: Filter to this date
            
        Returns:
            Dict mapping asset_id to total dividend amount
        """
        stmt = (
            select(
                CashTransaction.asset_id,
                func.sum(CashTransaction.amount).label("total"),
            )
            .where(
                CashTransaction.transaction_type == CashTransactionType.DIVIDEND,
                CashTransaction.asset_id.is_not(None),
            )
            .group_by(CashTransaction.asset_id)
        )
        
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        results = self.session.execute(stmt).all()
        return {r.asset_id: r.total for r in results}

    def record_trade_cash_flow(
        self,
        trade: Trade,
        fees: float = 0.0,
    ) -> list[CashTransaction]:
        """
        Record cash flow from a trade.
        
        BUY/COVER: Cash outflow = -(shares * price + fees)
        SELL/SHORT: Cash inflow = +(shares * price - fees)
        
        Args:
            trade: The Trade object
            fees: Trading fees (recorded separately)
            
        Returns:
            List of created CashTransactions (trade + optional fee)
        """
        transactions = []
        ticker = trade.asset.ticker if trade.asset else "?"
        gross = trade.shares * trade.price
        
        # Map trade action to cash transaction type
        type_map = {
            TradeAction.BUY: CashTransactionType.BUY,
            TradeAction.SELL: CashTransactionType.SELL,
            TradeAction.SHORT: CashTransactionType.SHORT,
            TradeAction.COVER: CashTransactionType.COVER,
        }
        
        tx_type = type_map[trade.action]
        
        # Calculate signed amount
        if trade.action in (TradeAction.BUY, TradeAction.COVER):
            # Cash outflow
            amount = -gross
            desc = f"{trade.action.value} {trade.shares} {ticker} @ ${trade.price:.2f}"
        else:
            # Cash inflow (SELL, SHORT)
            amount = +gross
            desc = f"{trade.action.value} {trade.shares} {ticker} @ ${trade.price:.2f}"
        
        # Record main trade cash flow
        tx = self.create_transaction(
            transaction_date=trade.trade_date,
            transaction_type=tx_type,
            amount=amount,
            trade_id=trade.id,
            asset_id=trade.asset_id,
            description=desc,
        )
        transactions.append(tx)
        
        # Record fees separately if present
        if fees > 0:
            fee_tx = self.create_transaction(
                transaction_date=trade.trade_date,
                transaction_type=CashTransactionType.FEE,
                amount=-fees,  # Fees are always outflow
                trade_id=trade.id,
                asset_id=trade.asset_id,
                description=f"Trading fee for {ticker}",
            )
            transactions.append(fee_tx)
        
        return transactions
    
    def get_all_transactions(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        transaction_type: CashTransactionType | None = None,
        limit: int | None = None,
    ) -> Sequence[CashTransaction]:
        """Get all cash transactions with optional filters."""
        stmt = select(CashTransaction)
        
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        if transaction_type:
            stmt = stmt.where(CashTransaction.transaction_type == transaction_type)
        
        stmt = stmt.order_by(CashTransaction.transaction_date.desc(), CashTransaction.id.desc())
        
        if limit:
            stmt = stmt.limit(limit)
        
        return self.session.scalars(stmt).all()
    
    def get_balance(
        self,
        as_of_date: str | None = None,
    ) -> float:
        """
        Calculate cash balance as of a given date.
        
        Args:
            as_of_date: Calculate balance up to this date (inclusive).
                       If None, uses all transactions.
                       
        Returns:
            Cash balance (sum of all signed amounts)
        """
        stmt = select(func.sum(CashTransaction.amount))
        
        if as_of_date:
            stmt = stmt.where(CashTransaction.transaction_date <= as_of_date)
        
        result = self.session.scalar(stmt)
        return result or 0.0
    
    def get_summary(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """
        Get cash flow summary for a period.
        
        Returns:
            Dict with total_inflows, total_outflows, net_flow, 
            starting_balance, ending_balance
        """
        # Calculate starting balance (before start_date)
        starting_balance = 0.0
        if start_date:
            starting_balance = self.get_balance(as_of_date=start_date)
            # Subtract transactions on start_date to get balance before
            stmt = select(func.sum(CashTransaction.amount)).where(
                CashTransaction.transaction_date == start_date
            )
            on_start = self.session.scalar(stmt) or 0.0
            starting_balance -= on_start
        
        # Get transactions in range
        stmt = select(CashTransaction.amount)
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        amounts = self.session.scalars(stmt).all()
        
        total_inflows = sum(a for a in amounts if a > 0)
        total_outflows = sum(abs(a) for a in amounts if a < 0)
        net_flow = sum(amounts)
        
        # Ending balance
        ending_balance = self.get_balance(as_of_date=end_date)
        
        return {
            "starting_balance": starting_balance,
            "ending_balance": ending_balance,
            "total_inflows": total_inflows,
            "total_outflows": total_outflows,
            "net_flow": net_flow,
        }
    
    def get_balance_by_type(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, float]:
        """
        Get cash flow breakdown by transaction type.
        
        Returns:
            Dict mapping transaction type to total amount
        """
        stmt = select(
            CashTransaction.transaction_type,
            func.sum(CashTransaction.amount).label("total"),
        ).group_by(CashTransaction.transaction_type)
        
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        results = self.session.execute(stmt).all()
        
        return {r.transaction_type.value: r.total or 0.0 for r in results}
    
    def get_ledger(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Get cash ledger with running balance.
        
        Returns list of dicts with date, type, amount, description, balance.
        """
        # Get starting balance before the period
        starting_balance = 0.0
        if start_date:
            stmt = select(func.sum(CashTransaction.amount)).where(
                CashTransaction.transaction_date < start_date
            )
            starting_balance = self.session.scalar(stmt) or 0.0
        
        # Get transactions in period (chronological order for running balance)
        stmt = select(CashTransaction)
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        stmt = stmt.order_by(CashTransaction.transaction_date, CashTransaction.id)
        
        transactions = self.session.scalars(stmt).all()
        
        # Build ledger with running balance
        ledger = []
        balance = starting_balance
        
        for tx in transactions:
            balance += tx.amount
            ledger.append({
                "date": tx.transaction_date,
                "type": tx.transaction_type.value,
                "amount": tx.amount,
                "description": tx.description or "",
                "balance": balance,
            })
        
        # If limit specified, return last N entries
        if limit and len(ledger) > limit:
            ledger = ledger[-limit:]
        
        return ledger

    def get_daily_balances(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, float]:
        """
        Get end-of-day cash balance for each date with transactions.
        
        For NAV calculation, forward-fill these balances to cover
        days without transactions.
        
        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)
            
        Returns:
            Dict mapping date string to EOD cash balance
        """
        # Get starting balance before the period
        starting_balance = 0.0
        if start_date:
            stmt = select(func.sum(CashTransaction.amount)).where(
                CashTransaction.transaction_date < start_date
            )
            starting_balance = self.session.scalar(stmt) or 0.0
        
        # Get daily net cash flows
        stmt = (
            select(
                CashTransaction.transaction_date,
                func.sum(CashTransaction.amount).label("daily_flow"),
            )
            .group_by(CashTransaction.transaction_date)
            .order_by(CashTransaction.transaction_date)
        )
        
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        results = self.session.execute(stmt).all()
        
        # Build cumulative balance by date
        balances = {}
        running = starting_balance
        
        for r in results:
            running += r.daily_flow
            balances[r.transaction_date] = running
        
        return balances

    def list_all_chronological(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Sequence[CashTransaction]:
        """
        Get all cash transactions in chronological order.
        
        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)
            
        Returns:
            CashTransactions sorted by date ASC, id ASC
        """
        stmt = select(CashTransaction)
        
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_date >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_date <= end_date)
        
        stmt = stmt.order_by(
            CashTransaction.transaction_date.asc(),
            CashTransaction.id.asc(),
        )
        return self.session.scalars(stmt).all()

