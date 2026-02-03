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
    AssetType,
    CashTransaction,
    CashTransactionType,
    IndexPriceDaily,
    MarketIndex,
    Note,
    NoteTarget,
    NoteTargetKind,
    NoteType,
    Position,
    PriceDaily,
    Tag,
    Trade,
    TradeAction,
    ValuationMetric,
    ValuationMetricOverride,
    asset_tags,
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
        stmt = select(Asset).options(joinedload(Asset.position)).where(Asset.id == asset_id)
        return self.session.scalar(stmt)

    def create(
        self,
        ticker: str,
        status: AssetStatus,
        asset_type: AssetType = AssetType.STOCK,
        name: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        exchange: str | None = None,
    ) -> Asset:
        """Create a new asset."""
        asset = Asset(
            ticker=ticker.upper(),
            status=status,
            asset_type=asset_type,
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

    def get_by_tickers(self, tickers: list[str]) -> Sequence[Asset]:
        """Get assets by list of tickers."""
        if not tickers:
            return []
        upper_tickers = [t.upper() for t in tickers]
        stmt = select(Asset).where(Asset.ticker.in_(upper_tickers))
        return self.session.scalars(stmt).all()

    def delete_by_id(self, asset_id: int) -> bool:
        """
        Delete an asset and all related data.

        Relies on ON DELETE CASCADE for:
        - prices_daily
        - fundamentals_quarterly
        - valuation_metrics
        - valuation_metric_overrides
        - watchlist_targets
        - investment_thesis
        - trades
        - positions
        - note_targets (and cascaded notes)

        Returns:
            True if deleted, False if asset not found.
        """
        asset = self.get_by_id(asset_id)
        if not asset:
            return False

        self.session.delete(asset)
        self.session.flush()
        return True

    def has_trades(self, asset_id: int) -> bool:
        """Check if an asset has any trade records."""
        stmt = select(func.count(Trade.id)).where(Trade.asset_id == asset_id)
        count = self.session.scalar(stmt) or 0
        return count > 0

    def has_position(self, asset_id: int) -> bool:
        """Check if an asset has an active position (non-zero shares)."""
        stmt = select(Position).where(
            Position.asset_id == asset_id, (Position.long_shares > 0) | (Position.short_shares > 0)
        )
        return self.session.scalar(stmt) is not None


class PriceRepository:
    """Repository for price data operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_latest_date(self, asset_id: int) -> str | None:
        """Get the most recent price date for an asset."""
        stmt = select(func.max(PriceDaily.date)).where(PriceDaily.asset_id == asset_id)
        return self.session.scalar(stmt)

    def get_latest_price(self, asset_id: int) -> PriceDaily | None:
        """Get the most recent price record for an asset."""
        latest_date = self.get_latest_date(asset_id)
        if not latest_date:
            return None

        stmt = select(PriceDaily).where(
            PriceDaily.asset_id == asset_id, PriceDaily.date == latest_date
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
        subq = select(
            PriceDaily.asset_id,
            func.max(PriceDaily.date).label("max_date"),
        ).group_by(PriceDaily.asset_id)

        if asset_ids:
            subq = subq.where(PriceDaily.asset_id.in_(asset_ids))

        subq = subq.subquery()

        # Main query joining with subquery
        stmt = select(PriceDaily).join(
            subq, (PriceDaily.asset_id == subq.c.asset_id) & (PriceDaily.date == subq.c.max_date)
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
        start_date: str | None = None,
    ) -> list[dict]:
        """
        Get price history for multiple assets (FR-5).

        Uses raw close prices for NAV/returns calculation.
        Dividends are tracked separately in the cash ledger.

        Args:
            asset_ids: Optional list of asset IDs to filter.
            status: Optional asset status to filter.
            start_date: Optional start date (YYYY-MM-DD) for filtering.

        Returns:
            List of dicts with [date, ticker, close].
        """
        stmt = (
            select(
                PriceDaily.date,
                PriceDaily.asset_id,
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
        if start_date:
            stmt = stmt.where(PriceDaily.date >= start_date)

        stmt = stmt.order_by(PriceDaily.date)

        results = self.session.execute(stmt).all()
        return [
            {
                "date": r.date,
                "asset_id": r.asset_id,
                "ticker": r.ticker,
                "close": r.close,
            }
            for r in results
        ]

    def get_price_history_for_tickers(
        self,
        tickers: list[str],
        start_date: str | None = None,
    ) -> list[dict]:
        """
        Get price history for specific tickers.

        Args:
            tickers: List of ticker symbols.
            start_date: Optional start date (YYYY-MM-DD) for filtering.

        Returns:
            List of dicts with [date, ticker, close].
        """
        if not tickers:
            return []

        upper_tickers = [t.upper() for t in tickers]
        stmt = (
            select(
                PriceDaily.date,
                PriceDaily.asset_id,
                Asset.ticker,
                PriceDaily.close,
            )
            .join(Asset, PriceDaily.asset_id == Asset.id)
            .where(PriceDaily.close.is_not(None))
            .where(Asset.ticker.in_(upper_tickers))
        )

        if start_date:
            stmt = stmt.where(PriceDaily.date >= start_date)

        stmt = stmt.order_by(PriceDaily.date)

        results = self.session.execute(stmt).all()
        return [
            {
                "date": r.date,
                "asset_id": r.asset_id,
                "ticker": r.ticker,
                "close": r.close,
            }
            for r in results
        ]

    def get_latest_and_prior_prices_for_assets(
        self,
        asset_ids: list[int] | None = None,
    ) -> dict[int, dict]:
        """
        Get latest and prior close prices for multiple assets.

        Used for calculating 1-day price change (today unrealized P&L).

        Args:
            asset_ids: Optional list of asset IDs to filter.

        Returns:
            Dict mapping asset_id to {"latest_close": float, "prior_close": float | None, "latest_date": str}.
            prior_close is None if only one price point exists.
        """
        # Get last 2 prices per asset ordered by date desc
        from sqlalchemy import literal_column
        from sqlalchemy.orm import aliased

        # Use window function to rank dates per asset
        # ROW_NUMBER() OVER (PARTITION BY asset_id ORDER BY date DESC)
        row_num = (
            func.row_number()
            .over(partition_by=PriceDaily.asset_id, order_by=PriceDaily.date.desc())
            .label("rn")
        )

        subq = select(
            PriceDaily.asset_id,
            PriceDaily.date,
            PriceDaily.close,
            row_num,
        )

        if asset_ids:
            subq = subq.where(PriceDaily.asset_id.in_(asset_ids))

        subq = subq.subquery()

        # Get rows where rn <= 2 (latest and prior)
        stmt = (
            select(
                subq.c.asset_id,
                subq.c.date,
                subq.c.close,
                subq.c.rn,
            )
            .where(subq.c.rn <= 2)
            .order_by(subq.c.asset_id, subq.c.rn)
        )

        results = self.session.execute(stmt).all()

        # Build result dict
        prices_by_asset: dict[int, dict] = {}
        for row in results:
            asset_id = row.asset_id
            if asset_id not in prices_by_asset:
                prices_by_asset[asset_id] = {
                    "latest_close": None,
                    "prior_close": None,
                    "latest_date": None,
                }

            if row.rn == 1:
                prices_by_asset[asset_id]["latest_close"] = row.close
                prices_by_asset[asset_id]["latest_date"] = row.date
            elif row.rn == 2:
                prices_by_asset[asset_id]["prior_close"] = row.close

        return prices_by_asset


class ValuationRepository:
    """Repository for valuation metrics operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_asset_id(self, asset_id: int) -> ValuationMetric | None:
        """Get valuation metrics for an asset."""
        return self.session.get(ValuationMetric, asset_id)

    def get_all_with_assets(self) -> Sequence[ValuationMetric]:
        """Get all valuation metrics with asset data."""
        stmt = select(ValuationMetric).options(joinedload(ValuationMetric.asset))
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
        # Valuation Measures
        market_cap: float | None = None,
        enterprise_value: float | None = None,
        pe_trailing: float | None = None,
        pe_forward: float | None = None,
        peg: float | None = None,
        price_to_sales: float | None = None,
        price_to_book: float | None = None,
        ev_to_revenue: float | None = None,
        ev_ebitda: float | None = None,
        # Financial Highlights - Profitability
        profit_margin: float | None = None,
        return_on_assets: float | None = None,
        return_on_equity: float | None = None,
        # Financial Highlights - Income Statement
        revenue_ttm: float | None = None,
        net_income_ttm: float | None = None,
        diluted_eps_ttm: float | None = None,
        # Financial Highlights - Balance Sheet & Cash Flow
        total_cash: float | None = None,
        total_debt_to_equity: float | None = None,
        levered_free_cash_flow: float | None = None,
        # Legacy fields
        revenue_growth: float | None = None,
        eps_growth: float | None = None,
    ) -> ValuationMetric:
        """
        Insert or update valuation metrics (FR-8, FR-9).

        Yahoo-aligned fields matching "Statistics" page.
        Missing values are stored as NULL, no synthetic values allowed.
        """
        existing = self.get_by_asset_id(asset_id)

        if existing:
            # Valuation Measures
            existing.market_cap = market_cap
            existing.enterprise_value = enterprise_value
            existing.pe_trailing = pe_trailing
            existing.pe_forward = pe_forward
            existing.peg = peg
            existing.price_to_sales = price_to_sales
            existing.price_to_book = price_to_book
            existing.ev_to_revenue = ev_to_revenue
            existing.ev_ebitda = ev_ebitda
            # Financial Highlights - Profitability
            existing.profit_margin = profit_margin
            existing.return_on_assets = return_on_assets
            existing.return_on_equity = return_on_equity
            # Financial Highlights - Income Statement
            existing.revenue_ttm = revenue_ttm
            existing.net_income_ttm = net_income_ttm
            existing.diluted_eps_ttm = diluted_eps_ttm
            # Financial Highlights - Balance Sheet & Cash Flow
            existing.total_cash = total_cash
            existing.total_debt_to_equity = total_debt_to_equity
            existing.levered_free_cash_flow = levered_free_cash_flow
            # Legacy fields
            existing.revenue_growth = revenue_growth
            existing.eps_growth = eps_growth
            existing.updated_at = datetime.utcnow()
            self.session.flush()
            return existing

        valuation = ValuationMetric(
            asset_id=asset_id,
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            pe_trailing=pe_trailing,
            pe_forward=pe_forward,
            peg=peg,
            price_to_sales=price_to_sales,
            price_to_book=price_to_book,
            ev_to_revenue=ev_to_revenue,
            ev_ebitda=ev_ebitda,
            profit_margin=profit_margin,
            return_on_assets=return_on_assets,
            return_on_equity=return_on_equity,
            revenue_ttm=revenue_ttm,
            net_income_ttm=net_income_ttm,
            diluted_eps_ttm=diluted_eps_ttm,
            total_cash=total_cash,
            total_debt_to_equity=total_debt_to_equity,
            levered_free_cash_flow=levered_free_cash_flow,
            revenue_growth=revenue_growth,
            eps_growth=eps_growth,
        )
        self.session.add(valuation)
        self.session.flush()
        return valuation


class ValuationOverrideRepository:
    """Repository for valuation metric override operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_asset_id(self, asset_id: int) -> ValuationMetricOverride | None:
        """Get override for a single asset."""
        return self.session.get(ValuationMetricOverride, asset_id)

    def get_by_asset_ids(self, asset_ids: list[int]) -> dict[int, ValuationMetricOverride]:
        """
        Get overrides for multiple assets.

        Returns:
            Dict mapping asset_id to ValuationMetricOverride.
        """
        if not asset_ids:
            return {}

        stmt = select(ValuationMetricOverride).where(
            ValuationMetricOverride.asset_id.in_(asset_ids)
        )
        overrides = self.session.scalars(stmt).all()
        return {o.asset_id: o for o in overrides}

    def get_all(self) -> Sequence[ValuationMetricOverride]:
        """Get all overrides."""
        stmt = select(ValuationMetricOverride)
        return self.session.scalars(stmt).all()

    def upsert(
        self,
        asset_id: int,
        # Valuation Measures overrides
        market_cap_override: float | None = None,
        enterprise_value_override: float | None = None,
        pe_trailing_override: float | None = None,
        pe_forward_override: float | None = None,
        peg_override: float | None = None,
        price_to_sales_override: float | None = None,
        price_to_book_override: float | None = None,
        ev_to_revenue_override: float | None = None,
        ev_ebitda_override: float | None = None,
        # Financial Highlights - Profitability overrides
        profit_margin_override: float | None = None,
        return_on_assets_override: float | None = None,
        return_on_equity_override: float | None = None,
        # Financial Highlights - Income Statement overrides
        revenue_ttm_override: float | None = None,
        net_income_ttm_override: float | None = None,
        diluted_eps_ttm_override: float | None = None,
        # Financial Highlights - Balance Sheet & Cash Flow overrides
        total_cash_override: float | None = None,
        total_debt_to_equity_override: float | None = None,
        levered_free_cash_flow_override: float | None = None,
        # Legacy fields
        revenue_growth_override: float | None = None,
        eps_growth_override: float | None = None,
    ) -> ValuationMetricOverride:
        """
        Insert or update valuation metric overrides.

        NULL values mean no override (use fetched value).
        Supports all Yahoo-aligned metrics.
        """
        existing = self.get_by_asset_id(asset_id)

        if existing:
            # Valuation Measures overrides
            existing.market_cap_override = market_cap_override
            existing.enterprise_value_override = enterprise_value_override
            existing.pe_trailing_override = pe_trailing_override
            existing.pe_forward_override = pe_forward_override
            existing.peg_override = peg_override
            existing.price_to_sales_override = price_to_sales_override
            existing.price_to_book_override = price_to_book_override
            existing.ev_to_revenue_override = ev_to_revenue_override
            existing.ev_ebitda_override = ev_ebitda_override
            # Financial Highlights - Profitability overrides
            existing.profit_margin_override = profit_margin_override
            existing.return_on_assets_override = return_on_assets_override
            existing.return_on_equity_override = return_on_equity_override
            # Financial Highlights - Income Statement overrides
            existing.revenue_ttm_override = revenue_ttm_override
            existing.net_income_ttm_override = net_income_ttm_override
            existing.diluted_eps_ttm_override = diluted_eps_ttm_override
            # Financial Highlights - Balance Sheet & Cash Flow overrides
            existing.total_cash_override = total_cash_override
            existing.total_debt_to_equity_override = total_debt_to_equity_override
            existing.levered_free_cash_flow_override = levered_free_cash_flow_override
            # Legacy fields
            existing.revenue_growth_override = revenue_growth_override
            existing.eps_growth_override = eps_growth_override
            existing.updated_at = datetime.utcnow()
            self.session.flush()
            return existing

        override = ValuationMetricOverride(
            asset_id=asset_id,
            market_cap_override=market_cap_override,
            enterprise_value_override=enterprise_value_override,
            pe_trailing_override=pe_trailing_override,
            pe_forward_override=pe_forward_override,
            peg_override=peg_override,
            price_to_sales_override=price_to_sales_override,
            price_to_book_override=price_to_book_override,
            ev_to_revenue_override=ev_to_revenue_override,
            ev_ebitda_override=ev_ebitda_override,
            profit_margin_override=profit_margin_override,
            return_on_assets_override=return_on_assets_override,
            return_on_equity_override=return_on_equity_override,
            revenue_ttm_override=revenue_ttm_override,
            net_income_ttm_override=net_income_ttm_override,
            diluted_eps_ttm_override=diluted_eps_ttm_override,
            total_cash_override=total_cash_override,
            total_debt_to_equity_override=total_debt_to_equity_override,
            levered_free_cash_flow_override=levered_free_cash_flow_override,
            revenue_growth_override=revenue_growth_override,
            eps_growth_override=eps_growth_override,
        )
        self.session.add(override)
        self.session.flush()
        return override

    def delete(self, asset_id: int) -> bool:
        """Delete override for an asset."""
        override = self.get_by_asset_id(asset_id)
        if override:
            self.session.delete(override)
            self.session.flush()
            return True
        return False


class TradeRepository:
    """Repository for trade operations."""

    def __init__(self, session: Session):
        self.session = session

    def create_trade(
        self,
        asset_id: int,
        trade_at: datetime,
        action: TradeAction,
        shares: float,
        price: float,
        fees: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> Trade:
        """Create a new trade record."""
        trade = Trade(
            asset_id=asset_id,
            trade_at=trade_at,
            action=action,
            shares=shares,
            price=price,
            fees=fees,
            realized_pnl=realized_pnl,
        )
        self.session.add(trade)
        self.session.flush()
        return trade

    def get_by_id(self, trade_id: int) -> Trade | None:
        """Get trade by ID."""
        return self.session.get(Trade, trade_id)

    def get_trades_for_asset(
        self,
        asset_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Sequence[Trade]:
        """Get trades for an asset within optional date range."""
        stmt = select(Trade).where(Trade.asset_id == asset_id)

        if start_date:
            stmt = stmt.where(Trade.trade_at >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_at <= end_date)

        stmt = stmt.order_by(Trade.trade_at.desc(), Trade.id.desc())
        return self.session.scalars(stmt).all()

    def get_latest_trade_for_asset(self, asset_id: int) -> Trade | None:
        """Get the most recent trade for an asset."""
        stmt = (
            select(Trade)
            .where(Trade.asset_id == asset_id)
            .order_by(Trade.trade_at.desc(), Trade.id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def get_latest_trade_ids_by_ticker(self) -> dict[str, int]:
        """
        Get the latest trade ID for each ticker.

        Used to determine which trades are editable (only latest per ticker).

        Returns:
            Dict mapping ticker to latest trade_id
        """
        # Subquery to get max trade_at per asset
        from sqlalchemy import and_

        subq = (
            select(
                Trade.asset_id,
                func.max(Trade.trade_at).label("max_date"),
            )
            .group_by(Trade.asset_id)
            .subquery()
        )

        # For each asset, get the trade with max date (and max id for ties)
        # This requires a second level of filtering
        stmt = (
            select(Trade.id, Asset.ticker)
            .join(Asset, Trade.asset_id == Asset.id)
            .join(subq, and_(Trade.asset_id == subq.c.asset_id, Trade.trade_at == subq.c.max_date))
            .order_by(Trade.asset_id, Trade.id.desc())
        )

        results = self.session.execute(stmt).all()

        # Keep only the highest id per ticker (for same-datetime trades)
        latest_by_ticker: dict[str, int] = {}
        for trade_id, ticker in results:
            if ticker not in latest_by_ticker:
                latest_by_ticker[ticker] = trade_id

        return latest_by_ticker

    def get_all_trades(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        action: TradeAction | None = None,
        limit: int | None = None,
    ) -> Sequence[Trade]:
        """Get all trades with optional filters."""
        stmt = select(Trade).options(joinedload(Trade.asset))

        if start_date:
            stmt = stmt.where(Trade.trade_at >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_at <= end_date)
        if action:
            stmt = stmt.where(Trade.action == action)

        stmt = stmt.order_by(Trade.trade_at.desc(), Trade.id.desc())

        if limit:
            stmt = stmt.limit(limit)

        return self.session.scalars(stmt).all()

    def get_realized_pnl_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, float]:
        """Get realized P&L summary across all trades."""
        stmt = select(
            func.sum(Trade.realized_pnl).label("total_realized"),
            func.sum(func.abs(Trade.fees)).label("total_fees"),
        )

        if start_date:
            stmt = stmt.where(Trade.trade_at >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_at <= end_date)

        result = self.session.execute(stmt).first()

        return {
            "total_realized_pnl": result.total_realized or 0.0,
            "total_fees": result.total_fees or 0.0,
            "net_realized_pnl": (result.total_realized or 0.0) - (result.total_fees or 0.0),
        }

    def update_trade(
        self,
        trade_id: int,
        shares: float | None = None,
        price: float | None = None,
        fees: float | None = None,
        trade_at: str | datetime | None = None,
    ) -> Trade | None:
        """
        Update an existing trade's editable fields.

        Note: Does NOT recalculate realized_pnl - caller must handle reconciliation.

        Args:
            trade_id: ID of trade to update
            shares: New share count (if provided)
            price: New price per share (if provided)
            fees: New fee amount (if provided)
            trade_at: New trade date (if provided)

        Returns:
            Updated Trade or None if not found
        """
        trade = self.get_by_id(trade_id)
        if not trade:
            return None

        if shares is not None:
            trade.shares = shares
        if price is not None:
            trade.price = price
        if fees is not None:
            trade.fees = fees
        if trade_at is not None:
            trade.trade_at = trade_at

        self.session.flush()
        return trade

    def delete_trade(self, trade_id: int) -> bool:
        """
        Delete a trade by ID.

        Note: Does NOT handle cascading effects - caller must handle reconciliation.

        Returns:
            True if deleted, False if not found
        """
        trade = self.get_by_id(trade_id)
        if not trade:
            return False

        self.session.delete(trade)
        self.session.flush()
        return True

    def list_all_chronological(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
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
            stmt = stmt.where(Trade.trade_at >= start_date)
        if end_date:
            stmt = stmt.where(Trade.trade_at <= end_date)

        stmt = stmt.order_by(Trade.trade_at.asc(), Trade.id.asc())
        return self.session.scalars(stmt).all()

    def list_trades_for_asset_chronological(self, asset_id: int) -> Sequence[Trade]:
        """
        Get all trades for an asset in chronological order.

        Used for position reconstruction after trade edit/delete.

        Returns:
            Trades sorted by trade_at ASC, id ASC
        """
        stmt = (
            select(Trade)
            .where(Trade.asset_id == asset_id)
            .order_by(Trade.trade_at.asc(), Trade.id.asc())
        )
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
            .where((Position.long_shares > 0) | (Position.short_shares > 0))
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
                Asset.asset_type,
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
                (Position.long_shares > 0) | (Position.short_shares > 0),
            )
        )

        results = self.session.execute(stmt).all()
        return [
            {
                "asset_id": r.asset_id,
                "ticker": r.ticker,
                "asset_type": r.asset_type,
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
        stmt = select(Trade).where(Trade.asset_id == asset_id).order_by(Trade.trade_at, Trade.id)
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
        stmt = select(Position.asset_id).where(
            (Position.long_shares > 0) | (Position.short_shares > 0)
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
        stmt = select(Position.asset_id, Position.net_invested).where(
            (Position.long_shares > 0) | (Position.short_shares > 0)
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
        transaction_at: datetime,
        transaction_type: CashTransactionType,
        amount: float,
        trade_id: int | None = None,
        asset_id: int | None = None,
        description: str | None = None,
    ) -> CashTransaction:
        """
        Create a new cash transaction.

        Args:
            transaction_at: Date of transaction
            transaction_type: Type of transaction
            amount: Signed amount (+ inflow, - outflow)
            trade_id: Optional reference to related trade
            asset_id: Optional reference to related asset
            description: Optional description

        Returns:
            Created CashTransaction
        """
        transaction = CashTransaction(
            transaction_at=transaction_at,
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
        transaction_at: datetime | None = None,
        description: str | None = None,
    ) -> CashTransaction:
        """
        Record a cash deposit (capital injection).

        Args:
            amount: Deposit amount (positive)
            transaction_at: Date (defaults to today)a
            description: Optional description

        Returns:
            Created CashTransaction
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        if transaction_at is None:
            transaction_at = datetime.now()

        return self.create_transaction(
            transaction_at=transaction_at,
            transaction_type=CashTransactionType.DEPOSIT,
            amount=amount,
            description=description or "Cash deposit",
        )

    def withdraw(
        self,
        amount: float,
        transaction_at: datetime | None = None,
        description: str | None = None,
    ) -> CashTransaction:
        """
        Record a cash withdrawal.

        Args:
            amount: Withdrawal amount (positive, will be stored as negative)
            transaction_at: Date (defaults to today)
            description: Optional description

        Returns:
            Created CashTransaction
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        if transaction_at is None:
            transaction_at = datetime.now()

        return self.create_transaction(
            transaction_at=transaction_at,
            transaction_type=CashTransactionType.WITHDRAW,
            amount=-amount,  # Store as negative (outflow)
            description=description or "Cash withdrawal",
        )

    def record_dividend(
        self,
        asset_id: int,
        amount: float,
        transaction_at: datetime,
        description: str | None = None,
    ) -> tuple[CashTransaction, bool]:
        """
        Record a dividend payment with idempotency.

        Uses deduplication strategy: asset_id + transaction_at + amount
        to prevent duplicate dividend entries on reruns.

        Args:
            asset_id: Asset ID for dividend attribution
            amount: Dividend amount (positive, cash inflow)
            transaction_at: Ex-dividend or payment date (YYYY-MM-DD)
            description: Optional description

        Returns:
            Tuple of (CashTransaction, created) where created=False for duplicates

        Raises:
            ValueError: If amount is not positive
        """
        if amount <= 0:
            raise ValueError("Dividend amount must be positive")

        # Check for duplicate
        existing = self.find_dividend(asset_id, amount, transaction_at)
        if existing:
            return existing, False

        # Create new dividend transaction
        transaction = CashTransaction(
            transaction_at=transaction_at,
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
        transaction_at: datetime,
    ) -> CashTransaction | None:
        """
        Find existing dividend matching the dedup key.

        Dedup key: asset_id + transaction_at + amount

        Args:
            asset_id: Asset ID
            amount: Dividend amount
            transaction_at: Transaction date

        Returns:
            Existing CashTransaction if found, else None
        """
        stmt = select(CashTransaction).where(
            CashTransaction.asset_id == asset_id,
            CashTransaction.transaction_at == transaction_at,
            CashTransaction.transaction_type == CashTransactionType.DIVIDEND,
            CashTransaction.amount == amount,
        )
        return self.session.scalar(stmt)

    def get_dividends(
        self,
        asset_id: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
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
        stmt = select(CashTransaction).where(
            CashTransaction.transaction_type == CashTransactionType.DIVIDEND
        )

        if asset_id is not None:
            stmt = stmt.where(CashTransaction.asset_id == asset_id)
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

        stmt = stmt.order_by(CashTransaction.transaction_at.desc())
        return self.session.scalars(stmt).all()

    def get_dividend_summary_by_asset(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
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
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

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
            transaction_at=trade.trade_at,
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
                transaction_at=trade.trade_at,
                transaction_type=CashTransactionType.FEE,
                amount=-fees,  # Fees are always outflow
                trade_id=trade.id,
                asset_id=trade.asset_id,
                description=f"Trading fee for {ticker}",
            )
            transactions.append(fee_tx)

        return transactions

    def delete_transactions_by_trade_id(self, trade_id: int) -> int:
        """
        Delete all cash transactions linked to a specific trade.

        Used during trade reconciliation to remove and regenerate cash flows.

        Args:
            trade_id: ID of the trade whose cash transactions should be deleted

        Returns:
            Number of transactions deleted
        """
        stmt = select(CashTransaction).where(CashTransaction.trade_id == trade_id)
        transactions = self.session.scalars(stmt).all()
        count = len(transactions)

        for tx in transactions:
            self.session.delete(tx)

        self.session.flush()
        return count

    def get_all_transactions(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        transaction_type: CashTransactionType | None = None,
        limit: int | None = None,
    ) -> Sequence[CashTransaction]:
        """Get all cash transactions with optional filters."""
        stmt = select(CashTransaction)

        if start_date:
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)
        if transaction_type:
            stmt = stmt.where(CashTransaction.transaction_type == transaction_type)

        stmt = stmt.order_by(CashTransaction.transaction_at.desc(), CashTransaction.id.desc())

        if limit:
            stmt = stmt.limit(limit)

        return self.session.scalars(stmt).all()

    def get_balance(
        self,
        as_of_date: datetime | None = None,
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
            stmt = stmt.where(CashTransaction.transaction_at <= as_of_date)

        result = self.session.scalar(stmt)
        return result or 0.0

    def get_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
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
                CashTransaction.transaction_at == start_date
            )
            on_start = self.session.scalar(stmt) or 0.0
            starting_balance -= on_start

        # Get transactions in range
        stmt = select(CashTransaction.amount)
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

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
        start_date: datetime | None = None,
        end_date: datetime | None = None,
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
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

        results = self.session.execute(stmt).all()

        return {r.transaction_type.value: r.total or 0.0 for r in results}

    def get_ledger(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
        sort_desc: bool = True,
    ) -> list[dict]:
        """
        Get cash ledger with running balance.

        Returns list of dicts with date, type, amount, description, balance.
        """
        # Get starting balance before the period
        starting_balance = 0.0
        if start_date:
            stmt = select(func.sum(CashTransaction.amount)).where(
                CashTransaction.transaction_at < start_date
            )
            starting_balance = self.session.scalar(stmt) or 0.0

        # Get transactions in period (chronological order for running balance)
        stmt = select(CashTransaction)
        if start_date:
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

        if sort_desc:
            stmt = stmt.order_by(
                CashTransaction.transaction_at.desc(),
            )
        else:
            stmt = stmt.order_by(
                CashTransaction.transaction_at.asc(),
            )

        transactions = self.session.scalars(stmt).all()

        # Build ledger with running balance
        ledger = []
        balance = starting_balance

        for tx in transactions:
            balance += tx.amount
            ledger.append(
                {
                    "date": tx.transaction_at,
                    "type": tx.transaction_type.value,
                    "amount": tx.amount,
                    "description": tx.description or "",
                    "balance": balance,
                }
            )

        # If limit specified, return last N entries
        if limit and len(ledger) > limit:
            ledger = ledger[-limit:]

        return ledger

    def get_daily_balances(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[datetime, float]:
        """
        Get end-of-day cash balance for each date with transactions.

        For NAV calculation, forward-fill these balances to cover
        days without transactions.

        Args:
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            Dict mapping datetime to EOD cash balance
        """
        # Get starting balance before the period
        starting_balance = 0.0
        if start_date:
            stmt = select(func.sum(CashTransaction.amount)).where(
                CashTransaction.transaction_at < start_date
            )
            starting_balance = self.session.scalar(stmt) or 0.0

        # Get daily net cash flows
        stmt = (
            select(
                CashTransaction.transaction_at,
                func.sum(CashTransaction.amount).label("daily_flow"),
            )
            .group_by(CashTransaction.transaction_at)
            .order_by(CashTransaction.transaction_at)
        )

        if start_date:
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

        results = self.session.execute(stmt).all()

        # Build cumulative balance by date
        balances = {}
        running = starting_balance

        for r in results:
            running += r.daily_flow
            balances[r.transaction_at] = running

        return balances

    def list_all_chronological(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
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
            stmt = stmt.where(CashTransaction.transaction_at >= start_date)
        if end_date:
            stmt = stmt.where(CashTransaction.transaction_at <= end_date)

        stmt = stmt.order_by(
            CashTransaction.transaction_at.asc(),
            CashTransaction.id.asc(),
        )
        return self.session.scalars(stmt).all()


class NoteTargetRepository:
    """Repository for NoteTarget operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, target_id: int) -> NoteTarget | None:
        """Get target by ID."""
        return self.session.get(NoteTarget, target_id)

    def get_or_create_asset_target(self, asset_id: int) -> NoteTarget:
        """Get or create a target for an asset."""
        stmt = select(NoteTarget).where(
            NoteTarget.kind == NoteTargetKind.ASSET,
            NoteTarget.asset_id == asset_id,
        )
        target = self.session.scalar(stmt)
        if target:
            return target

        target = NoteTarget(kind=NoteTargetKind.ASSET, asset_id=asset_id)
        self.session.add(target)
        self.session.flush()
        return target

    def get_or_create_trade_target(self, trade_id: int) -> NoteTarget:
        """Get or create a target for a trade."""
        stmt = select(NoteTarget).where(
            NoteTarget.kind == NoteTargetKind.TRADE,
            NoteTarget.trade_id == trade_id,
        )
        target = self.session.scalar(stmt)
        if target:
            return target

        target = NoteTarget(kind=NoteTargetKind.TRADE, trade_id=trade_id)
        self.session.add(target)
        self.session.flush()
        return target

    def get_or_create_market_target(self, symbol: str, name: str | None = None) -> NoteTarget:
        """Get or create a target for a market/index symbol."""
        symbol = symbol.upper()
        stmt = select(NoteTarget).where(
            NoteTarget.kind == NoteTargetKind.MARKET,
            NoteTarget.symbol == symbol,
        )
        target = self.session.scalar(stmt)
        if target:
            # Update name if provided
            if name and not target.symbol_name:
                target.symbol_name = name
                self.session.flush()
            return target

        target = NoteTarget(
            kind=NoteTargetKind.MARKET,
            symbol=symbol,
            symbol_name=name,
        )
        self.session.add(target)
        self.session.flush()
        return target

    def get_or_create_journal_target(self) -> NoteTarget:
        """Get or create a journal target (singleton for general entries)."""
        stmt = select(NoteTarget).where(NoteTarget.kind == NoteTargetKind.JOURNAL)
        target = self.session.scalar(stmt)
        if target:
            return target

        target = NoteTarget(kind=NoteTargetKind.JOURNAL)
        self.session.add(target)
        self.session.flush()
        return target

    def get_asset_target(self, asset_id: int) -> NoteTarget | None:
        """Get target for an asset if exists."""
        stmt = select(NoteTarget).where(
            NoteTarget.kind == NoteTargetKind.ASSET,
            NoteTarget.asset_id == asset_id,
        )
        return self.session.scalar(stmt)

    def list_market_targets(self) -> Sequence[NoteTarget]:
        """List all market targets."""
        stmt = (
            select(NoteTarget)
            .where(NoteTarget.kind == NoteTargetKind.MARKET)
            .order_by(NoteTarget.symbol)
        )
        return self.session.scalars(stmt).all()


class NoteRepository:
    """Repository for Note operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, note_id: int) -> Note | None:
        """Get note by ID."""
        return self.session.get(Note, note_id)

    def create(
        self,
        target_id: int,
        body_md: str,
        note_type: NoteType = NoteType.JOURNAL,
        title: str | None = None,
        summary: str | None = None,
        key_points: str | None = None,
        tags: str | None = None,
    ) -> Note:
        """Create a new note."""
        note = Note(
            target_id=target_id,
            note_type=note_type,
            title=title,
            summary=summary,
            key_points=key_points,
            body_md=body_md,
            tags=tags,
        )
        self.session.add(note)
        self.session.flush()
        return note

    def update(
        self,
        note_id: int,
        body_md: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        key_points: str | None = None,
        tags: str | None = None,
        note_type: NoteType | None = None,
    ) -> Note | None:
        """Update an existing note."""
        note = self.get_by_id(note_id)
        if not note:
            return None

        if body_md is not None:
            note.body_md = body_md
        if title is not None:
            note.title = title
        if summary is not None:
            note.summary = summary
        if key_points is not None:
            note.key_points = key_points
        if tags is not None:
            note.tags = tags
        if note_type is not None:
            note.note_type = note_type

        self.session.flush()
        return note

    def list_by_target(
        self,
        target_id: int,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> Sequence[Note]:
        """List notes for a target, ordered by pinned then created_at desc."""
        stmt = (
            select(Note)
            .options(joinedload(Note.target).joinedload(NoteTarget.asset))
            .where(Note.target_id == target_id)
        )

        if not include_archived:
            stmt = stmt.where(Note.status == "ACTIVE")

        # Pinned first, then by created_at desc
        stmt = stmt.order_by(Note.pinned.desc(), Note.created_at.desc())

        if limit:
            stmt = stmt.limit(limit)

        return self.session.scalars(stmt).all()

    def list_by_asset(
        self,
        asset_id: int,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> Sequence[Note]:
        """List notes for an asset (via target)."""
        stmt = (
            select(Note)
            .options(joinedload(Note.target).joinedload(NoteTarget.asset))
            .join(NoteTarget)
            .where(
                NoteTarget.kind == NoteTargetKind.ASSET,
                NoteTarget.asset_id == asset_id,
            )
        )

        if not include_archived:
            stmt = stmt.where(Note.status == "ACTIVE")

        stmt = stmt.order_by(Note.pinned.desc(), Note.created_at.desc())

        if limit:
            stmt = stmt.limit(limit)

        return self.session.scalars(stmt).all()

    def list_by_type(
        self,
        note_type: NoteType,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> Sequence[Note]:
        """List notes by type."""
        stmt = (
            select(Note)
            .options(joinedload(Note.target).joinedload(NoteTarget.asset))
            .where(Note.note_type == note_type)
        )

        if not include_archived:
            stmt = stmt.where(Note.status == "ACTIVE")

        stmt = stmt.order_by(Note.pinned.desc(), Note.created_at.desc())

        if limit:
            stmt = stmt.limit(limit)

        return self.session.scalars(stmt).all()

    def list_recent(
        self,
        limit: int = 20,
        include_archived: bool = False,
    ) -> Sequence[Note]:
        """List recent notes across all targets."""
        stmt = select(Note).options(joinedload(Note.target).joinedload(NoteTarget.asset))

        if not include_archived:
            stmt = stmt.where(Note.status == "ACTIVE")

        stmt = stmt.order_by(Note.created_at.desc()).limit(limit)

        return self.session.scalars(stmt).all()

    def search_by_tag(
        self,
        tag: str,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> Sequence[Note]:
        """Search notes containing a tag."""
        # SQLite LIKE for comma-separated tags
        pattern = f"%{tag}%"
        stmt = (
            select(Note)
            .options(joinedload(Note.target).joinedload(NoteTarget.asset))
            .where(Note.tags.like(pattern))
        )

        if not include_archived:
            stmt = stmt.where(Note.status == "ACTIVE")

        stmt = stmt.order_by(Note.created_at.desc())

        if limit:
            stmt = stmt.limit(limit)

        return self.session.scalars(stmt).all()

    def set_status(self, note_id: int, status: str) -> Note | None:
        """Set note status (ACTIVE, ARCHIVED, DELETED)."""
        note = self.get_by_id(note_id)
        if not note:
            return None
        note.status = status
        self.session.flush()
        return note

    def set_pinned(self, note_id: int, pinned: bool) -> Note | None:
        """Set pinned status."""
        note = self.get_by_id(note_id)
        if not note:
            return None
        note.pinned = pinned
        self.session.flush()
        return note

    def delete(self, note_id: int) -> bool:
        """Soft delete a note."""
        note = self.get_by_id(note_id)
        if not note:
            return False
        note.status = "DELETED"
        self.session.flush()
        return True

    def hard_delete(self, note_id: int) -> bool:
        """Permanently delete a note."""
        note = self.get_by_id(note_id)
        if not note:
            return False
        self.session.delete(note)
        self.session.flush()
        return True


class MarketIndexRepository:
    """Repository for MarketIndex operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, index_id: int) -> "MarketIndex | None":
        """Get market index by ID."""
        from db.models import MarketIndex

        return self.session.get(MarketIndex, index_id)

    def get_by_symbol(self, symbol: str) -> "MarketIndex | None":
        """Get market index by symbol."""
        from db.models import MarketIndex

        stmt = select(MarketIndex).where(MarketIndex.symbol == symbol.upper())
        return self.session.scalar(stmt)

    def get_all(self) -> "Sequence[MarketIndex]":
        """Get all market indices."""
        from db.models import MarketIndex

        stmt = select(MarketIndex).order_by(MarketIndex.symbol)
        return self.session.scalars(stmt).all()

    def get_by_category(self, category: str) -> "Sequence[MarketIndex]":
        """Get indices filtered by category."""
        from db.models import MarketIndex, IndexCategory

        stmt = (
            select(MarketIndex).where(MarketIndex.category == category).order_by(MarketIndex.symbol)
        )
        return self.session.scalars(stmt).all()

    def create(
        self,
        symbol: str,
        name: str,
        description: str | None = None,
        category: str = "EQUITY",
    ) -> "MarketIndex":
        """Create a new market index."""
        from db.models import MarketIndex, IndexCategory

        index = MarketIndex(
            symbol=symbol.upper(),
            name=name,
            description=description,
            category=IndexCategory(category),
        )
        self.session.add(index)
        self.session.flush()
        return index

    def get_or_create(
        self,
        symbol: str,
        name: str,
        **kwargs,
    ) -> tuple["MarketIndex", bool]:
        """
        Get existing index or create new one.

        Returns:
            Tuple of (index, created) where created is True if new.
        """
        index = self.get_by_symbol(symbol)
        if index:
            return index, False
        return self.create(symbol=symbol, name=name, **kwargs), True


class IndexPriceRepository:
    """Repository for index price data operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_latest_date(self, index_id: int) -> str | None:
        """Get the most recent price date for an index."""
        from db.models import IndexPriceDaily

        stmt = select(func.max(IndexPriceDaily.date)).where(IndexPriceDaily.index_id == index_id)
        return self.session.scalar(stmt)

    def get_latest_price(self, index_id: int) -> "IndexPriceDaily | None":
        """Get the most recent price record for an index."""
        from db.models import IndexPriceDaily

        latest_date = self.get_latest_date(index_id)
        if not latest_date:
            return None

        stmt = select(IndexPriceDaily).where(
            IndexPriceDaily.index_id == index_id, IndexPriceDaily.date == latest_date
        )
        return self.session.scalar(stmt)

    def get_price_history(
        self,
        index_id: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "Sequence[IndexPriceDaily]":
        """Get price history for an index within date range."""
        from db.models import IndexPriceDaily

        stmt = select(IndexPriceDaily).where(IndexPriceDaily.index_id == index_id)

        if start_date:
            stmt = stmt.where(IndexPriceDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(IndexPriceDaily.date <= end_date)

        stmt = stmt.order_by(IndexPriceDaily.date)
        return self.session.scalars(stmt).all()

    def get_price_history_for_indices(
        self,
        index_ids: list[int] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """
        Get price history for multiple indices.

        Returns:
            List of dicts with [date, index_id, symbol, close].
        """
        from db.models import IndexPriceDaily, MarketIndex

        stmt = (
            select(
                IndexPriceDaily.date,
                IndexPriceDaily.index_id,
                MarketIndex.symbol,
                IndexPriceDaily.close,
            )
            .join(MarketIndex, IndexPriceDaily.index_id == MarketIndex.id)
            .where(IndexPriceDaily.close.is_not(None))
        )

        if index_ids:
            stmt = stmt.where(IndexPriceDaily.index_id.in_(index_ids))
        if start_date:
            stmt = stmt.where(IndexPriceDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(IndexPriceDaily.date <= end_date)

        stmt = stmt.order_by(IndexPriceDaily.date)

        results = self.session.execute(stmt).all()
        return [
            {
                "date": r.date,
                "index_id": r.index_id,
                "symbol": r.symbol,
                "close": r.close,
            }
            for r in results
        ]

    def bulk_upsert_prices(
        self,
        index_id: int,
        price_records: list[dict],
    ) -> int:
        """
        Bulk insert/update price records (idempotent).

        Args:
            index_id: Index ID for the prices.
            price_records: List of dicts with date, open, high, low, close, etc.

        Returns:
            Number of records inserted.
        """
        from db.models import IndexPriceDaily

        count = 0
        for record in price_records:
            # Check if exists
            stmt = select(IndexPriceDaily).where(
                IndexPriceDaily.index_id == index_id,
                IndexPriceDaily.date == record["date"],
            )
            existing = self.session.scalar(stmt)

            if not existing:
                price = IndexPriceDaily(
                    index_id=index_id,
                    date=record["date"],
                    open=record.get("open"),
                    high=record.get("high"),
                    low=record.get("low"),
                    close=record.get("close"),
                    volume=record.get("volume"),
                )
                self.session.add(price)
                count += 1

        self.session.flush()
        return count

    def get_latest_prices_for_indices(
        self,
        index_ids: list[int] | None = None,
    ) -> dict[int, "IndexPriceDaily"]:
        """
        Get latest price for multiple indices.

        Returns:
            Dict mapping index_id to latest IndexPriceDaily.
        """
        from db.models import IndexPriceDaily

        # Subquery to get max date per index
        subq = select(
            IndexPriceDaily.index_id,
            func.max(IndexPriceDaily.date).label("max_date"),
        ).group_by(IndexPriceDaily.index_id)

        if index_ids:
            subq = subq.where(IndexPriceDaily.index_id.in_(index_ids))

        subq = subq.subquery()

        # Main query joining with subquery
        stmt = select(IndexPriceDaily).join(
            subq,
            (IndexPriceDaily.index_id == subq.c.index_id)
            & (IndexPriceDaily.date == subq.c.max_date),
        )

        prices = self.session.scalars(stmt).all()
        return {p.index_id: p for p in prices}


class TagRepository:
    """
    Repository for Tag CRUD and Asset-Tag association operations.
    
    Supports:
    - Tag CRUD (create, read, update/rename, delete)
    - Attach/detach tags to/from assets
    - Query assets by tags (OR semantics)
    - Query untagged assets
    
    Tag names are case-insensitive unique (enforced at service layer).
    """

    def __init__(self, session: Session):
        self.session = session

    # ──────────────────────────────────────────────────────────────────────────
    # Tag CRUD
    # ──────────────────────────────────────────────────────────────────────────

    def get_by_id(self, tag_id: int) -> Tag | None:
        """Get tag by ID."""
        return self.session.get(Tag, tag_id)

    def get_by_name(self, name: str) -> Tag | None:
        """Get tag by name (case-insensitive)."""
        stmt = select(Tag).where(func.lower(Tag.name) == name.lower())
        return self.session.scalar(stmt)

    def get_all(self) -> Sequence[Tag]:
        """Get all tags ordered by name."""
        stmt = select(Tag).order_by(Tag.name)
        return self.session.scalars(stmt).all()

    def get_all_with_asset_counts(self) -> list[tuple[Tag, int]]:
        """
        Get all tags with count of associated assets.
        
        Returns:
            List of (Tag, asset_count) tuples ordered by name.
        """
        stmt = (
            select(Tag, func.count(asset_tags.c.asset_id).label("asset_count"))
            .outerjoin(asset_tags, Tag.id == asset_tags.c.tag_id)
            .group_by(Tag.id)
            .order_by(Tag.name)
        )
        result = self.session.execute(stmt).all()
        return [(row[0], row[1]) for row in result]

    def create(self, name: str, description: str | None = None) -> Tag:
        """
        Create a new tag.
        
        Args:
            name: Tag name (will be stripped, uniqueness enforced at service layer)
            description: Optional description
            
        Returns:
            The created Tag.
        """
        tag = Tag(name=name.strip(), description=description)
        self.session.add(tag)
        self.session.flush()
        return tag

    def get_or_create(self, name: str, description: str | None = None) -> tuple[Tag, bool]:
        """
        Get existing tag by name or create new one.
        
        Args:
            name: Tag name (case-insensitive match)
            description: Optional description (only used if creating)
            
        Returns:
            Tuple of (tag, created) where created is True if new.
        """
        existing = self.get_by_name(name)
        if existing:
            return existing, False
        return self.create(name, description), True

    def rename(self, tag_id: int, new_name: str) -> Tag | None:
        """
        Rename a tag.
        
        Args:
            tag_id: ID of tag to rename
            new_name: New name (uniqueness enforced at service layer)
            
        Returns:
            Updated Tag or None if not found.
        """
        tag = self.get_by_id(tag_id)
        if tag:
            tag.name = new_name.strip()
            self.session.flush()
        return tag

    def update_description(self, tag_id: int, description: str | None) -> Tag | None:
        """Update tag description."""
        tag = self.get_by_id(tag_id)
        if tag:
            tag.description = description
            self.session.flush()
        return tag

    def delete(self, tag_id: int) -> bool:
        """
        Delete a tag (hard delete, cascades to asset_tags).
        
        Returns:
            True if deleted, False if not found.
        """
        tag = self.get_by_id(tag_id)
        if not tag:
            return False
        self.session.delete(tag)
        self.session.flush()
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Asset-Tag Association
    # ──────────────────────────────────────────────────────────────────────────

    def get_tags_for_asset(self, asset_id: int) -> Sequence[Tag]:
        """Get all tags attached to an asset."""
        stmt = (
            select(Tag)
            .join(asset_tags, Tag.id == asset_tags.c.tag_id)
            .where(asset_tags.c.asset_id == asset_id)
            .order_by(Tag.name)
        )
        return self.session.scalars(stmt).all()

    def get_assets_for_tag(self, tag_id: int) -> Sequence[Asset]:
        """Get all assets with a specific tag."""
        stmt = (
            select(Asset)
            .join(asset_tags, Asset.id == asset_tags.c.asset_id)
            .where(asset_tags.c.tag_id == tag_id)
            .order_by(Asset.ticker)
        )
        return self.session.scalars(stmt).all()

    def attach_tag(self, asset_id: int, tag_id: int) -> bool:
        """
        Attach a tag to an asset.
        
        Returns:
            True if attached, False if already attached or invalid IDs.
        """
        # Check if already attached
        stmt = select(asset_tags).where(
            asset_tags.c.asset_id == asset_id,
            asset_tags.c.tag_id == tag_id,
        )
        if self.session.execute(stmt).first():
            return False  # Already attached
        
        # Insert association
        self.session.execute(
            asset_tags.insert().values(asset_id=asset_id, tag_id=tag_id)
        )
        self.session.flush()
        return True

    def detach_tag(self, asset_id: int, tag_id: int) -> bool:
        """
        Detach a tag from an asset.
        
        Returns:
            True if detached, False if was not attached.
        """
        result = self.session.execute(
            asset_tags.delete().where(
                asset_tags.c.asset_id == asset_id,
                asset_tags.c.tag_id == tag_id,
            )
        )
        self.session.flush()
        return result.rowcount > 0

    def set_asset_tags(self, asset_id: int, tag_ids: list[int]) -> None:
        """
        Set the tags for an asset (replace all existing tags).
        
        Args:
            asset_id: The asset to update
            tag_ids: List of tag IDs to attach (empty list removes all tags)
        """
        # Remove all existing tags
        self.session.execute(
            asset_tags.delete().where(asset_tags.c.asset_id == asset_id)
        )
        # Add new tags
        if tag_ids:
            for tag_id in tag_ids:
                self.session.execute(
                    asset_tags.insert().values(asset_id=asset_id, tag_id=tag_id)
                )
        self.session.flush()

    def attach_tag_to_assets(self, tag_id: int, asset_ids: list[int]) -> int:
        """
        Attach a tag to multiple assets.
        
        Returns:
            Number of new attachments made.
        """
        count = 0
        for asset_id in asset_ids:
            if self.attach_tag(asset_id, tag_id):
                count += 1
        return count

    def detach_tag_from_assets(self, tag_id: int, asset_ids: list[int]) -> int:
        """
        Detach a tag from multiple assets.
        
        Returns:
            Number of detachments made.
        """
        count = 0
        for asset_id in asset_ids:
            if self.detach_tag(asset_id, tag_id):
                count += 1
        return count

    # ──────────────────────────────────────────────────────────────────────────
    # Filtering Assets by Tags
    # ──────────────────────────────────────────────────────────────────────────

    def get_assets_by_tags(
        self,
        tag_ids: list[int],
        status: AssetStatus | None = None,
    ) -> Sequence[Asset]:
        """
        Get assets that have ANY of the specified tags (OR semantics).
        
        Args:
            tag_ids: List of tag IDs to filter by
            status: Optional status filter (OWNED, WATCHLIST)
            
        Returns:
            Assets matching any of the tags, ordered by ticker.
        """
        if not tag_ids:
            return []
        
        stmt = (
            select(Asset)
            .distinct()
            .join(asset_tags, Asset.id == asset_tags.c.asset_id)
            .where(asset_tags.c.tag_id.in_(tag_ids))
        )
        
        if status:
            stmt = stmt.where(Asset.status == status)
        
        stmt = stmt.order_by(Asset.ticker)
        return self.session.scalars(stmt).all()

    def get_assets_by_tag_names(
        self,
        tag_names: list[str],
        status: AssetStatus | None = None,
    ) -> Sequence[Asset]:
        """
        Get assets that have ANY of the specified tags by name (OR semantics).
        
        Args:
            tag_names: List of tag names to filter by (case-insensitive)
            status: Optional status filter
            
        Returns:
            Assets matching any of the tags, ordered by ticker.
        """
        if not tag_names:
            return []
        
        lower_names = [n.lower() for n in tag_names]
        stmt = (
            select(Asset)
            .distinct()
            .join(asset_tags, Asset.id == asset_tags.c.asset_id)
            .join(Tag, asset_tags.c.tag_id == Tag.id)
            .where(func.lower(Tag.name).in_(lower_names))
        )
        
        if status:
            stmt = stmt.where(Asset.status == status)
        
        stmt = stmt.order_by(Asset.ticker)
        return self.session.scalars(stmt).all()

    def get_untagged_assets(self, status: AssetStatus | None = None) -> Sequence[Asset]:
        """
        Get assets that have no tags attached.
        
        Args:
            status: Optional status filter
            
        Returns:
            Untagged assets ordered by ticker.
        """
        # Subquery to find assets that have at least one tag
        tagged_asset_ids = select(asset_tags.c.asset_id).distinct()
        
        stmt = select(Asset).where(Asset.id.not_in(tagged_asset_ids))
        
        if status:
            stmt = stmt.where(Asset.status == status)
        
        stmt = stmt.order_by(Asset.ticker)
        return self.session.scalars(stmt).all()
