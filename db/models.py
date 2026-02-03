"""
SQLAlchemy ORM Models for Portfolio Analytics System.

Defines all database entities following the requirements:
- Assets (stocks only in Phase 1, extensible for ETFs/crypto)
- Price data (EOD prices)
- Positions (holdings with cost basis)
- Valuation metrics (auto-fetched from Yahoo Finance)
- Watchlist targets and investment thesis
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Association table for Asset <-> Tag many-to-many relationship
asset_tags = Table(
    "asset_tags",
    Base.metadata,
    Column("asset_id", Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
    Index("idx_asset_tags_tag", "tag_id"),  # For "find assets by tag" queries
)


class Tag(Base):
    """
    Tag entity for categorizing assets.
    
    Supports many-to-many relationship with Asset.
    Examples: "AI", "FSD", "Magnificent Seven", "Semiconductor"
    
    Tag names are case-insensitive unique (enforced at service layer).
    Hard delete cascades to asset_tags join table.
    """
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    assets: Mapped[list["Asset"]] = relationship(
        "Asset", secondary=asset_tags, back_populates="tags"
    )

    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name={self.name!r})>"


class AssetStatus(str, Enum):
    """Asset status in portfolio."""
    OWNED = "OWNED"
    WATCHLIST = "WATCHLIST"


class AssetType(str, Enum):
    """Type of asset."""
    STOCK = "STOCK"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    BOND = "BOND"
    DERIVATIVE = "DERIVATIVE"


class ConfidenceLevel(str, Enum):
    """Confidence level for investment thesis."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TradeAction(str, Enum):
    """Trade action type for position management."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class Asset(Base):
    """
    Core asset entity representing a tradable security.
    
    Phase 1: US-listed stocks only.
    Future: Extensible for ETFs, crypto, options.
    """
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    currency: Mapped[str] = mapped_column(String(10), default="USD")
    exchange: Mapped[Optional[str]] = mapped_column(String(50))
    asset_type: Mapped[AssetType] = mapped_column(
        SQLEnum(AssetType, native_enum=False, length=20),
        nullable=False,
        default=AssetType.STOCK
    )
    status: Mapped[AssetStatus] = mapped_column(
        SQLEnum(AssetStatus, native_enum=False, length=20),
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    prices: Mapped[list["PriceDaily"]] = relationship(
        "PriceDaily", back_populates="asset", cascade="all, delete-orphan"
    )
    position: Mapped[Optional["Position"]] = relationship(
        "Position", back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    fundamentals: Mapped[list["FundamentalQuarterly"]] = relationship(
        "FundamentalQuarterly", back_populates="asset", cascade="all, delete-orphan"
    )
    valuation: Mapped[Optional["ValuationMetric"]] = relationship(
        "ValuationMetric", back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    valuation_override: Mapped[Optional["ValuationMetricOverride"]] = relationship(
        "ValuationMetricOverride", back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    watchlist_target: Mapped[Optional["WatchlistTarget"]] = relationship(
        "WatchlistTarget", back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    thesis: Mapped[Optional["InvestmentThesis"]] = relationship(
        "InvestmentThesis", back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    tags: Mapped[list["Tag"]] = relationship(
        "Tag", secondary=asset_tags, back_populates="assets"
    )

    def __repr__(self) -> str:
        return f"<Asset(id={self.id}, ticker={self.ticker}, status={self.status})>"


class PriceDaily(Base):
    """
    End-of-day (EOD) price data.
    
    FR-1: Daily EOD prices from Yahoo Finance
    FR-2: Raw values only, no interpolation
    """
    __tablename__ = "prices_daily"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[str] = mapped_column(String(10), primary_key=True)  # YYYY-MM-DD format
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    adjusted_close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[BigInteger]] = mapped_column(BigInteger)

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="prices")

    __table_args__ = (
        Index("idx_prices_asset_date", "asset_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<PriceDaily(asset_id={self.asset_id}, date={self.date}, close={self.close})>"


class FundamentalQuarterly(Base):
    """
    Quarterly fundamental data for deeper analysis.
    Future extension for fundamental-based screening.
    """
    __tablename__ = "fundamentals_quarterly"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    quarter: Mapped[str] = mapped_column(String(10), primary_key=True)  # e.g., "2024Q1"
    revenue: Mapped[Optional[float]] = mapped_column(Float)
    eps: Mapped[Optional[float]] = mapped_column(Float)
    free_cash_flow: Mapped[Optional[float]] = mapped_column(Float)
    roe: Mapped[Optional[float]] = mapped_column(Float)
    roic: Mapped[Optional[float]] = mapped_column(Float)
    gross_margin: Mapped[Optional[float]] = mapped_column(Float)
    operating_margin: Mapped[Optional[float]] = mapped_column(Float)
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="fundamentals")

    def __repr__(self) -> str:
        return f"<FundamentalQuarterly(asset_id={self.asset_id}, quarter={self.quarter})>"


class Trade(Base):
    """
    Trade ledger for position management.
    
    Records all buy/sell/short/cover transactions with realized P&L.
    Supports long and short position tracking with average cost method.
    """
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trade_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # Full timestamp
    action: Mapped[TradeAction] = mapped_column(
        SQLEnum(TradeAction, native_enum=False, length=10),
        nullable=False
    )
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset")

    __table_args__ = (
        Index("idx_trades_asset_date", "asset_id", "trade_at"),
    )

    def __repr__(self) -> str:
        return f"<Trade(id={self.id}, action={self.action}, shares={self.shares})>"


class Position(Base):
    """
    Current position state per asset (long/short inventory).
    
    FR-3: Position tracking with average cost basis.
    Maintains average cost for long positions and average price for short positions.
    Tracks cumulative realized P&L for reporting.
    Replaces old lot-based Position model - now uses Trade ledger for audit trail.
    
    Fields:
        long_avg_cost: Tax basis - weighted average of purchase prices (for tax reporting)
        net_invested: Cash still at risk = Total Cash Out - Total Cash In (for P&L display)
    """
    __tablename__ = "positions"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    long_shares: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    long_avg_cost: Mapped[Optional[float]] = mapped_column(Float)  # Tax basis
    net_invested: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # Cash at risk
    short_shares: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    short_avg_price: Mapped[Optional[float]] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc)
    )

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset")

    @property
    def net_shares(self) -> float:
        """Net shares (long - short)."""
        return self.long_shares - self.short_shares

    @property
    def gross_shares(self) -> float:
        """Gross shares (long + short)."""
        return self.long_shares + self.short_shares

    @property
    def net_invested_avg_cost(self) -> float | None:
        """Net Invested Avg Cost = net_invested / long_shares."""
        if self.long_shares > 0:
            return self.net_invested / self.long_shares
        return None

    def __repr__(self) -> str:
        return f"<Position(asset_id={self.asset_id}, long={self.long_shares}, short={self.short_shares})>"


class ValuationMetric(Base):
    """
    Valuation metrics auto-fetched from Yahoo Finance.
    
    Yahoo-aligned fields matching "Statistics" page:
    - Valuation Measures: Market Cap, EV, P/E, PEG, P/S, P/B, EV/Rev, EV/EBITDA
    - Financial Highlights: Margins, ROA, ROE, Revenue, Net Income, EPS, Cash, Debt, FCF
    
    FR-8: Auto-fetch via yfinance Ticker.info
    FR-9: NULL for missing data, no synthetic values
    """
    __tablename__ = "valuation_metrics"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    
    # Valuation Measures
    market_cap: Mapped[Optional[float]] = mapped_column(Float)
    enterprise_value: Mapped[Optional[float]] = mapped_column(Float)
    pe_trailing: Mapped[Optional[float]] = mapped_column(Float)
    pe_forward: Mapped[Optional[float]] = mapped_column(Float)
    peg: Mapped[Optional[float]] = mapped_column(Float)
    price_to_sales: Mapped[Optional[float]] = mapped_column(Float)
    price_to_book: Mapped[Optional[float]] = mapped_column(Float)
    ev_to_revenue: Mapped[Optional[float]] = mapped_column(Float)
    ev_ebitda: Mapped[Optional[float]] = mapped_column(Float)
    
    # Financial Highlights - Profitability
    profit_margin: Mapped[Optional[float]] = mapped_column(Float)
    return_on_assets: Mapped[Optional[float]] = mapped_column(Float)
    return_on_equity: Mapped[Optional[float]] = mapped_column(Float)
    
    # Financial Highlights - Income Statement
    revenue_ttm: Mapped[Optional[float]] = mapped_column(Float)
    net_income_ttm: Mapped[Optional[float]] = mapped_column(Float)
    diluted_eps_ttm: Mapped[Optional[float]] = mapped_column(Float)
    
    # Financial Highlights - Balance Sheet & Cash Flow
    total_cash: Mapped[Optional[float]] = mapped_column(Float)
    total_debt_to_equity: Mapped[Optional[float]] = mapped_column(Float)
    levered_free_cash_flow: Mapped[Optional[float]] = mapped_column(Float)
    
    # Legacy fields (kept for backward compatibility, can be deprecated)
    revenue_growth: Mapped[Optional[float]] = mapped_column(Float)
    eps_growth: Mapped[Optional[float]] = mapped_column(Float)
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc)
    )

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="valuation")

    def __repr__(self) -> str:
        return f"<ValuationMetric(asset_id={self.asset_id}, pe_forward={self.pe_forward})>"


class ValuationMetricOverride(Base):
    """
    User-provided override values for valuation metrics.
    
    Stores manual adjustments that take precedence over auto-fetched values.
    Supports all Yahoo-aligned metrics with optional *_override fields.
    NULL means no override (use fetched value).
    """
    __tablename__ = "valuation_metric_overrides"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    
    # Valuation Measures overrides
    market_cap_override: Mapped[Optional[float]] = mapped_column(Float)
    enterprise_value_override: Mapped[Optional[float]] = mapped_column(Float)
    pe_trailing_override: Mapped[Optional[float]] = mapped_column(Float)
    pe_forward_override: Mapped[Optional[float]] = mapped_column(Float)
    peg_override: Mapped[Optional[float]] = mapped_column(Float)
    price_to_sales_override: Mapped[Optional[float]] = mapped_column(Float)
    price_to_book_override: Mapped[Optional[float]] = mapped_column(Float)
    ev_to_revenue_override: Mapped[Optional[float]] = mapped_column(Float)
    ev_ebitda_override: Mapped[Optional[float]] = mapped_column(Float)
    
    # Financial Highlights - Profitability overrides
    profit_margin_override: Mapped[Optional[float]] = mapped_column(Float)
    return_on_assets_override: Mapped[Optional[float]] = mapped_column(Float)
    return_on_equity_override: Mapped[Optional[float]] = mapped_column(Float)
    
    # Financial Highlights - Income Statement overrides
    revenue_ttm_override: Mapped[Optional[float]] = mapped_column(Float)
    net_income_ttm_override: Mapped[Optional[float]] = mapped_column(Float)
    diluted_eps_ttm_override: Mapped[Optional[float]] = mapped_column(Float)
    
    # Financial Highlights - Balance Sheet & Cash Flow overrides
    total_cash_override: Mapped[Optional[float]] = mapped_column(Float)
    total_debt_to_equity_override: Mapped[Optional[float]] = mapped_column(Float)
    levered_free_cash_flow_override: Mapped[Optional[float]] = mapped_column(Float)
    
    # Legacy fields (kept for backward compatibility)
    revenue_growth_override: Mapped[Optional[float]] = mapped_column(Float)
    eps_growth_override: Mapped[Optional[float]] = mapped_column(Float)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc)
    )

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="valuation_override")

    def __repr__(self) -> str:
        return f"<ValuationMetricOverride(asset_id={self.asset_id})>"


class WatchlistTarget(Base):
    """
    Target prices and margin of safety for watchlist items.
    Supports disciplined buying decisions.
    """
    __tablename__ = "watchlist_targets"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    fair_value: Mapped[Optional[float]] = mapped_column(Float)
    target_buy_price: Mapped[Optional[float]] = mapped_column(Float)
    margin_of_safety: Mapped[Optional[float]] = mapped_column(Float)
    confidence_level: Mapped[Optional[ConfidenceLevel]] = mapped_column(
        SQLEnum(ConfidenceLevel, native_enum=False, length=20)
    )
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="watchlist_target")

    def __repr__(self) -> str:
        return f"<WatchlistTarget(asset_id={self.asset_id}, fair_value={self.fair_value})>"


class InvestmentThesis(Base):
    """
    Investment thesis and risk documentation.
    Supports long-term disciplined investing.
    """
    __tablename__ = "investment_thesis"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    thesis: Mapped[Optional[str]] = mapped_column(Text)
    key_risks: Mapped[Optional[str]] = mapped_column(Text)
    red_flags: Mapped[Optional[str]] = mapped_column(Text)
    last_reviewed: Mapped[Optional[str]] = mapped_column(String(10))  # YYYY-MM-DD

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="thesis")

    def __repr__(self) -> str:
        return f"<InvestmentThesis(asset_id={self.asset_id})>"


class CashTransactionType(str, Enum):
    """Type of cash transaction."""
    DEPOSIT = "DEPOSIT"      # Cash deposit (initial capital, add funds)
    WITHDRAW = "WITHDRAW"    # Cash withdrawal
    BUY = "BUY"              # Cash out for buying securities
    SELL = "SELL"            # Cash in from selling securities
    COVER = "COVER"          # Cash out for covering short
    SHORT = "SHORT"          # Cash in from shorting
    FEE = "FEE"              # Trading fees (separate tracking)
    DIVIDEND = "DIVIDEND"    # Dividend received
    INTEREST = "INTEREST"    # Interest earned/paid


class NoteTargetKind(str, Enum):
    """Kind of entity a note is attached to."""
    ASSET = "ASSET"          # Note about a specific stock/ETF
    TRADE = "TRADE"          # Note about a specific trade
    MARKET = "MARKET"        # Note about market/index (e.g., S&P 500, ^GSPC)
    JOURNAL = "JOURNAL"      # General journal entry (not attached to anything)


class NoteType(str, Enum):
    """Type/category of note content."""
    JOURNAL = "JOURNAL"          # General journal entry
    THESIS = "THESIS"            # Investment thesis
    RISK = "RISK"                # Risk analysis
    TRADE_PLAN = "TRADE_PLAN"    # Plan for a trade (e.g., "buy 10 TSLA at 416")
    TRADE_REVIEW = "TRADE_REVIEW"  # Post-trade review/postmortem
    MARKET_VIEW = "MARKET_VIEW"  # Market outlook/analysis
    EARNINGS = "EARNINGS"        # Earnings analysis
    NEWS = "NEWS"                # News summary/reaction
    OTHER = "OTHER"              # Miscellaneous


class NoteStatus(str, Enum):
    """Status of a note."""
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class CashTransaction(Base):
    """
    Cash transaction ledger for tracking cash position changes.
    
    Records all cash inflows and outflows including:
    - Deposits/withdrawals of capital
    - Trade-related cash flows (buy/sell)
    - Fees, dividends, interest
    
    Conventions:
    - Positive amount = cash inflow (DEPOSIT, SELL, SHORT, DIVIDEND, INTEREST)
    - Negative amount = cash outflow (WITHDRAW, BUY, COVER, FEE)
    
    Dividend Attribution:
    - asset_id (optional) links dividends to specific securities
    - Enables validation against holdings and per-asset income analytics
    """
    __tablename__ = "cash_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    transaction_type: Mapped[CashTransactionType] = mapped_column(
        SQLEnum(CashTransactionType, native_enum=False, length=20),
        nullable=False
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)  # Signed: + inflow, - outflow
    asset_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="SET NULL"), nullable=True
    )  # For DIVIDEND attribution
    trade_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("trades.id", ondelete="SET NULL"), nullable=True
    )
    description: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    asset: Mapped[Optional["Asset"]] = relationship("Asset")
    trade: Mapped[Optional["Trade"]] = relationship("Trade")

    __table_args__ = (
        Index("idx_cash_transactions_date", "transaction_at"),
        Index("idx_cash_transactions_type", "transaction_type"),
        Index("idx_cash_transactions_asset", "asset_id"),
    )

    def __repr__(self) -> str:
        return f"<CashTransaction(id={self.id}, type={self.transaction_type}, amount={self.amount})>"


class NoteTarget(Base):
    """
    Target entity for notes - what the note is attached to.
    
    Normalizes note attachment so notes can reference:
    - A specific asset (stock/ETF)
    - A specific trade
    - A market/index symbol (e.g., ^GSPC for S&P 500)
    - Nothing (journal entry)
    """
    __tablename__ = "note_targets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[NoteTargetKind] = mapped_column(
        SQLEnum(NoteTargetKind, native_enum=False, length=20),
        nullable=False
    )
    asset_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=True
    )
    trade_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("trades.id", ondelete="CASCADE"), nullable=True
    )
    symbol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # For MARKET targets
    symbol_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Display name
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    asset: Mapped[Optional["Asset"]] = relationship("Asset")
    trade: Mapped[Optional["Trade"]] = relationship("Trade")
    notes: Mapped[list["Note"]] = relationship(
        "Note", back_populates="target", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_note_targets_kind", "kind"),
        Index("idx_note_targets_asset", "asset_id"),
        Index("idx_note_targets_trade", "trade_id"),
        Index("idx_note_targets_symbol", "symbol"),
    )

    def __repr__(self) -> str:
        if self.kind == NoteTargetKind.ASSET:
            return f"<NoteTarget(id={self.id}, kind=ASSET, asset_id={self.asset_id})>"
        elif self.kind == NoteTargetKind.TRADE:
            return f"<NoteTarget(id={self.id}, kind=TRADE, trade_id={self.trade_id})>"
        elif self.kind == NoteTargetKind.MARKET:
            return f"<NoteTarget(id={self.id}, kind=MARKET, symbol={self.symbol})>"
        return f"<NoteTarget(id={self.id}, kind=JOURNAL)>"


class Note(Base):
    """
    Investment note/journal entry.
    
    Notes can be attached to assets, trades, market symbols, or be general journal entries.
    Designed to coexist with InvestmentThesis - thesis is structured, notes are freeform.
    
    Fields:
        title: Optional short title for the note
        note_type: Category (THESIS, RISK, TRADE_PLAN, MARKET_VIEW, etc.)
        summary: Brief summary for table/list display (can be LLM-generated later)
        key_points: Comma or newline-separated key points for quick reference
        body_md: Full note content in Markdown
        tags: Comma-separated tags for filtering (e.g., "bullish,earnings,AAPL")
        pinned: Pin important notes to top of lists
        status: ACTIVE, ARCHIVED, DELETED (soft delete)
    """
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("note_targets.id", ondelete="CASCADE"), nullable=False
    )
    note_type: Mapped[NoteType] = mapped_column(
        SQLEnum(NoteType, native_enum=False, length=20),
        nullable=False,
        default=NoteType.JOURNAL
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # For table display
    key_points: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Key takeaways
    body_md: Mapped[str] = mapped_column(Text, nullable=False)  # Full content in Markdown
    tags: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Comma-separated
    pinned: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    status: Mapped[NoteStatus] = mapped_column(
        SQLEnum(NoteStatus, native_enum=False, length=20),
        nullable=False,
        default=NoteStatus.ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc)
    )

    # Relationships
    target: Mapped["NoteTarget"] = relationship("NoteTarget", back_populates="notes")

    __table_args__ = (
        Index("idx_notes_target", "target_id"),
        Index("idx_notes_type", "note_type"),
        Index("idx_notes_status", "status"),
        Index("idx_notes_created", "created_at"),
    )

    @property
    def tags_list(self) -> list[str]:
        """Parse tags string into list."""
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    def __repr__(self) -> str:
        return f"<Note(id={self.id}, type={self.note_type}, title={self.title!r})>"


class IndexCategory(str, Enum):
    """Category of market index."""
    EQUITY = "EQUITY"
    VOLATILITY = "VOLATILITY"
    COMMODITY = "COMMODITY"
    BOND = "BOND"
    CURRENCY = "CURRENCY"


class MarketIndex(Base):
    """
    Market benchmark index for portfolio comparison.
    
    Tracks major indices like S&P 500, Russell 2000, VIX, etc.
    Used for correlation analysis and performance comparison.
    """
    __tablename__ = "market_indices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[IndexCategory] = mapped_column(
        SQLEnum(IndexCategory, native_enum=False, length=20),
        nullable=False,
        default=IndexCategory.EQUITY
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    prices: Mapped[list["IndexPriceDaily"]] = relationship(
        "IndexPriceDaily", back_populates="index", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<MarketIndex(id={self.id}, symbol={self.symbol}, name={self.name})>"


class IndexPriceDaily(Base):
    """
    End-of-day price data for market indices.
    
    Similar to PriceDaily but for benchmark indices.
    """
    __tablename__ = "index_prices_daily"

    index_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("market_indices.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[str] = mapped_column(String(10), primary_key=True)  # YYYY-MM-DD format
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[BigInteger]] = mapped_column(BigInteger)

    # Relationships
    index: Mapped["MarketIndex"] = relationship("MarketIndex", back_populates="prices")

    __table_args__ = (
        Index("idx_index_prices_index_date", "index_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<IndexPriceDaily(index_id={self.index_id}, date={self.date}, close={self.close})>"
