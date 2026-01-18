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
    CheckConstraint,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class AssetStatus(str, Enum):
    """Asset status in portfolio."""
    OWNED = "OWNED"
    WATCHLIST = "WATCHLIST"


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
    watchlist_target: Mapped[Optional["WatchlistTarget"]] = relationship(
        "WatchlistTarget", back_populates="asset", uselist=False, cascade="all, delete-orphan"
    )
    thesis: Mapped[Optional["InvestmentThesis"]] = relationship(
        "InvestmentThesis", back_populates="asset", uselist=False, cascade="all, delete-orphan"
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
    volume: Mapped[Optional[int]] = mapped_column(Integer)

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
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
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
        Index("idx_trades_asset_date", "asset_id", "trade_date"),
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
    """
    __tablename__ = "positions"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    long_shares: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    long_avg_cost: Mapped[Optional[float]] = mapped_column(Float)
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

    def __repr__(self) -> str:
        return f"<Position(asset_id={self.asset_id}, long={self.long_shares}, short={self.short_shares})>"


class ValuationMetric(Base):
    """
    Valuation metrics auto-fetched from Yahoo Finance.
    
    FR-8: Auto-fetch Forward P/E, PEG, EV/EBITDA, growth metrics
    FR-9: NULL for missing data, no synthetic values
    """
    __tablename__ = "valuation_metrics"

    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    pe_forward: Mapped[Optional[float]] = mapped_column(Float)
    peg: Mapped[Optional[float]] = mapped_column(Float)
    ev_ebitda: Mapped[Optional[float]] = mapped_column(Float)
    revenue_growth: Mapped[Optional[float]] = mapped_column(Float)
    eps_growth: Mapped[Optional[float]] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc)
    )

    # Relationships
    asset: Mapped["Asset"] = relationship("Asset", back_populates="valuation")

    def __repr__(self) -> str:
        return f"<ValuationMetric(asset_id={self.asset_id}, pe_forward={self.pe_forward})>"


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
