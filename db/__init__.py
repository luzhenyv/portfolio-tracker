"""
Database package initialization.

Exports commonly used components for convenient imports:
    from db import get_db, Asset, Position, etc.
"""

from db.models import (
    Asset,
    AssetStatus,
    AssetType,
    Base,
    CashTransaction,
    CashTransactionType,
    ConfidenceLevel,
    FundamentalQuarterly,
    IndexCategory,
    IndexPriceDaily,
    InvestmentThesis,
    MarketIndex,
    Note,
    NoteTarget,
    NoteTargetKind,
    NoteType,
    Position,  # Current position state (formerly PositionState)
    PriceDaily,
    Trade,  # Transaction history
    TradeAction,
    ValuationMetric,
    WatchlistTarget,
)
from db.session import (
    DatabaseManager,
    get_db,
    init_db,
)
from db.repositories import (
    MarketIndexRepository,
    IndexPriceRepository,
)

__all__ = [
    # Models
    "Asset",
    "AssetStatus",
    "AssetType",
    "Base",
    "CashTransaction",
    "CashTransactionType",
    "ConfidenceLevel",
    "FundamentalQuarterly",
    "IndexCategory",
    "IndexPriceDaily",
    "InvestmentThesis",
    "MarketIndex",
    "Note",
    "NoteTarget",
    "NoteTargetKind",
    "NoteType",
    "Position",  # Current position state
    "PriceDaily",
    "Trade",  # Transaction ledger
    "TradeAction",
    "ValuationMetric",
    "WatchlistTarget",
    # Session management
    "DatabaseManager",
    "get_db",
    "init_db",
    # Repositories
    "MarketIndexRepository",
    "IndexPriceRepository",
]
