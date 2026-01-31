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
    Tag,
    Trade,  # Transaction history
    TradeAction,
    ValuationMetric,
    WatchlistTarget,
    asset_tags,
)
from db.session import (
    DatabaseManager,
    get_db,
    init_db,
)
from db.repositories import (
    MarketIndexRepository,
    IndexPriceRepository,
    TagRepository,
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
    "Tag",
    "Trade",  # Transaction ledger
    "TradeAction",
    "ValuationMetric",
    "WatchlistTarget",
    "asset_tags",
    # Session management
    "DatabaseManager",
    "get_db",
    "init_db",
    # Repositories
    "MarketIndexRepository",
    "IndexPriceRepository",
    "TagRepository",
]
