"""
Database package initialization.

Exports commonly used components for convenient imports:
    from db import get_db, Asset, Position, etc.
"""

from db.models import (
    Asset,
    AssetStatus,
    Base,
    CashTransaction,
    CashTransactionType,
    ConfidenceLevel,
    FundamentalQuarterly,
    InvestmentThesis,
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

__all__ = [
    # Models
    "Asset",
    "AssetStatus",
    "Base",
    "CashTransaction",
    "CashTransactionType",
    "ConfidenceLevel",
    "FundamentalQuarterly",
    "InvestmentThesis",
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
]
