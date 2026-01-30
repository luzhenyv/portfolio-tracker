"""
Services layer for business logic orchestration.

Provides reusable services that can be consumed by CLI, Streamlit, FastAPI, etc.
"""

from services.asset_service import (
    AssetCreationResult,
    create_asset_with_data,
    print_asset_creation_result,
)
from services.position_service import (
    TradeResult,
    buy_position,
    sell_position,
    print_trade_result,
)
from services.market_index_service import (
    IndexSummary,
    IndexPricePoint,
    IndexPriceHistory,
    SyncResult,
    get_all_indices,
    get_index_by_symbol,
    get_index_by_id,
    get_indices_by_category,
    get_equity_indices,
    get_index_prices,
    get_latest_index_prices,
    get_index_returns,
    create_index,
    get_or_create_index,
    sync_index_prices,
    add_index_prices,
    get_configured_indices,
    ensure_indices_seeded,
)

__all__ = [
    # Asset service
    "AssetCreationResult",
    "create_asset_with_data",
    "print_asset_creation_result",
    # Position service
    "TradeResult",
    "buy_position",
    "sell_position",
    "print_trade_result",
    # Market index service
    "IndexSummary",
    "IndexPricePoint",
    "IndexPriceHistory",
    "SyncResult",
    "get_all_indices",
    "get_index_by_symbol",
    "get_index_by_id",
    "get_indices_by_category",
    "get_equity_indices",
    "get_index_prices",
    "get_latest_index_prices",
    "get_index_returns",
    "create_index",
    "get_or_create_index",
    "sync_index_prices",
    "add_index_prices",
    "get_configured_indices",
    "ensure_indices_seeded",
]
