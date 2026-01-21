"""
Trade Service - Handles trade history and reporting.
"""

import logging
from db import get_db
from db.repositories import TradeRepository

logger = logging.getLogger(__name__)

def get_recent_trades(limit: int = 10):
    """Get recent trade activity."""
    db = get_db()
    with db.session() as session:
        trade_repo = TradeRepository(session)
        return trade_repo.get_all_trades(limit=limit)
