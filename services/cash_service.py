"""
Cash Service - Handles cash balance and ledger operations.
"""

import logging
from db import get_db
from db.repositories import CashRepository

logger = logging.getLogger(__name__)

def get_cash_balance() -> float:
    """Get current cash balance."""
    db = get_db()
    with db.session() as session:
        cash_repo = CashRepository(session)
        return cash_repo.get_balance()

def deposit_cash(amount: float, transaction_at: str, description: str | None = None):
    """Execute cash deposit."""
    db = get_db()
    with db.session() as session:
        cash_repo = CashRepository(session)
        return cash_repo.deposit(
            amount=amount,
            transaction_at=transaction_at,
            description=description,
        )

def withdraw_cash(amount: float, transaction_at: str, description: str | None = None):
    """Execute cash withdrawal."""
    db = get_db()
    with db.session() as session:
        cash_repo = CashRepository(session)
        return cash_repo.withdraw(
            amount=amount,
            transaction_at=transaction_at,
            description=description,
        )

def get_cash_ledger(limit: int = 10):
    """Get recent cash transactions."""
    db = get_db()
    with db.session() as session:
        cash_repo = CashRepository(session)
        return cash_repo.get_ledger(limit=limit)
