"""
Dividend Ingestion Job.

Provides dividend recording and validation for the portfolio:
1. Record dividend payments as cash transactions
2. Validate dividends against holdings
3. Check for anomalies (duplicate, magnitude outliers)

Dividends are treated as explicit cashflows to maintain accurate NAV
when using close prices (not adjusted_close) for returns calculation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from db import get_db, init_db
from db.models import (
    Asset,
    CashTransaction,
    CashTransactionType,
    Trade,
    TradeAction,
)
from db.repositories import (
    AssetRepository,
    CashRepository,
    PriceRepository,
    TradeRepository,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DividendValidationResult:
    """Result of dividend validation checks."""
    is_valid: bool
    warnings: list[str]
    errors: list[str]


@dataclass
class DividendRecord:
    """Dividend data for ingestion."""
    ticker: str
    amount: float
    transaction_at: str
    description: str | None = None


@dataclass
class DividendIngestResult:
    """Result of dividend ingestion."""
    success: bool
    ticker: str
    amount: float
    transaction_at: str
    transaction_id: int | None
    is_duplicate: bool
    validation: DividendValidationResult
    message: str


class DividendValidator:
    """
    Validates dividend records before ingestion.
    
    Checks:
    1. Structural: amount > 0, valid date, asset exists
    2. Holdings: verify non-zero shares near dividend date
    3. Magnitude: implied yield vs close price (flag outliers)
    """
    
    # Implied yield thresholds (annualized)
    MAX_QUARTERLY_YIELD = 0.20  # 20% quarterly = 80% annual (extreme outlier)
    WARN_QUARTERLY_YIELD = 0.05  # 5% quarterly = 20% annual (high but possible)
    
    def __init__(self):
        pass
    
    def validate(
        self,
        ticker: str,
        amount: float,
        transaction_at: str,
        asset_id: int | None = None,
    ) -> DividendValidationResult:
        """
        Run all validation checks on a dividend record.
        
        Args:
            ticker: Asset ticker symbol
            amount: Dividend amount (should be positive)
            transaction_at: Date in YYYY-MM-DD format
            asset_id: Optional asset ID (will be looked up if not provided)
            
        Returns:
            DividendValidationResult with is_valid, warnings, and errors
        """
        errors = []
        warnings = []
        
        # 1. Structural validation
        structural = self._validate_structural(ticker, amount, transaction_at)
        errors.extend(structural.errors)
        warnings.extend(structural.warnings)
        
        if structural.errors:
            # Don't proceed with other checks if basic structure is invalid
            return DividendValidationResult(
                is_valid=False,
                warnings=warnings,
                errors=errors,
            )
        
        # Lookup asset if not provided
        db = get_db()
        with db.session() as session:
            asset_repo = AssetRepository(session)
            asset = asset_repo.get_by_ticker(ticker)
            
            if not asset:
                errors.append(f"Asset '{ticker}' not found in database")
                return DividendValidationResult(
                    is_valid=False,
                    warnings=warnings,
                    errors=errors,
                )
            
            asset_id = asset.id
            
            # 2. Holdings validation
            holdings = self._validate_holdings(session, asset_id, transaction_at)
            errors.extend(holdings.errors)
            warnings.extend(holdings.warnings)
            
            # 3. Magnitude validation
            magnitude = self._validate_magnitude(
                session, asset_id, amount, transaction_at
            )
            # Magnitude issues are warnings only (don't block)
            warnings.extend(magnitude.warnings)
        
        return DividendValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )
    
    def _validate_structural(
        self,
        ticker: str,
        amount: float,
        transaction_at: str,
    ) -> DividendValidationResult:
        """Validate basic structure: amount > 0, valid date format."""
        errors = []
        warnings = []
        
        # Amount must be positive
        if amount <= 0:
            errors.append(f"Dividend amount must be positive (got {amount})")
        
        # Validate date format
        try:
            date_obj = datetime.strptime(transaction_at, "%Y-%m-%d")
            
            # Warn if date is in the future
            if date_obj.date() > datetime.now().date():
                warnings.append(f"Dividend date {transaction_at} is in the future")
            
            # Warn if date is very old (> 5 years)
            five_years_ago = datetime.now() - timedelta(days=5 * 365)
            if date_obj < five_years_ago:
                warnings.append(f"Dividend date {transaction_at} is more than 5 years ago")
                
        except ValueError:
            errors.append(f"Invalid date format: {transaction_at} (expected YYYY-MM-DD)")
        
        # Ticker should be non-empty
        if not ticker or not ticker.strip():
            errors.append("Ticker cannot be empty")
        
        return DividendValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )
    
    def _validate_holdings(
        self,
        session,
        asset_id: int,
        transaction_at: str,
    ) -> DividendValidationResult:
        """
        Validate that we held shares around the dividend date.
        
        Reconstructs holdings from trade history to check if position
        was non-zero near the ex-dividend date.
        """
        errors = []
        warnings = []
        
        trade_repo = TradeRepository(session)
        
        # Get all trades for this asset up to dividend date
        trades = trade_repo.get_trades_for_asset(
            asset_id=asset_id,
            end_date=transaction_at,
        )
        
        if not trades:
            warnings.append(
                f"No trade history found for asset before {transaction_at}. "
                "Cannot verify holdings."
            )
            return DividendValidationResult(
                is_valid=True,
                warnings=warnings,
                errors=errors,
            )
        
        # Reconstruct position as of dividend date
        # Trades are returned in descending order, so reverse for chronological
        shares = 0.0
        for trade in reversed(list(trades)):
            if trade.action == TradeAction.BUY:
                shares += trade.shares
            elif trade.action == TradeAction.SELL:
                shares -= trade.shares
            elif trade.action == TradeAction.SHORT:
                shares -= trade.shares
            elif trade.action == TradeAction.COVER:
                shares += trade.shares
        
        if shares <= 0:
            warnings.append(
                f"Holdings on {transaction_at} appear to be {shares:.2f} shares. "
                "Verify this dividend is correct."
            )
        
        return DividendValidationResult(
            is_valid=True,
            warnings=warnings,
            errors=errors,
        )
    
    def _validate_magnitude(
        self,
        session,
        asset_id: int,
        amount: float,
        transaction_at: str,
    ) -> DividendValidationResult:
        """
        Check if dividend amount is reasonable vs stock price.
        
        Computes implied quarterly yield and flags outliers.
        """
        warnings = []
        
        price_repo = PriceRepository(session)
        
        # Get price around dividend date
        prices = price_repo.get_price_history(
            asset_id=asset_id,
            start_date=transaction_at,
            end_date=transaction_at,
        )
        
        if not prices:
            # Try to get closest price before dividend date
            all_prices = price_repo.get_price_history(
                asset_id=asset_id,
                end_date=transaction_at,
            )
            if all_prices:
                prices = [all_prices[-1]]  # Most recent before date
        
        if not prices:
            warnings.append(
                f"No price data available around {transaction_at}. "
                "Cannot validate dividend magnitude."
            )
            return DividendValidationResult(
                is_valid=True,
                warnings=warnings,
                errors=[],
            )
        
        price = prices[0].close or prices[0].adjusted_close
        if not price or price <= 0:
            return DividendValidationResult(
                is_valid=True,
                warnings=warnings,
                errors=[],
            )
        
        # Calculate implied quarterly yield
        # Note: amount is per-share for validation purposes
        # This assumes amount is total dividend, not per-share
        # Actual validation requires knowing number of shares held
        implied_yield = amount / price
        
        if implied_yield > self.MAX_QUARTERLY_YIELD:
            warnings.append(
                f"Dividend yield {implied_yield:.1%} vs price ${price:.2f} "
                f"seems extremely high. Verify amount ${amount:.2f}."
            )
        elif implied_yield > self.WARN_QUARTERLY_YIELD:
            warnings.append(
                f"Dividend yield {implied_yield:.1%} vs price ${price:.2f} "
                f"is higher than typical. Amount: ${amount:.2f}."
            )
        
        return DividendValidationResult(
            is_valid=True,
            warnings=warnings,
            errors=[],
        )


class DividendIngestor:
    """
    Handles dividend ingestion with validation and deduplication.
    
    Workflow:
    1. Validate dividend record
    2. Check for duplicates (idempotency)
    3. Insert cash transaction with asset attribution
    """
    
    def __init__(self):
        self.validator = DividendValidator()
    
    def ingest_dividend(
        self,
        ticker: str,
        amount: float,
        transaction_at: str,
        description: str | None = None,
        skip_validation: bool = False,
    ) -> DividendIngestResult:
        """
        Ingest a single dividend record.
        
        Args:
            ticker: Asset ticker symbol
            amount: Dividend amount (positive)
            transaction_at: Date in YYYY-MM-DD format
            description: Optional description
            skip_validation: Skip validation checks (not recommended)
            
        Returns:
            DividendIngestResult with outcome details
        """
        # Default description
        if not description:
            description = f"Dividend from {ticker}"
        
        # Validate
        validation = DividendValidationResult(is_valid=True, warnings=[], errors=[])
        if not skip_validation:
            validation = self.validator.validate(ticker, amount, transaction_at)
            
            if not validation.is_valid:
                return DividendIngestResult(
                    success=False,
                    ticker=ticker,
                    amount=amount,
                    transaction_at=transaction_at,
                    transaction_id=None,
                    is_duplicate=False,
                    validation=validation,
                    message=f"Validation failed: {'; '.join(validation.errors)}",
                )
        
        db = get_db()
        with db.session() as session:
            asset_repo = AssetRepository(session)
            cash_repo = CashRepository(session)
            
            # Get asset
            asset = asset_repo.get_by_ticker(ticker)
            if not asset:
                return DividendIngestResult(
                    success=False,
                    ticker=ticker,
                    amount=amount,
                    transaction_at=transaction_at,
                    transaction_id=None,
                    is_duplicate=False,
                    validation=validation,
                    message=f"Asset '{ticker}' not found",
                )
            
            # Check for duplicate
            is_dup, existing_id = self._check_duplicate(
                session, asset.id, amount, transaction_at, description
            )
            
            if is_dup:
                return DividendIngestResult(
                    success=True,
                    ticker=ticker,
                    amount=amount,
                    transaction_at=transaction_at,
                    transaction_id=existing_id,
                    is_duplicate=True,
                    validation=validation,
                    message="Dividend already recorded (duplicate)",
                )
            
            # Create cash transaction
            tx = cash_repo.create_transaction(
                transaction_at=transaction_at,
                transaction_type=CashTransactionType.DIVIDEND,
                amount=amount,  # Positive for inflow
                asset_id=asset.id,
                description=description,
            )
            
            session.commit()
            
            # Log warnings if any
            for warning in validation.warnings:
                logger.warning(f"⚠️ {ticker}: {warning}")
            
            return DividendIngestResult(
                success=True,
                ticker=ticker,
                amount=amount,
                transaction_at=transaction_at,
                transaction_id=tx.id,
                is_duplicate=False,
                validation=validation,
                message="Dividend recorded successfully",
            )
    
    def _check_duplicate(
        self,
        session,
        asset_id: int,
        amount: float,
        transaction_at: str,
        description: str | None,
    ) -> tuple[bool, int | None]:
        """
        Check if this dividend is already recorded.
        
        Dedup strategy: asset_id + transaction_at + amount
        (description is optional in matching)
        """
        from sqlalchemy import select, and_
        
        # Query for matching dividend
        stmt = (
            select(CashTransaction)
            .where(
                and_(
                    CashTransaction.asset_id == asset_id,
                    CashTransaction.transaction_at == transaction_at,
                    CashTransaction.transaction_type == CashTransactionType.DIVIDEND,
                    CashTransaction.amount == amount,
                )
            )
        )
        
        existing = session.scalar(stmt)
        if existing:
            return True, existing.id
        
        return False, None
    
    def ingest_batch(
        self,
        dividends: list[DividendRecord],
        skip_validation: bool = False,
    ) -> list[DividendIngestResult]:
        """
        Ingest multiple dividend records.
        
        Args:
            dividends: List of DividendRecord objects
            skip_validation: Skip validation checks
            
        Returns:
            List of DividendIngestResult for each record
        """
        results = []
        
        for div in dividends:
            result = self.ingest_dividend(
                ticker=div.ticker,
                amount=div.amount,
                transaction_at=div.transaction_at,
                description=div.description,
                skip_validation=skip_validation,
            )
            results.append(result)
            
            # Log result
            if result.success:
                if result.is_duplicate:
                    logger.info(f"↩️  {div.ticker}: Duplicate skipped")
                else:
                    logger.info(
                        f"✅ {div.ticker}: ${div.amount:.2f} on {div.transaction_at}"
                    )
            else:
                logger.error(f"❌ {div.ticker}: {result.message}")
        
        return results


def run_dividend_ingest(dividends: list[dict]) -> int:
    """
    Entry point for dividend ingestion job.
    
    Args:
        dividends: List of dicts with keys: ticker, amount, transaction_at, description
        
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    init_db()
    
    ingestor = DividendIngestor()
    
    records = [
        DividendRecord(
            ticker=d["ticker"],
            amount=d["amount"],
            transaction_at=d["transaction_at"],
            description=d.get("description"),
        )
        for d in dividends
    ]
    
    results = ingestor.ingest_batch(records)
    
    # Summary
    success_count = sum(1 for r in results if r.success and not r.is_duplicate)
    dup_count = sum(1 for r in results if r.is_duplicate)
    error_count = sum(1 for r in results if not r.success)
    
    logger.info("=" * 50)
    logger.info(f"Dividend Ingestion Complete:")
    logger.info(f"  ✅ Recorded: {success_count}")
    logger.info(f"  ↩️  Duplicates: {dup_count}")
    logger.info(f"  ❌ Errors: {error_count}")
    logger.info("=" * 50)
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    import sys
    
    # Example usage with hardcoded test data
    # In practice, dividends would come from CSV import, broker API, or manual entry
    
    if len(sys.argv) < 4:
        print("Usage: python -m jobs.dividend_ingest <ticker> <amount> <date> [description]")
        print("Example: python -m jobs.dividend_ingest AAPL 0.24 2024-01-15 'Q4 2023 dividend'")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    amount = float(sys.argv[2])
    date = sys.argv[3]
    description = sys.argv[4] if len(sys.argv) > 4 else None
    
    result = run_dividend_ingest([{
        "ticker": ticker,
        "amount": amount,
        "transaction_at": date,
        "description": description,
    }])
    
    sys.exit(result)
