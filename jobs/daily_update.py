"""
Daily Update Job Runner.

Coordinates the daily data refresh workflow:
1. Fetch EOD prices for all assets
2. Fetch valuation metrics for all assets
3. Report status

FR-1: Idempotent daily execution
FR-2: Must not crash on missing data
"""

import logging
import sys
from datetime import datetime
from typing import NamedTuple

from db import init_db
from data.fetch_prices import PriceFetcher, ValuationFetcher, FetchResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class JobResult(NamedTuple):
    """Result summary from a job run."""
    success: bool
    total_assets: int
    successful_assets: int
    total_records: int
    errors: list[str]


class DailyUpdateJob:
    """
    Daily data update job coordinator.
    
    Orchestrates price and valuation fetching with proper
    error handling and reporting.
    """
    
    def __init__(self):
        self.price_fetcher = PriceFetcher()
        self.valuation_fetcher = ValuationFetcher()
    
    def _summarize_results(
        self,
        results: list[FetchResult],
        job_name: str,
    ) -> JobResult:
        """Summarize fetch results into a JobResult."""
        successful = [r for r in results if r.success]
        errors = [f"{r.ticker}: {r.message}" for r in results if not r.success]
        total_records = sum(r.records_count for r in results)
        
        return JobResult(
            success=len(errors) == 0,
            total_assets=len(results),
            successful_assets=len(successful),
            total_records=total_records,
            errors=errors,
        )
    
    def run_price_update(self) -> JobResult:
        """
        Run price data update for all assets.
        
        Returns:
            JobResult with summary.
        """
        logger.info("📈 Starting price update...")
        
        results = self.price_fetcher.fetch_all_assets()
        summary = self._summarize_results(results, "Price Update")
        
        logger.info(
            f"📈 Price update complete: "
            f"{summary.successful_assets}/{summary.total_assets} assets, "
            f"{summary.total_records} new records"
        )
        
        return summary
    
    def run_valuation_update(self) -> JobResult:
        """
        Run valuation metrics update for all assets.
        
        Returns:
            JobResult with summary.
        """
        logger.info("📊 Starting valuation update...")
        
        results = self.valuation_fetcher.fetch_all_assets()
        summary = self._summarize_results(results, "Valuation Update")
        
        logger.info(
            f"📊 Valuation update complete: "
            f"{summary.successful_assets}/{summary.total_assets} assets"
        )
        
        return summary
    
    def run_full_update(self) -> dict[str, JobResult]:
        """
        Run complete daily update workflow.
        
        Returns:
            Dict mapping job name to JobResult.
        """
        logger.info("=" * 50)
        logger.info(f"🚀 Daily Update Started at {datetime.now()}")
        logger.info("=" * 50)
        
        results = {}
        
        # Initialize database
        init_db()
        
        # Run price update
        results["prices"] = self.run_price_update()
        
        # Run valuation update
        results["valuations"] = self.run_valuation_update()
        
        # Summary
        logger.info("=" * 50)
        logger.info("✅ Daily Update Complete")
        
        for name, result in results.items():
            status = "✓" if result.success else "✗"
            logger.info(
                f"  {status} {name}: "
                f"{result.successful_assets}/{result.total_assets} assets"
            )
            
            if result.errors:
                for error in result.errors:
                    logger.warning(f"    ⚠️ {error}")
        
        logger.info("=" * 50)
        
        return results


def run_daily_update():
    """Entry point for daily update job."""
    job = DailyUpdateJob()
    results = job.run_full_update()
    
    # Return non-zero exit code if any errors
    all_success = all(r.success for r in results.values())
    return 0 if all_success else 1


def run_prices_only():
    """Entry point for price-only update."""
    init_db()
    job = DailyUpdateJob()
    result = job.run_price_update()
    return 0 if result.success else 1


def run_valuations_only():
    """Entry point for valuation-only update."""
    init_db()
    job = DailyUpdateJob()
    result = job.run_valuation_update()
    return 0 if result.success else 1


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "prices":
            sys.exit(run_prices_only())
        elif command == "valuations":
            sys.exit(run_valuations_only())
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m jobs.daily_update [prices|valuations]")
            sys.exit(1)
    else:
        sys.exit(run_daily_update())
