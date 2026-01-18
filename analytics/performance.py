"""
Performance analytics module.

Provides return calculations and performance attribution.
Designed for long-term holding review (quarters or longer).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from db import get_db, AssetStatus
from db.repositories import AssetRepository, PositionRepository, PriceRepository
from config import config


@dataclass
class PerformanceMetrics:
    """Performance metrics for a position or portfolio."""
    total_return: float
    annualized_return: float | None
    holding_period_days: int
    start_value: float
    end_value: float


@dataclass
class AssetPerformance:
    """Performance data for a single asset."""
    ticker: str
    asset_id: int
    total_return: float
    annualized_return: float | None
    holding_period_days: int
    first_price: float | None
    last_price: float | None


class PerformanceAnalyzer:
    """
    Calculates performance metrics for portfolio and individual positions.
    
    Focus on long-term returns appropriate for the investment philosophy.
    """
    
    def __init__(self):
        self.config = config.risk
    
    def _annualize_return(
        self,
        total_return: float,
        days: int,
    ) -> float | None:
        """
        Annualize a total return.
        
        Args:
            total_return: Total return (e.g., 0.25 for 25%)
            days: Holding period in days
            
        Returns:
            Annualized return or None if insufficient data.
        """
        if days < 30:  # Less than a month is not meaningful
            return None
        
        years = days / 365.0
        
        if total_return <= -1:  # Total loss
            return -1.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def get_price_returns(
        self,
        asset_id: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """
        Get price return series for an asset.
        
        Args:
            asset_id: Asset database ID.
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).
            
        Returns:
            Series of daily returns indexed by date.
        """
        db = get_db()
        
        with db.session() as session:
            price_repo = PriceRepository(session)
            prices = price_repo.get_price_history(asset_id, start_date, end_date)
        
        if not prices:
            return pd.Series(dtype=float)
        
        df = pd.DataFrame([
            {"date": p.date, "close": p.adjusted_close or p.close}
            for p in prices
            if p.adjusted_close or p.close
        ])
        
        if df.empty:
            return pd.Series(dtype=float)
        
        df = df.sort_values("date")
        df["return"] = df["close"].pct_change()
        
        return df.set_index("date")["return"].dropna()
    
    def compute_asset_performance(
        self,
        asset_id: int,
        start_date: str | None = None,
    ) -> AssetPerformance | None:
        """
        Compute performance metrics for a single asset.
        
        Args:
            asset_id: Asset database ID.
            start_date: Optional start date for calculation.
            
        Returns:
            AssetPerformance or None if insufficient data.
        """
        db = get_db()
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            price_repo = PriceRepository(session)
            
            asset = asset_repo.get_by_id(asset_id)
            if not asset:
                return None
            
            prices = price_repo.get_price_history(asset_id, start_date)
        
        if len(prices) < 2:
            return None
        
        first_price = prices[0].adjusted_close or prices[0].close
        last_price = prices[-1].adjusted_close or prices[-1].close
        
        if not first_price or not last_price:
            return None
        
        total_return = (last_price - first_price) / first_price
        
        first_date = datetime.strptime(prices[0].date, "%Y-%m-%d")
        last_date = datetime.strptime(prices[-1].date, "%Y-%m-%d")
        holding_days = (last_date - first_date).days
        
        return AssetPerformance(
            ticker=asset.ticker,
            asset_id=asset_id,
            total_return=total_return,
            annualized_return=self._annualize_return(total_return, holding_days),
            holding_period_days=holding_days,
            first_price=first_price,
            last_price=last_price,
        )
    
    def compute_portfolio_performance(
        self,
        lookback_days: int | None = None,
    ) -> PerformanceMetrics | None:
        """
        Compute overall portfolio performance.
        
        Args:
            lookback_days: Optional lookback period (default: all history).
            
        Returns:
            PerformanceMetrics or None if insufficient data.
        """
        from analytics.portfolio import PortfolioAnalyzer
        from analytics.risk import RiskAnalyzer
        
        risk_analyzer = RiskAnalyzer()
        
        # Get weights
        weights = risk_analyzer.get_cost_based_weights()
        if weights.empty:
            return None
        
        # Determine date range
        start_date = None
        if lookback_days:
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Load price history
        prices = risk_analyzer.load_price_history()
        if prices.empty:
            return None
        
        if start_date:
            prices = prices[prices["date"] >= start_date]
        
        returns = risk_analyzer._compute_returns(prices)
        if returns.empty:
            return None
        
        # Compute portfolio returns
        portfolio_returns = risk_analyzer.compute_portfolio_returns(returns, weights)
        
        # Calculate metrics
        cumulative_return = (1 + portfolio_returns).prod() - 1
        holding_days = len(portfolio_returns)
        
        return PerformanceMetrics(
            total_return=cumulative_return,
            annualized_return=self._annualize_return(cumulative_return, holding_days),
            holding_period_days=holding_days,
            start_value=1.0,  # Normalized
            end_value=1.0 + cumulative_return,
        )
    
    def get_period_returns(
        self,
        periods: list[int] = [30, 90, 180, 365],
    ) -> dict[str, float | None]:
        """
        Get portfolio returns for multiple lookback periods.
        
        Args:
            periods: List of lookback periods in days.
            
        Returns:
            Dict mapping period label to return value.
        """
        results = {}
        
        for days in periods:
            perf = self.compute_portfolio_performance(lookback_days=days)
            label = f"{days}d"
            if days >= 365:
                label = f"{days // 365}y"
            elif days >= 30:
                label = f"{days // 30}m"
            
            results[label] = perf.total_return if perf else None
        
        return results


def compute_performance_metrics(lookback_days: int | None = None) -> PerformanceMetrics | None:
    """
    Convenience function for portfolio performance.
    
    Args:
        lookback_days: Optional lookback period.
        
    Returns:
        PerformanceMetrics or None.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.compute_portfolio_performance(lookback_days)


def get_period_returns() -> dict[str, float | None]:
    """
    Get returns for standard periods (1m, 3m, 6m, 1y).
    
    Returns:
        Dict of period returns.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.get_period_returns()


if __name__ == "__main__":
    from db import init_db
    init_db()
    
    analyzer = PerformanceAnalyzer()
    
    print("\n📈 Portfolio Performance")
    perf = analyzer.compute_portfolio_performance()
    
    if perf:
        print(f"  Total Return: {perf.total_return:.1%}")
        if perf.annualized_return is not None:
            print(f"  Annualized Return: {perf.annualized_return:.1%}")
        print(f"  Holding Period: {perf.holding_period_days} days")
    else:
        print("  Insufficient data")
    
    print("\n📊 Period Returns")
    period_returns = analyzer.get_period_returns()
    for period, ret in period_returns.items():
        if ret is not None:
            print(f"  {period}: {ret:.1%}")
        else:
            print(f"  {period}: N/A")
