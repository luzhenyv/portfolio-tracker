"""
Risk analytics module.

FR-5: Asset-Level Risk Metrics
- Historical volatility (annualized)
- Maximum drawdown
- Return series derived from EOD prices

FR-6: Portfolio-Level Risk Metrics
- Weighted portfolio volatility
- Risk contribution by position

FR-7: Risk Constraints
- Explainable metrics only
- No probabilistic forecasting or VaR models
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from config import config
from db import get_db, AssetStatus
from db.repositories import AssetRepository, PositionRepository, PriceRepository


@dataclass
class AssetRiskMetrics:
    """Risk metrics for a single asset."""
    ticker: str
    asset_id: int
    volatility: float | None  # Annualized
    max_drawdown: float | None
    data_points: int


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics."""
    portfolio_volatility: float
    portfolio_max_drawdown: float
    correlation_matrix: pd.DataFrame | None


class RiskAnalyzer:
    """
    Computes risk metrics for assets and portfolio.
    
    FR-7: All metrics are historical and explainable.
    No probabilistic forecasting or VaR models.
    """
    
    def __init__(self):
        self.config = config.risk
    
    def _compute_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns from price data.
        
        Args:
            prices_df: DataFrame with columns [date, ticker, adjusted_close]
            
        Returns:
            DataFrame pivoted with date index, ticker columns, log returns values.
        """
        if prices_df.empty:
            return pd.DataFrame()
        
        pivot = prices_df.pivot(index="date", columns="ticker", values="adjusted_close")
        pivot = pivot.sort_index()
        
        # Log returns for better statistical properties
        returns = np.log(pivot / pivot.shift(1)).dropna()
        return returns
    
    def _annualized_volatility(self, returns: pd.Series | pd.DataFrame) -> float | pd.Series:
        """
        Calculate annualized volatility from returns.
        
        FR-5: Historical volatility (annualized)
        """
        return returns.std() * np.sqrt(self.config.trading_days_per_year)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from return series.
        
        FR-5: Maximum drawdown metric.
        """
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())
    
    def load_price_history(self, asset_ids: list[int] | None = None) -> pd.DataFrame:
        """
        Load price history for owned assets.
        
        Args:
            asset_ids: Optional list of asset IDs to filter.
            
        Returns:
            DataFrame with date, ticker, adjusted_close columns.
        """
        db = get_db()
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            
            if asset_ids:
                assets = [asset_repo.get_by_id(aid) for aid in asset_ids]
                assets = [a for a in assets if a is not None]
            else:
                assets = list(asset_repo.get_by_status(AssetStatus.OWNED))
            
            price_repo = PriceRepository(session)
            
            records = []
            for asset in assets:
                prices = price_repo.get_price_history(asset.id)
                for p in prices:
                    if p.adjusted_close is not None:
                        records.append({
                            "date": p.date,
                            "ticker": asset.ticker,
                            "adjusted_close": p.adjusted_close,
                        })
        
        return pd.DataFrame(records)
    
    def get_cost_based_weights(self) -> pd.Series:
        """
        Get cost-based portfolio weights.
        
        Reflects original capital allocation decisions.
        
        Returns:
            Series with ticker index and weight values.
        """
        db = get_db()
        
        with db.session() as session:
            position_repo = PositionRepository(session)
            position_data = position_repo.get_position_summary()
        
        if not position_data:
            return pd.Series(dtype=float)
        
        df = pd.DataFrame(position_data)
        df["weight"] = df["total_cost"] / df["total_cost"].sum()
        return df.set_index("ticker")["weight"]
    
    def compute_asset_risk(self, ticker: str, returns: pd.DataFrame) -> AssetRiskMetrics:
        """
        Compute risk metrics for a single asset.
        
        Args:
            ticker: Asset ticker symbol.
            returns: DataFrame of returns (from _compute_returns).
            
        Returns:
            AssetRiskMetrics for the asset.
        """
        if ticker not in returns.columns:
            return AssetRiskMetrics(
                ticker=ticker,
                asset_id=0,
                volatility=None,
                max_drawdown=None,
                data_points=0,
            )
        
        asset_returns = returns[ticker].dropna()
        data_points = len(asset_returns)
        
        # Check minimum data requirement
        if data_points < self.config.min_price_history_days:
            return AssetRiskMetrics(
                ticker=ticker,
                asset_id=0,
                volatility=None,
                max_drawdown=None,
                data_points=data_points,
            )
        
        return AssetRiskMetrics(
            ticker=ticker,
            asset_id=0,
            volatility=float(self._annualized_volatility(asset_returns)),
            max_drawdown=self._max_drawdown(asset_returns),
            data_points=data_points,
        )
    
    def compute_portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:
        """
        Compute portfolio returns from individual asset returns.
        
        FR-6: Weighted portfolio calculation.
        
        Args:
            returns: DataFrame of asset returns.
            weights: Series of portfolio weights.
            
        Returns:
            Series of portfolio returns.
        """
        aligned_weights = weights.reindex(returns.columns).fillna(0)
        return returns.dot(aligned_weights)
    
    def compute_risk_metrics(self) -> dict:
        """
        Compute comprehensive risk metrics.
        
        Returns:
            Dict containing:
            - asset_volatility: Series of annualized volatility per asset
            - asset_max_drawdown: Series of max drawdown per asset
            - portfolio_volatility: Float
            - portfolio_max_drawdown: Float
            - correlation: DataFrame correlation matrix
        """
        prices = self.load_price_history()
        
        if prices.empty:
            return {
                "asset_volatility": pd.Series(dtype=float),
                "asset_max_drawdown": pd.Series(dtype=float),
                "portfolio_volatility": 0.0,
                "portfolio_max_drawdown": 0.0,
                "correlation": pd.DataFrame(),
            }
        
        returns = self._compute_returns(prices)
        
        if returns.empty:
            return {
                "asset_volatility": pd.Series(dtype=float),
                "asset_max_drawdown": pd.Series(dtype=float),
                "portfolio_volatility": 0.0,
                "portfolio_max_drawdown": 0.0,
                "correlation": pd.DataFrame(),
            }
        
        weights = self.get_cost_based_weights()
        
        # Asset-level metrics
        asset_vol = self._annualized_volatility(returns)
        asset_mdd = returns.apply(self._max_drawdown)
        
        # Portfolio-level metrics
        portfolio_ret = self.compute_portfolio_returns(returns, weights)
        portfolio_vol = float(self._annualized_volatility(portfolio_ret))
        portfolio_mdd = self._max_drawdown(portfolio_ret)
        
        # Correlation matrix
        corr = returns.corr() if len(returns.columns) > 1 else pd.DataFrame()
        
        return {
            "asset_volatility": asset_vol,
            "asset_max_drawdown": asset_mdd,
            "portfolio_volatility": portfolio_vol,
            "portfolio_max_drawdown": portfolio_mdd,
            "correlation": corr,
        }


def compute_risk_metrics() -> dict:
    """
    Convenience function for risk metrics computation.
    
    Returns:
        Dict of risk metrics.
    """
    analyzer = RiskAnalyzer()
    return analyzer.compute_risk_metrics()


def load_price_history() -> pd.DataFrame:
    """
    Load price history for owned assets.
    
    Returns:
        DataFrame with date, ticker, adjusted_close columns.
    """
    analyzer = RiskAnalyzer()
    return analyzer.load_price_history()


def portfolio_weights() -> pd.Series:
    """
    Get cost-based portfolio weights.
    
    Returns:
        Series with ticker index and weight values.
    """
    analyzer = RiskAnalyzer()
    return analyzer.get_cost_based_weights()


def portfolio_weights_cost_based() -> pd.Series:
    """Alias for portfolio_weights (backward compatibility)."""
    return portfolio_weights()


if __name__ == "__main__":
    from db import init_db
    init_db()
    
    metrics = compute_risk_metrics()
    
    print("\n📉 Asset Volatility (Annualized)")
    if not metrics["asset_volatility"].empty:
        for ticker, vol in metrics["asset_volatility"].items():
            print(f"  {ticker}: {vol:.1%}")
    else:
        print("  No data available")
    
    print("\n📉 Asset Max Drawdown")
    if not metrics["asset_max_drawdown"].empty:
        for ticker, mdd in metrics["asset_max_drawdown"].items():
            print(f"  {ticker}: {mdd:.1%}")
    else:
        print("  No data available")
    
    print(f"\n📊 Portfolio Volatility: {metrics['portfolio_volatility']:.1%}")
    print(f"📊 Portfolio Max Drawdown: {metrics['portfolio_max_drawdown']:.1%}")
    
    if not metrics["correlation"].empty:
        print("\n🔗 Correlation Matrix")
        print(metrics["correlation"].round(2).to_string())
