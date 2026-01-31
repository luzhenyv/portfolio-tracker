"""
Portfolio optimization module.

Implements Modern Portfolio Theory (MPT) optimization:
- Efficient Frontier computation
- Maximum Sharpe Ratio portfolio
- Minimum Volatility portfolio
- Rebalancing recommendations

Uses scipy.optimize for portfolio weight optimization.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize

from analytics.risk import RiskAnalyzer
from config import config
from db import get_db, AssetStatus
from db.repositories import PriceRepository


@dataclass
class OptimalPortfolio:
    """Represents an optimal portfolio allocation."""

    weights: pd.Series  # ticker -> weight
    expected_return: float  # Annualized expected return
    volatility: float  # Annualized volatility (risk)
    sharpe_ratio: float  # Risk-adjusted return


@dataclass
class EfficientFrontierResult:
    """Results from efficient frontier computation."""

    portfolios: pd.DataFrame  # Columns: expected_return, volatility, sharpe_ratio, weights (dict)
    max_sharpe_portfolio: OptimalPortfolio
    min_volatility_portfolio: OptimalPortfolio
    current_portfolio: OptimalPortfolio | None  # Current allocation metrics


@dataclass
class RebalanceRecommendation:
    """Rebalancing recommendation for a single asset."""

    ticker: str
    current_weight: float
    optimal_weight: float
    weight_diff: float
    current_shares: float
    target_shares: float
    shares_to_trade: float
    action: Literal["BUY", "SELL", "HOLD"]
    trade_value: float  # Estimated trade value in USD


class PortfolioOptimizer:
    """
    Portfolio optimizer using Modern Portfolio Theory.

    Computes efficient frontier and optimal allocations
    for maximum Sharpe ratio and minimum volatility portfolios.
    """

    def __init__(self, risk_free_rate: float | None = None):
        """
        Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.04 for 4%).
                           Defaults to config value.
        """
        self.risk_free_rate = risk_free_rate or config.risk.risk_free_rate
        self.trading_days = config.risk.trading_days_per_year
        self.risk_analyzer = RiskAnalyzer()

    def _load_returns(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """
        Load historical returns for assets.

        Args:
            tickers: Optional list of tickers. If None, loads all owned assets.

        Returns:
            DataFrame with date index and ticker columns containing log returns.
        """
        db = get_db()
        with db.session() as session:
            price_repo = PriceRepository(session)
            if tickers:
                records = price_repo.get_price_history_for_tickers(tickers)
            else:
                records = price_repo.get_price_history_for_assets(status=AssetStatus.OWNED)
            prices = pd.DataFrame(records)

        if prices.empty:
            return pd.DataFrame()

        return self.risk_analyzer._compute_returns(prices)

    def compute_expected_returns(
        self,
        returns: pd.DataFrame,
        method: Literal["historical", "ewma"] = "historical",
    ) -> pd.Series:
        """
        Compute expected annual returns for each asset.

        Args:
            returns: DataFrame of daily log returns.
            method: 'historical' for simple mean, 'ewma' for exponentially weighted.

        Returns:
            Series of annualized expected returns per ticker.
        """
        if method == "ewma":
            # Exponentially weighted mean with 60-day half-life
            daily_mean = returns.ewm(halflife=60).mean().iloc[-1]
        else:
            daily_mean = returns.mean()

        # Annualize returns
        return daily_mean * self.trading_days

    def compute_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute annualized covariance matrix.

        Args:
            returns: DataFrame of daily log returns.

        Returns:
            Annualized covariance matrix.
        """
        daily_cov = returns.cov()
        return daily_cov * self.trading_days

    def _portfolio_stats(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.

        Args:
            weights: Portfolio weights array.
            expected_returns: Expected returns array.
            cov_matrix: Covariance matrix.

        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio).
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if portfolio_volatility > 0:
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe = 0.0

        return float(portfolio_return), float(portfolio_volatility), float(sharpe)

    def _negative_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Objective function: negative Sharpe ratio (for minimization)."""
        _, _, sharpe = self._portfolio_stats(weights, expected_returns, cov_matrix)
        return -sharpe

    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Objective function: portfolio volatility."""
        _, vol, _ = self._portfolio_stats(weights, expected_returns, cov_matrix)
        return vol

    def _optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        objective: Literal["max_sharpe", "min_volatility"],
        n_assets: int,
    ) -> np.ndarray:
        """
        Optimize portfolio weights for given objective.

        Args:
            expected_returns: Expected returns array.
            cov_matrix: Covariance matrix.
            objective: 'max_sharpe' or 'min_volatility'.
            n_assets: Number of assets.

        Returns:
            Optimal weights array.
        """
        # Initial guess: equal weights
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: long-only (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Select objective function
        if objective == "max_sharpe":
            obj_func = self._negative_sharpe
        else:
            obj_func = self._portfolio_volatility

        result = optimize.minimize(
            obj_func,
            init_weights,
            args=(expected_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def compute_efficient_frontier(
        self,
        n_portfolios: int = 500,
        tickers: list[str] | None = None,
        current_weights: pd.Series | None = None,
    ) -> EfficientFrontierResult | None:
        """
        Compute the efficient frontier for a portfolio.

        Args:
            n_portfolios: Number of random portfolios to generate.
            tickers: Optional list of tickers. If None, uses all owned assets.
            current_weights: Current portfolio weights for comparison.

        Returns:
            EfficientFrontierResult or None if insufficient data.
        """
        # Load returns
        returns = self._load_returns(tickers)
        if returns.empty or len(returns.columns) < 2:
            return None

        # Ensure minimum data points
        if len(returns) < config.risk.min_price_history_days:
            return None

        tickers_list = returns.columns.tolist()
        n_assets = len(tickers_list)

        # Compute expected returns and covariance
        expected_returns = self.compute_expected_returns(returns)
        cov_matrix = self.compute_covariance_matrix(returns)

        # Convert to numpy arrays
        exp_ret_arr = expected_returns.values
        cov_arr = cov_matrix.values

        # Generate random portfolios for frontier visualization
        portfolios_data = []
        np.random.seed(42)  # Reproducibility

        for _ in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)

            port_return, port_vol, port_sharpe = self._portfolio_stats(
                weights, exp_ret_arr, cov_arr
            )

            portfolios_data.append(
                {
                    "expected_return": port_return,
                    "volatility": port_vol,
                    "sharpe_ratio": port_sharpe,
                    "weights": dict(zip(tickers_list, weights)),
                }
            )

        portfolios_df = pd.DataFrame(portfolios_data)

        # Optimize for max Sharpe ratio
        max_sharpe_weights = self._optimize_portfolio(
            exp_ret_arr, cov_arr, "max_sharpe", n_assets
        )
        max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = self._portfolio_stats(
            max_sharpe_weights, exp_ret_arr, cov_arr
        )
        max_sharpe_portfolio = OptimalPortfolio(
            weights=pd.Series(max_sharpe_weights, index=tickers_list),
            expected_return=max_sharpe_ret,
            volatility=max_sharpe_vol,
            sharpe_ratio=max_sharpe_sr,
        )

        # Optimize for minimum volatility
        min_vol_weights = self._optimize_portfolio(
            exp_ret_arr, cov_arr, "min_volatility", n_assets
        )
        min_vol_ret, min_vol_vol, min_vol_sr = self._portfolio_stats(
            min_vol_weights, exp_ret_arr, cov_arr
        )
        min_vol_portfolio = OptimalPortfolio(
            weights=pd.Series(min_vol_weights, index=tickers_list),
            expected_return=min_vol_ret,
            volatility=min_vol_vol,
            sharpe_ratio=min_vol_sr,
        )

        # Current portfolio stats (if provided)
        current_portfolio = None
        if current_weights is not None:
            # Align current weights with tickers in returns
            aligned_weights = current_weights.reindex(tickers_list).fillna(0)
            # Normalize to sum to 1
            if aligned_weights.sum() > 0:
                aligned_weights = aligned_weights / aligned_weights.sum()
                curr_ret, curr_vol, curr_sr = self._portfolio_stats(
                    aligned_weights.values, exp_ret_arr, cov_arr
                )
                current_portfolio = OptimalPortfolio(
                    weights=aligned_weights,
                    expected_return=curr_ret,
                    volatility=curr_vol,
                    sharpe_ratio=curr_sr,
                )

        return EfficientFrontierResult(
            portfolios=portfolios_df,
            max_sharpe_portfolio=max_sharpe_portfolio,
            min_volatility_portfolio=min_vol_portfolio,
            current_portfolio=current_portfolio,
        )

    def generate_rebalance_recommendations(
        self,
        current_weights: pd.Series,
        current_shares: pd.Series,
        current_prices: pd.Series,
        target_portfolio: OptimalPortfolio,
        portfolio_value: float,
        threshold: float = 0.02,
    ) -> list[RebalanceRecommendation]:
        """
        Generate rebalancing recommendations.

        Args:
            current_weights: Current portfolio weights (ticker -> weight).
            current_shares: Current shares held (ticker -> shares).
            current_prices: Current prices (ticker -> price).
            target_portfolio: Target optimal portfolio.
            portfolio_value: Total portfolio value in USD.
            threshold: Minimum weight difference to recommend action (default 2%).

        Returns:
            List of RebalanceRecommendation objects.
        """
        recommendations = []

        # Get all tickers from both current and target
        all_tickers = set(current_weights.index) | set(target_portfolio.weights.index)

        for ticker in sorted(all_tickers):
            curr_weight = current_weights.get(ticker, 0.0)
            optimal_weight = target_portfolio.weights.get(ticker, 0.0)
            weight_diff = optimal_weight - curr_weight

            curr_shares = current_shares.get(ticker, 0.0)
            curr_price = current_prices.get(ticker, 0.0)

            # Calculate target shares
            target_value = optimal_weight * portfolio_value
            target_shares = target_value / curr_price if curr_price > 0 else 0

            shares_diff = target_shares - curr_shares
            trade_value = shares_diff * curr_price

            # Determine action
            if abs(weight_diff) < threshold:
                action = "HOLD"
            elif weight_diff > 0:
                action = "BUY"
            else:
                action = "SELL"

            recommendations.append(
                RebalanceRecommendation(
                    ticker=ticker,
                    current_weight=curr_weight,
                    optimal_weight=optimal_weight,
                    weight_diff=weight_diff,
                    current_shares=curr_shares,
                    target_shares=target_shares,
                    shares_to_trade=shares_diff,
                    action=action,
                    trade_value=trade_value,
                )
            )

        return recommendations


def compute_efficient_frontier(
    current_weights: pd.Series | None = None,
    n_portfolios: int = 500,
) -> EfficientFrontierResult | None:
    """
    Convenience function to compute efficient frontier.

    Args:
        current_weights: Optional current portfolio weights.
        n_portfolios: Number of random portfolios to simulate.

    Returns:
        EfficientFrontierResult or None if insufficient data.
    """
    optimizer = PortfolioOptimizer()
    return optimizer.compute_efficient_frontier(
        n_portfolios=n_portfolios,
        current_weights=current_weights,
    )


def get_rebalance_recommendations(
    current_weights: pd.Series,
    current_shares: pd.Series,
    current_prices: pd.Series,
    portfolio_value: float,
    threshold: float = 0.02,
) -> list[RebalanceRecommendation]:
    """
    Convenience function to get rebalancing recommendations.

    Computes efficient frontier and generates recommendations
    based on the maximum Sharpe ratio portfolio.

    Args:
        current_weights: Current portfolio weights.
        current_shares: Current shares per ticker.
        current_prices: Current prices per ticker.
        portfolio_value: Total portfolio value.
        threshold: Minimum weight difference threshold.

    Returns:
        List of RebalanceRecommendation objects.
    """
    optimizer = PortfolioOptimizer()
    frontier = optimizer.compute_efficient_frontier(current_weights=current_weights)

    if frontier is None:
        return []

    return optimizer.generate_rebalance_recommendations(
        current_weights=current_weights,
        current_shares=current_shares,
        current_prices=current_prices,
        target_portfolio=frontier.max_sharpe_portfolio,
        portfolio_value=portfolio_value,
        threshold=threshold,
    )


if __name__ == "__main__":
    from db import init_db
    from analytics import portfolio_weights, compute_portfolio

    init_db()

    print("📊 Computing Efficient Frontier...")
    weights = portfolio_weights()
    print(f"\nCurrent weights:\n{weights}")

    frontier = compute_efficient_frontier(current_weights=weights)

    if frontier:
        print("\n✨ Max Sharpe Portfolio:")
        print(f"  Expected Return: {frontier.max_sharpe_portfolio.expected_return:.1%}")
        print(f"  Volatility: {frontier.max_sharpe_portfolio.volatility:.1%}")
        print(f"  Sharpe Ratio: {frontier.max_sharpe_portfolio.sharpe_ratio:.2f}")
        print("\n  Weights:")
        for ticker, weight in frontier.max_sharpe_portfolio.weights.items():
            print(f"    {ticker}: {weight:.1%}")

        print("\n🛡️ Min Volatility Portfolio:")
        print(f"  Expected Return: {frontier.min_volatility_portfolio.expected_return:.1%}")
        print(f"  Volatility: {frontier.min_volatility_portfolio.volatility:.1%}")
        print(f"  Sharpe Ratio: {frontier.min_volatility_portfolio.sharpe_ratio:.2f}")

        if frontier.current_portfolio:
            print("\n📍 Current Portfolio:")
            print(f"  Expected Return: {frontier.current_portfolio.expected_return:.1%}")
            print(f"  Volatility: {frontier.current_portfolio.volatility:.1%}")
            print(f"  Sharpe Ratio: {frontier.current_portfolio.sharpe_ratio:.2f}")
    else:
        print("❌ Insufficient data for frontier calculation")
