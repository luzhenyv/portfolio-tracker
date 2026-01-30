"""
Benchmark comparison analytics module.

Provides portfolio vs benchmark comparison metrics:
- Correlation analysis between portfolio and indices
- Relative returns (alpha generation)
- Beta calculations against selected benchmarks
- Rolling performance comparisons

Designed to work with market indices stored in the database.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from analytics.portfolio import PortfolioAnalyzer
from config import config
from db import get_db
from db.repositories import (
    IndexPriceRepository,
    MarketIndexRepository,
    PriceRepository,
    PositionRepository,
)


@dataclass
class BenchmarkMetrics:
    """Comparison metrics against a single benchmark."""

    benchmark_symbol: str
    benchmark_name: str
    # Returns comparison
    portfolio_return: float
    benchmark_return: float
    relative_return: float  # Portfolio return - Benchmark return (alpha proxy)
    # Risk metrics
    correlation: float | None  # Pearson correlation of returns
    beta: float | None  # Sensitivity to benchmark movements
    alpha: float | None  # Jensen's alpha (annualized excess return)
    tracking_error: float | None  # Std dev of return difference (annualized)
    information_ratio: float | None  # Relative return / tracking error
    # R-squared
    r_squared: float | None  # % of portfolio variance explained by benchmark
    # Data quality
    overlapping_days: int  # Number of days with both portfolio and benchmark data
    start_date: str
    end_date: str


@dataclass
class RollingBetaResult:
    """Rolling beta calculation result."""

    benchmark_symbol: str
    dates: list[str]
    betas: list[float]
    window_days: int


@dataclass
class CorrelationMatrix:
    """Correlation matrix between portfolio and multiple benchmarks."""

    # Full correlation matrix including portfolio and all benchmarks
    matrix: pd.DataFrame
    # Portfolio correlations with each benchmark
    portfolio_correlations: dict[str, float]


@dataclass
class BenchmarkComparisonResult:
    """Complete benchmark comparison analysis."""

    # Individual benchmark comparisons
    benchmarks: list[BenchmarkMetrics]
    # Correlation analysis
    correlation_matrix: CorrelationMatrix | None
    # Best performing benchmark (highest correlation)
    best_correlation_benchmark: str | None
    # Analysis period
    start_date: str
    end_date: str


class BenchmarkAnalyzer:
    """
    Analyzes portfolio performance relative to market benchmarks.

    Provides correlation, beta, and relative return analysis
    against configured market indices.
    """

    def __init__(self):
        self.config = config.risk
        self.portfolio_analyzer = PortfolioAnalyzer()

    def _compute_returns(
        self, prices_df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """
        Compute log returns from price data.

        Args:
            prices_df: DataFrame with columns [date, ticker/symbol, close]
            price_col: Column name for prices.

        Returns:
            DataFrame pivoted with date index, ticker columns, log returns values.
        """
        if prices_df.empty:
            return pd.DataFrame()

        # Determine the column name for grouping (ticker for assets, symbol for indices)
        group_col = "ticker" if "ticker" in prices_df.columns else "symbol"

        pivot = prices_df.pivot(index="date", columns=group_col, values=price_col)
        pivot = pivot.sort_index()

        # Log returns for better statistical properties
        returns = np.log(pivot / pivot.shift(1)).dropna()
        return returns

    def _get_portfolio_returns(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """
        Get portfolio weighted returns.

        Args:
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Series of daily portfolio returns with date index.
        """
        db = get_db()

        with db.session() as session:
            price_repo = PriceRepository(session)
            position_repo = PositionRepository(session)

            # Get owned asset prices
            price_records = price_repo.get_price_history_for_assets(
                status="OWNED",
                start_date=start_date,
                end_date=end_date,
            )

        if not price_records:
            return pd.Series(dtype=float)

        prices_df = pd.DataFrame(price_records)
        returns = self._compute_returns(prices_df)

        if returns.empty:
            return pd.Series(dtype=float)

        # Get portfolio weights
        weights = self.portfolio_analyzer.get_portfolio_weights()

        # Align weights to available assets
        aligned_weights = weights.reindex(returns.columns).fillna(0)
        if aligned_weights.sum() == 0:
            return pd.Series(dtype=float)

        # Normalize weights to sum to 1
        aligned_weights = aligned_weights / aligned_weights.sum()

        # Compute weighted portfolio returns
        portfolio_returns = returns.dot(aligned_weights)
        portfolio_returns.name = "portfolio"

        return portfolio_returns

    def _get_benchmark_returns(
        self,
        benchmark_symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Get returns for benchmark indices.

        Args:
            benchmark_symbols: List of benchmark symbols. If None, uses all.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            DataFrame with date index and benchmark symbols as columns.
        """
        db = get_db()

        with db.session() as session:
            index_repo = MarketIndexRepository(session)
            price_repo = IndexPriceRepository(session)

            # Get index IDs
            indices = index_repo.get_all()
            if benchmark_symbols:
                indices = [i for i in indices if i.symbol in benchmark_symbols]

            if not indices:
                return pd.DataFrame()

            index_ids = [i.id for i in indices]

            # Get price history
            price_records = price_repo.get_price_history_for_indices(
                index_ids=index_ids,
                start_date=start_date,
                end_date=end_date,
            )

        if not price_records:
            return pd.DataFrame()

        prices_df = pd.DataFrame(price_records)
        returns = self._compute_returns(prices_df)

        return returns

    def _calculate_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> tuple[float | None, float | None, float | None]:
        """
        Calculate beta using OLS regression.

        Beta = Cov(Rp, Rm) / Var(Rm)
        Alpha = Mean(Rp) - Beta * Mean(Rm), annualized

        Args:
            portfolio_returns: Portfolio return series.
            benchmark_returns: Benchmark return series.

        Returns:
            Tuple of (beta, alpha, r_squared).
        """
        # Align series
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1, join="inner"
        ).dropna()

        if len(aligned) < self.config.min_price_history_days:
            return None, None, None

        port_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]

        # Calculate beta using covariance method
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        variance = np.var(bench_ret, ddof=1)

        if variance == 0:
            return None, None, None

        beta = covariance / variance

        # Calculate alpha (annualized)
        trading_days = self.config.trading_days_per_year
        mean_port = port_ret.mean() * trading_days
        mean_bench = bench_ret.mean() * trading_days
        alpha = mean_port - beta * mean_bench

        # Calculate R-squared
        correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
        r_squared = correlation**2

        return float(beta), float(alpha), float(r_squared)

    def _calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float | None:
        """
        Calculate annualized tracking error.

        Tracking Error = Std(Rp - Rb) * sqrt(252)

        Args:
            portfolio_returns: Portfolio return series.
            benchmark_returns: Benchmark return series.

        Returns:
            Annualized tracking error or None if insufficient data.
        """
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1, join="inner"
        ).dropna()

        if len(aligned) < self.config.min_price_history_days:
            return None

        diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tracking_error = diff.std() * np.sqrt(self.config.trading_days_per_year)

        return float(tracking_error)

    def compare_to_benchmark(
        self,
        benchmark_symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BenchmarkMetrics | None:
        """
        Compare portfolio performance to a single benchmark.

        Args:
            benchmark_symbol: Symbol of the benchmark index (e.g., "SPX").
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).

        Returns:
            BenchmarkMetrics or None if insufficient data.
        """
        # Get returns
        portfolio_returns = self._get_portfolio_returns(start_date, end_date)
        benchmark_returns_df = self._get_benchmark_returns(
            [benchmark_symbol], start_date, end_date
        )

        if portfolio_returns.empty or benchmark_returns_df.empty:
            return None

        if benchmark_symbol not in benchmark_returns_df.columns:
            return None

        benchmark_returns = benchmark_returns_df[benchmark_symbol]

        # Align data
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1, join="inner"
        ).dropna()

        if len(aligned) < self.config.min_price_history_days:
            return None

        port_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]

        # Get benchmark name
        db = get_db()
        with db.session() as session:
            index_repo = MarketIndexRepository(session)
            index = index_repo.get_by_symbol(benchmark_symbol)
            benchmark_name = index.name if index else benchmark_symbol

        # Calculate metrics
        # Total returns (cumulative)
        portfolio_total_return = float(np.exp(port_ret.sum()) - 1)
        benchmark_total_return = float(np.exp(bench_ret.sum()) - 1)
        relative_return = portfolio_total_return - benchmark_total_return

        # Correlation
        correlation = float(np.corrcoef(port_ret, bench_ret)[0, 1])

        # Beta, alpha, R-squared
        beta, alpha, r_squared = self._calculate_beta(port_ret, bench_ret)

        # Tracking error
        tracking_error = self._calculate_tracking_error(port_ret, bench_ret)

        # Information ratio
        information_ratio = None
        if tracking_error and tracking_error > 0:
            # Annualized relative return
            days = len(aligned)
            ann_relative_return = relative_return * (
                self.config.trading_days_per_year / days
            )
            information_ratio = ann_relative_return / tracking_error

        return BenchmarkMetrics(
            benchmark_symbol=benchmark_symbol,
            benchmark_name=benchmark_name,
            portfolio_return=portfolio_total_return,
            benchmark_return=benchmark_total_return,
            relative_return=relative_return,
            correlation=correlation,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            r_squared=r_squared,
            overlapping_days=len(aligned),
            start_date=aligned.index.min(),
            end_date=aligned.index.max(),
        )

    def compare_to_all_benchmarks(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        benchmark_symbols: list[str] | None = None,
    ) -> BenchmarkComparisonResult:
        """
        Compare portfolio to all available benchmarks.

        Args:
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).
            benchmark_symbols: Optional list of specific benchmarks to compare.

        Returns:
            BenchmarkComparisonResult with all comparisons.
        """
        # Get all available benchmarks if not specified
        if benchmark_symbols is None:
            db = get_db()
            with db.session() as session:
                index_repo = MarketIndexRepository(session)
                indices = index_repo.get_all()
                benchmark_symbols = [i.symbol for i in indices]

        if not benchmark_symbols:
            return BenchmarkComparisonResult(
                benchmarks=[],
                correlation_matrix=None,
                best_correlation_benchmark=None,
                start_date=start_date or "",
                end_date=end_date or "",
            )

        # Compare to each benchmark
        benchmarks = []
        for symbol in benchmark_symbols:
            metrics = self.compare_to_benchmark(symbol, start_date, end_date)
            if metrics:
                benchmarks.append(metrics)

        # Calculate correlation matrix
        correlation_matrix = self.compute_correlation_matrix(
            benchmark_symbols, start_date, end_date
        )

        # Find best correlation benchmark
        best_correlation_benchmark = None
        if benchmarks:
            valid_benchmarks = [b for b in benchmarks if b.correlation is not None]
            if valid_benchmarks:
                best = max(valid_benchmarks, key=lambda x: abs(x.correlation))
                best_correlation_benchmark = best.benchmark_symbol

        # Determine date range from results
        result_start = start_date or ""
        result_end = end_date or ""
        if benchmarks:
            result_start = min(b.start_date for b in benchmarks)
            result_end = max(b.end_date for b in benchmarks)

        return BenchmarkComparisonResult(
            benchmarks=benchmarks,
            correlation_matrix=correlation_matrix,
            best_correlation_benchmark=best_correlation_benchmark,
            start_date=result_start,
            end_date=result_end,
        )

    def compute_correlation_matrix(
        self,
        benchmark_symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> CorrelationMatrix | None:
        """
        Compute correlation matrix between portfolio and benchmarks.

        Args:
            benchmark_symbols: List of benchmark symbols to include.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            CorrelationMatrix or None if insufficient data.
        """
        # Get returns
        portfolio_returns = self._get_portfolio_returns(start_date, end_date)
        benchmark_returns = self._get_benchmark_returns(
            benchmark_symbols, start_date, end_date
        )

        if portfolio_returns.empty or benchmark_returns.empty:
            return None

        # Combine into single DataFrame
        portfolio_df = portfolio_returns.to_frame("Portfolio")
        combined = pd.concat([portfolio_df, benchmark_returns], axis=1, join="inner")

        if len(combined) < self.config.min_price_history_days:
            return None

        # Compute correlation matrix
        corr_matrix = combined.corr()

        # Extract portfolio correlations
        portfolio_corrs = {}
        for col in corr_matrix.columns:
            if col != "Portfolio":
                portfolio_corrs[col] = float(corr_matrix.loc["Portfolio", col])

        return CorrelationMatrix(
            matrix=corr_matrix,
            portfolio_correlations=portfolio_corrs,
        )

    def compute_rolling_beta(
        self,
        benchmark_symbol: str,
        window_days: int = 60,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> RollingBetaResult | None:
        """
        Compute rolling beta over time.

        Args:
            benchmark_symbol: Benchmark to calculate beta against.
            window_days: Rolling window size in trading days.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            RollingBetaResult or None if insufficient data.
        """
        # Get returns
        portfolio_returns = self._get_portfolio_returns(start_date, end_date)
        benchmark_returns_df = self._get_benchmark_returns(
            [benchmark_symbol], start_date, end_date
        )

        if portfolio_returns.empty or benchmark_returns_df.empty:
            return None

        if benchmark_symbol not in benchmark_returns_df.columns:
            return None

        benchmark_returns = benchmark_returns_df[benchmark_symbol]

        # Align data
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1, join="inner"
        ).dropna()

        if len(aligned) < window_days:
            return None

        port_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]

        # Calculate rolling beta
        dates = []
        betas = []

        for i in range(window_days, len(aligned) + 1):
            window_port = port_ret.iloc[i - window_days : i]
            window_bench = bench_ret.iloc[i - window_days : i]

            covariance = np.cov(window_port, window_bench)[0, 1]
            variance = np.var(window_bench, ddof=1)

            if variance > 0:
                beta = covariance / variance
                betas.append(float(beta))
                dates.append(aligned.index[i - 1])

        if not betas:
            return None

        return RollingBetaResult(
            benchmark_symbol=benchmark_symbol,
            dates=dates,
            betas=betas,
            window_days=window_days,
        )

    def compute_relative_performance(
        self,
        benchmark_symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Compute cumulative relative performance over time.

        Args:
            benchmark_symbol: Benchmark to compare against.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            DataFrame with columns [portfolio, benchmark, relative] indexed by date.
        """
        # Get returns
        portfolio_returns = self._get_portfolio_returns(start_date, end_date)
        benchmark_returns_df = self._get_benchmark_returns(
            [benchmark_symbol], start_date, end_date
        )

        if portfolio_returns.empty or benchmark_returns_df.empty:
            return None

        if benchmark_symbol not in benchmark_returns_df.columns:
            return None

        benchmark_returns = benchmark_returns_df[benchmark_symbol]

        # Align data
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1, join="inner"
        ).dropna()

        if aligned.empty:
            return None

        aligned.columns = ["portfolio", "benchmark"]

        # Calculate cumulative returns
        cumulative = (1 + aligned).cumprod() - 1
        cumulative["relative"] = cumulative["portfolio"] - cumulative["benchmark"]

        return cumulative


def compute_benchmark_comparison(
    benchmark_symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BenchmarkComparisonResult:
    """
    Convenience function for benchmark comparison.

    Args:
        benchmark_symbols: Optional list of benchmarks to compare.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        BenchmarkComparisonResult with all metrics.
    """
    analyzer = BenchmarkAnalyzer()
    return analyzer.compare_to_all_benchmarks(
        start_date=start_date,
        end_date=end_date,
        benchmark_symbols=benchmark_symbols,
    )


def compute_beta(
    benchmark_symbol: str = "SPX",
    start_date: str | None = None,
    end_date: str | None = None,
) -> float | None:
    """
    Convenience function to compute portfolio beta against a benchmark.

    Args:
        benchmark_symbol: Benchmark index symbol.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        Beta value or None if insufficient data.
    """
    analyzer = BenchmarkAnalyzer()
    metrics = analyzer.compare_to_benchmark(benchmark_symbol, start_date, end_date)
    return metrics.beta if metrics else None


def compute_correlation(
    benchmark_symbol: str = "SPX",
    start_date: str | None = None,
    end_date: str | None = None,
) -> float | None:
    """
    Convenience function to compute portfolio correlation with a benchmark.

    Args:
        benchmark_symbol: Benchmark index symbol.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        Correlation value or None if insufficient data.
    """
    analyzer = BenchmarkAnalyzer()
    metrics = analyzer.compare_to_benchmark(benchmark_symbol, start_date, end_date)
    return metrics.correlation if metrics else None


if __name__ == "__main__":
    from db import init_db

    init_db()

    print("\n📊 Benchmark Comparison Analysis")
    print("=" * 50)

    result = compute_benchmark_comparison()

    if not result.benchmarks:
        print("No benchmark data available.")
        print("Run 'python -m scripts.seed_market_indices' to populate indices.")
    else:
        print(f"\nAnalysis Period: {result.start_date} to {result.end_date}")

        for metrics in result.benchmarks:
            print(f"\n📈 {metrics.benchmark_name} ({metrics.benchmark_symbol})")
            print(f"  Portfolio Return: {metrics.portfolio_return:+.2%}")
            print(f"  Benchmark Return: {metrics.benchmark_return:+.2%}")
            print(f"  Relative Return:  {metrics.relative_return:+.2%}")

            if metrics.correlation is not None:
                print(f"  Correlation:      {metrics.correlation:.3f}")
            if metrics.beta is not None:
                print(f"  Beta:             {metrics.beta:.3f}")
            if metrics.alpha is not None:
                print(f"  Alpha (ann.):     {metrics.alpha:+.2%}")
            if metrics.tracking_error is not None:
                print(f"  Tracking Error:   {metrics.tracking_error:.2%}")
            if metrics.information_ratio is not None:
                print(f"  Information Ratio: {metrics.information_ratio:.3f}")
            if metrics.r_squared is not None:
                print(f"  R-squared:        {metrics.r_squared:.3f}")

            print(f"  Data Points:      {metrics.overlapping_days} days")

        if result.correlation_matrix:
            print("\n🔗 Correlation Matrix")
            print(result.correlation_matrix.matrix.round(3).to_string())

        if result.best_correlation_benchmark:
            print(f"\n✨ Highest correlation with: {result.best_correlation_benchmark}")
