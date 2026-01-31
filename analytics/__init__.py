"""
Analytics package initialization.

Exports commonly used analytics functions for convenient imports:
    from analytics import compute_portfolio, compute_risk_metrics, etc.
"""

from analytics.portfolio import (
    PortfolioAnalyzer,
    PositionMetrics,
    PortfolioSummary,
    compute_portfolio,
    load_positions,
    load_latest_prices,
    portfolio_weights,
    portfolio_weights_cost_based,
)
from analytics.risk import (
    RiskAnalyzer,
    AssetRiskMetrics,
    PortfolioRiskMetrics,
    compute_risk_metrics,
)
from analytics.valuation import (
    ValuationAnalyzer,
    YahooValuationData,
    ValuationSignal,
    MetricBand,
    YAHOO_VALUATION_FIELDS,
    run_valuation,
    load_valuation_inputs,
)
from analytics.performance import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    AssetPerformance,
    compute_performance_metrics,
    get_period_returns,
)
from analytics.benchmark import (
    BenchmarkAnalyzer,
    BenchmarkMetrics,
    BenchmarkComparisonResult,
    CorrelationMatrix,
    RollingBetaResult,
    compute_benchmark_comparison,
    compute_beta,
    compute_correlation,
)
from analytics.optimizer import (
    PortfolioOptimizer,
    OptimalPortfolio,
    EfficientFrontierResult,
    RebalanceRecommendation,
    compute_efficient_frontier,
    get_rebalance_recommendations,
)

__all__ = [
    # Portfolio
    "PortfolioAnalyzer",
    "PositionMetrics",
    "PortfolioSummary",
    "compute_portfolio",
    "load_positions",
    "load_latest_prices",
    "portfolio_weights",
    "portfolio_weights_cost_based",
    # Risk
    "RiskAnalyzer",
    "AssetRiskMetrics",
    "PortfolioRiskMetrics",
    "compute_risk_metrics",
    # Valuation
    "ValuationAnalyzer",
    "YahooValuationData",
    "ValuationSignal",
    "MetricBand",
    "YAHOO_VALUATION_FIELDS",
    "run_valuation",
    "load_valuation_inputs",
    # Performance
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "AssetPerformance",
    "compute_performance_metrics",
    "get_period_returns",
    # Benchmark
    "BenchmarkAnalyzer",
    "BenchmarkMetrics",
    "BenchmarkComparisonResult",
    "CorrelationMatrix",
    "RollingBetaResult",
    "compute_benchmark_comparison",
    "compute_beta",
    "compute_correlation",
    # Optimizer
    "PortfolioOptimizer",
    "OptimalPortfolio",
    "EfficientFrontierResult",
    "RebalanceRecommendation",
    "compute_efficient_frontier",
    "get_rebalance_recommendations",
]
