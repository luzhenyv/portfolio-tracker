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
    ValuationAssessment,
    ValuationSignal,
    MetricBand,
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
    "ValuationAssessment",
    "ValuationSignal",
    "MetricBand",
    "run_valuation",
    "load_valuation_inputs",
    # Performance
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "AssetPerformance",
    "compute_performance_metrics",
    "get_period_returns",
]
