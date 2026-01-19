"""
Decision Engine for portfolio management.

FR-10: Decision Signals
- HOLD, REDUCE, REVIEW for owned positions
- BUY, WAIT, AVOID for watchlist

FR-11: Rule-Based Logic
- Deterministic decisions
- Based on allocation, risk, drawdown, valuation

FR-12: Explainability
- Every decision includes textual reasons
- No black-box models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import pandas as pd

from config import config
from analytics.portfolio import PortfolioAnalyzer
from analytics.risk import RiskAnalyzer


class PositionAction(str, Enum):
    """Decision actions for owned positions."""
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    REVIEW = "REVIEW"


@dataclass
class PositionDecision:
    """Decision output for a portfolio position."""
    ticker: str
    weight: float
    volatility: float | None
    max_drawdown: float | None
    action: PositionAction
    reasons: list[str] = field(default_factory=list)
    
    @property
    def reasons_text(self) -> str:
        """Formatted reasons string."""
        return "; ".join(self.reasons) if self.reasons else "No concerns"


class DecisionEngine:
    """
    Rule-based decision engine for portfolio management.
    
    FR-11: All decisions are deterministic and rule-based.
    FR-12: Every action is explainable with explicit reasons.
    """
    
    def __init__(self):
        self.config = config.decision
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
    
    def _evaluate_concentration_risk(
        self,
        weight: float,
    ) -> tuple[PositionAction | None, list[str]]:
        """
        Evaluate concentration risk based on position weight.
        
        Returns:
            Tuple of (action if triggered, reasons).
        """
        reasons = []
        
        if weight > self.config.concentration_extreme_pct:
            return PositionAction.REDUCE, [
                f"Extreme concentration ({weight:.0%} > {self.config.concentration_extreme_pct:.0%})"
            ]
        
        if weight > self.config.concentration_danger_pct:
            reasons.append(
                f"High concentration ({weight:.0%} > {self.config.concentration_danger_pct:.0%})"
            )
            return PositionAction.REVIEW, reasons
        
        if weight > self.config.concentration_warning_pct:
            reasons.append(
                f"Elevated concentration ({weight:.0%})"
            )
        
        return None, reasons
    
    def _evaluate_volatility_risk(
        self,
        weight: float,
        volatility: float | None,
    ) -> tuple[PositionAction | None, list[str]]:
        """
        Evaluate volatility risk.
        
        High concentration + high volatility = increased risk.
        """
        if volatility is None:
            return None, []
        
        reasons = []
        
        if (weight > self.config.concentration_warning_pct and 
            volatility > self.config.high_volatility_threshold):
            reasons.append(
                f"High volatility ({volatility:.0%}) with significant weight"
            )
            return PositionAction.REVIEW, reasons
        
        if volatility > self.config.high_volatility_threshold:
            reasons.append(f"High volatility ({volatility:.0%})")
        
        return None, reasons
    
    def _evaluate_drawdown_risk(
        self,
        max_drawdown: float | None,
    ) -> tuple[PositionAction | None, list[str]]:
        """
        Evaluate drawdown risk.
        
        Severe historical drawdowns warrant review.
        """
        if max_drawdown is None:
            return None, []
        
        reasons = []
        
        if max_drawdown < self.config.severe_drawdown_threshold:
            reasons.append(
                f"Severe historical drawdown ({max_drawdown:.0%})"
            )
            return PositionAction.REVIEW, reasons
        
        if max_drawdown < self.config.moderate_drawdown_threshold:
            reasons.append(
                f"Notable drawdown history ({max_drawdown:.0%})"
            )
        
        return None, reasons
    
    def evaluate_position(
        self,
        ticker: str,
        weight: float,
        volatility: float | None,
        max_drawdown: float | None,
    ) -> PositionDecision:
        """
        Evaluate a single position and generate decision.
        
        FR-11: Deterministic, rule-based evaluation.
        FR-12: All reasons are captured and returned.
        """
        all_reasons = []
        triggered_action = PositionAction.HOLD
        
        # Check concentration risk (highest priority)
        action, reasons = self._evaluate_concentration_risk(weight)
        all_reasons.extend(reasons)
        if action and action.value > triggered_action.value:
            triggered_action = action
        
        # Check volatility risk
        action, reasons = self._evaluate_volatility_risk(weight, volatility)
        all_reasons.extend(reasons)
        if action and action.value > triggered_action.value:
            triggered_action = action
        
        # Check drawdown risk
        action, reasons = self._evaluate_drawdown_risk(max_drawdown)
        all_reasons.extend(reasons)
        if action and action.value > triggered_action.value:
            triggered_action = action
        
        return PositionDecision(
            ticker=ticker,
            weight=weight,
            volatility=volatility,
            max_drawdown=max_drawdown,
            action=triggered_action,
            reasons=all_reasons,
        )
    
    def run(self) -> list[PositionDecision]:
        """
        Run decision engine for all portfolio positions.
        
        Returns:
            List of PositionDecision for each position.
        """
        # Get portfolio weights
        weights = self.portfolio_analyzer.get_portfolio_weights()
        
        if weights.empty:
            return []
        
        # Get risk metrics
        risk_metrics = self.risk_analyzer.compute_risk_metrics()
        
        decisions = []
        
        for ticker in weights.index:
            weight = weights.loc[ticker]
            volatility = risk_metrics["asset_volatility"].get(ticker)
            max_drawdown = risk_metrics["asset_max_drawdown"].get(ticker)
            
            decision = self.evaluate_position(
                ticker=ticker,
                weight=weight,
                volatility=volatility,
                max_drawdown=max_drawdown,
            )
            decisions.append(decision)
        
        # Sort by action severity (REDUCE first, then REVIEW, then HOLD)
        action_order = {PositionAction.REDUCE: 0, PositionAction.REVIEW: 1, PositionAction.HOLD: 2}
        decisions.sort(key=lambda d: (action_order[d.action], -d.weight))
        
        return decisions
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get decisions as DataFrame.
        
        Returns:
            DataFrame with decision details.
        """
        decisions = self.run()
        
        if not decisions:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "ticker": d.ticker,
                "weight": round(d.weight, 3),
                "volatility": round(d.volatility, 3) if d.volatility else None,
                "max_drawdown": round(d.max_drawdown, 3) if d.max_drawdown else None,
                "action": d.action.value,
                "reasons": d.reasons_text,
            }
            for d in decisions
        ])


def decision_engine() -> pd.DataFrame:
    """
    Convenience function for running decision engine.
    
    Returns:
        DataFrame with position decisions.
    """
    engine = DecisionEngine()
    return engine.to_dataframe()


def get_portfolio_decisions() -> list[PositionDecision]:
    """
    Get portfolio decisions as objects.
    
    Returns:
        List of PositionDecision objects.
    """
    engine = DecisionEngine()
    return engine.run()


if __name__ == "__main__":
    from db import init_db
    init_db()
    
    df = decision_engine()
    
    if df.empty:
        print("⚠️ No positions to evaluate")
    else:
        print("\n🧠 Decision Engine Output")
        print(df.to_string(index=False))
