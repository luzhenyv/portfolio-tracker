"""
Valuation analytics module.

FR-8: Valuation Data Fetching
- Forward P/E, PEG ratio, EV/EBITDA
- Revenue growth, Earnings growth

FR-9: Missing Data Handling
- Missing fields stored as NULL
- No synthetic or estimated values

Valuation signals: BUY / WAIT / AVOID (watchlist)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from config import config
from db import get_db
from db.repositories import ValuationRepository


class ValuationSignal(str, Enum):
    """Valuation-based decision signals."""
    BUY = "BUY"
    WAIT = "WAIT"
    AVOID = "AVOID"


class MetricBand(str, Enum):
    """Valuation metric classification."""
    CHEAP = "CHEAP"
    FAIR = "FAIR"
    EXPENSIVE = "EXPENSIVE"
    UNKNOWN = "UNKNOWN"


@dataclass
class ValuationAssessment:
    """Valuation assessment for an asset."""
    ticker: str
    asset_id: int
    pe_forward: float | None
    peg: float | None
    ev_ebitda: float | None
    revenue_growth: float | None
    eps_growth: float | None
    pe_band: MetricBand
    peg_band: MetricBand
    signal: ValuationSignal
    reasons: list[str]


class ValuationAnalyzer:
    """
    Analyzes valuation metrics and generates signals.
    
    FR-11: Rule-Based Logic
    - Decisions are deterministic
    - Based on valuation bands
    
    FR-12: Explainability
    - Every decision includes textual reasons
    """
    
    def __init__(self):
        self.config = config.decision
    
    def _score_pe(self, value: float | None) -> MetricBand:
        """
        Score Forward P/E metric.
        
        Bands based on conservative value investing principles.
        """
        if value is None:
            return MetricBand.UNKNOWN
        
        if value < self.config.pe_cheap_threshold:
            return MetricBand.CHEAP
        elif value <= self.config.pe_fair_threshold:
            return MetricBand.FAIR
        else:
            return MetricBand.EXPENSIVE
    
    def _score_peg(self, value: float | None) -> MetricBand:
        """
        Score PEG ratio.
        
        PEG < 1: Potentially undervalued relative to growth
        PEG 1-1.5: Fair value
        PEG > 1.5: Potentially overvalued
        """
        if value is None:
            return MetricBand.UNKNOWN
        
        if value < self.config.peg_cheap_threshold:
            return MetricBand.CHEAP
        elif value <= self.config.peg_fair_threshold:
            return MetricBand.FAIR
        else:
            return MetricBand.EXPENSIVE
    
    def _score_ev_ebitda(self, value: float | None) -> MetricBand:
        """Score EV/EBITDA ratio."""
        if value is None:
            return MetricBand.UNKNOWN
        
        if value < self.config.ev_ebitda_cheap_threshold:
            return MetricBand.CHEAP
        elif value <= self.config.ev_ebitda_fair_threshold:
            return MetricBand.FAIR
        else:
            return MetricBand.EXPENSIVE
    
    def _determine_signal(
        self,
        pe_band: MetricBand,
        peg_band: MetricBand,
        ev_ebitda_band: MetricBand,
    ) -> tuple[ValuationSignal, list[str]]:
        """
        Determine valuation signal from metric bands.
        
        FR-11: Deterministic, rule-based logic.
        FR-12: Returns reasons for explainability.
        """
        bands = [pe_band, peg_band, ev_ebitda_band]
        known_bands = [b for b in bands if b != MetricBand.UNKNOWN]
        
        if not known_bands:
            return ValuationSignal.WAIT, ["Insufficient valuation data"]
        
        reasons = []
        cheap_count = known_bands.count(MetricBand.CHEAP)
        expensive_count = known_bands.count(MetricBand.EXPENSIVE)
        
        # AVOID: Multiple expensive signals
        if expensive_count >= 2:
            if pe_band == MetricBand.EXPENSIVE:
                reasons.append("P/E indicates overvaluation")
            if peg_band == MetricBand.EXPENSIVE:
                reasons.append("PEG suggests overvalued vs growth")
            if ev_ebitda_band == MetricBand.EXPENSIVE:
                reasons.append("EV/EBITDA is elevated")
            return ValuationSignal.AVOID, reasons
        
        # BUY: Multiple cheap signals
        if cheap_count >= 2:
            if pe_band == MetricBand.CHEAP:
                reasons.append("P/E indicates attractive valuation")
            if peg_band == MetricBand.CHEAP:
                reasons.append("PEG suggests undervalued vs growth")
            if ev_ebitda_band == MetricBand.CHEAP:
                reasons.append("EV/EBITDA is attractive")
            return ValuationSignal.BUY, reasons
        
        # WAIT: Mixed or inconclusive signals
        reasons.append("Mixed valuation signals")
        if pe_band != MetricBand.UNKNOWN:
            reasons.append(f"P/E: {pe_band.value.lower()}")
        if peg_band != MetricBand.UNKNOWN:
            reasons.append(f"PEG: {peg_band.value.lower()}")
        
        return ValuationSignal.WAIT, reasons
    
    def analyze_asset(
        self,
        ticker: str,
        asset_id: int,
        pe_forward: float | None,
        peg: float | None,
        ev_ebitda: float | None,
        revenue_growth: float | None,
        eps_growth: float | None,
    ) -> ValuationAssessment:
        """
        Analyze valuation for a single asset.
        
        Args:
            ticker: Asset ticker symbol.
            asset_id: Asset database ID.
            pe_forward: Forward P/E ratio.
            peg: PEG ratio.
            ev_ebitda: EV/EBITDA ratio.
            revenue_growth: Revenue growth rate.
            eps_growth: EPS growth rate.
            
        Returns:
            ValuationAssessment with signal and reasons.
        """
        pe_band = self._score_pe(pe_forward)
        peg_band = self._score_peg(peg)
        ev_ebitda_band = self._score_ev_ebitda(ev_ebitda)
        
        signal, reasons = self._determine_signal(pe_band, peg_band, ev_ebitda_band)
        
        return ValuationAssessment(
            ticker=ticker,
            asset_id=asset_id,
            pe_forward=pe_forward,
            peg=peg,
            ev_ebitda=ev_ebitda,
            revenue_growth=revenue_growth,
            eps_growth=eps_growth,
            pe_band=pe_band,
            peg_band=peg_band,
            signal=signal,
            reasons=reasons,
        )
    
    def run_valuation(self) -> list[ValuationAssessment]:
        """
        Run valuation analysis for all assets with valuation data.
        
        Returns:
            List of ValuationAssessment for each asset.
        """
        db = get_db()
        
        with db.session() as session:
            valuation_repo = ValuationRepository(session)
            valuations = valuation_repo.get_all_with_assets()
        
        assessments = []
        for v in valuations:
            assessment = self.analyze_asset(
                ticker=v.asset.ticker,
                asset_id=v.asset_id,
                pe_forward=v.pe_forward,
                peg=v.peg,
                ev_ebitda=v.ev_ebitda,
                revenue_growth=v.revenue_growth,
                eps_growth=v.eps_growth,
            )
            assessments.append(assessment)
        
        return assessments
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get valuation analysis as DataFrame.
        
        Returns:
            DataFrame with valuation metrics and signals.
        """
        assessments = self.run_valuation()
        
        if not assessments:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "ticker": a.ticker,
                "asset_id": a.asset_id,
                "pe_forward": a.pe_forward,
                "peg": a.peg,
                "ev_ebitda": a.ev_ebitda,
                "revenue_growth": a.revenue_growth,
                "eps_growth": a.eps_growth,
                "valuation_action": a.signal.value,
                "reasons": "; ".join(a.reasons),
            }
            for a in assessments
        ])


def run_valuation() -> pd.DataFrame:
    """
    Convenience function for valuation analysis.
    
    Returns:
        DataFrame with valuation metrics and signals.
    """
    analyzer = ValuationAnalyzer()
    return analyzer.to_dataframe()


def load_valuation_inputs() -> pd.DataFrame:
    """
    Load raw valuation data (backward compatibility).
    
    Returns:
        DataFrame with valuation metrics.
    """
    return run_valuation()


if __name__ == "__main__":
    from db import init_db
    init_db()
    
    df = run_valuation()
    
    if df.empty:
        print("⚠️ No valuation data available")
    else:
        print("\n📐 Valuation Engine Output")
        display_cols = [
            "ticker", "pe_forward", "peg", "ev_ebitda",
            "revenue_growth", "eps_growth", "valuation_action"
        ]
        print(df[display_cols].to_string(index=False))
