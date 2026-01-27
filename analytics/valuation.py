"""
Valuation analytics module - Yahoo Finance aligned.

Provides Yahoo-style "Statistics" page metrics:
- Valuation Measures: Market Cap, EV, P/E, PEG, P/S, P/B, EV/Rev, EV/EBITDA
- Financial Highlights: Margins, ROA, ROE, Revenue, Net Income, EPS, Cash, Debt, FCF

FR-8: Valuation Data Fetching from yfinance Ticker.info
FR-9: Missing fields stored as NULL, no synthetic values

Valuation signals: BUY / WAIT / AVOID (watchlist)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd

from config import config
from db import get_db
from db.repositories import ValuationRepository, ValuationOverrideRepository


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


# Define all Yahoo-aligned metric fields
YAHOO_VALUATION_FIELDS = [
    # Valuation Measures
    "market_cap",
    "enterprise_value",
    "pe_trailing",
    "pe_forward",
    "peg",
    "price_to_sales",
    "price_to_book",
    "ev_to_revenue",
    "ev_ebitda",
    # Financial Highlights - Profitability
    "profit_margin",
    "return_on_assets",
    "return_on_equity",
    # Financial Highlights - Income Statement
    "revenue_ttm",
    "net_income_ttm",
    "diluted_eps_ttm",
    # Financial Highlights - Balance Sheet & Cash Flow
    "total_cash",
    "total_debt_to_equity",
    "levered_free_cash_flow",
]


@dataclass
class YahooValuationData:
    """Yahoo-aligned valuation data for an asset."""
    ticker: str
    asset_id: int
    updated_at: datetime | None = None
    
    # Valuation Measures
    market_cap: float | None = None
    enterprise_value: float | None = None
    pe_trailing: float | None = None
    pe_forward: float | None = None
    peg: float | None = None
    price_to_sales: float | None = None
    price_to_book: float | None = None
    ev_to_revenue: float | None = None
    ev_ebitda: float | None = None
    
    # Financial Highlights - Profitability
    profit_margin: float | None = None
    return_on_assets: float | None = None
    return_on_equity: float | None = None
    
    # Financial Highlights - Income Statement
    revenue_ttm: float | None = None
    net_income_ttm: float | None = None
    diluted_eps_ttm: float | None = None
    
    # Financial Highlights - Balance Sheet & Cash Flow
    total_cash: float | None = None
    total_debt_to_equity: float | None = None
    levered_free_cash_flow: float | None = None
    
    # Override flags - True if user override is applied
    market_cap_overridden: bool = False
    enterprise_value_overridden: bool = False
    pe_trailing_overridden: bool = False
    pe_forward_overridden: bool = False
    peg_overridden: bool = False
    price_to_sales_overridden: bool = False
    price_to_book_overridden: bool = False
    ev_to_revenue_overridden: bool = False
    ev_ebitda_overridden: bool = False
    profit_margin_overridden: bool = False
    return_on_assets_overridden: bool = False
    return_on_equity_overridden: bool = False
    revenue_ttm_overridden: bool = False
    net_income_ttm_overridden: bool = False
    diluted_eps_ttm_overridden: bool = False
    total_cash_overridden: bool = False
    total_debt_to_equity_overridden: bool = False
    levered_free_cash_flow_overridden: bool = False
    
    # Signal (derived from valuation metrics)
    pe_band: MetricBand = MetricBand.UNKNOWN
    peg_band: MetricBand = MetricBand.UNKNOWN
    signal: ValuationSignal = ValuationSignal.WAIT
    reasons: list[str] = field(default_factory=list)


class ValuationAnalyzer:
    """
    Analyzes valuation metrics and generates signals.
    
    Yahoo-aligned output with all Statistics page fields.
    
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
    
    def _get_effective_value(
        self,
        fetched: float | None,
        override: float | None,
    ) -> tuple[float | None, bool]:
        """
        Get effective value and override flag.
        
        Returns:
            Tuple of (effective_value, is_overridden)
        """
        if override is not None:
            return override, True
        return fetched, False
    
    def run_valuation(self) -> list[YahooValuationData]:
        """
        Run valuation analysis for all assets with valuation data.
        
        Merges user overrides with fetched values. For each metric:
        effective_value = override if present else fetched_value
        
        Returns:
            List of YahooValuationData for each asset.
        """
        db = get_db()
        
        with db.session() as session:
            valuation_repo = ValuationRepository(session)
            override_repo = ValuationOverrideRepository(session)
            
            valuations = valuation_repo.get_all_with_assets()
            
            # Get all overrides for assets with valuations
            asset_ids = [v.asset_id for v in valuations]
            overrides = override_repo.get_by_asset_ids(asset_ids)
        
        results = []
        for v in valuations:
            override = overrides.get(v.asset_id)
            
            # Build effective values for all fields
            data = YahooValuationData(
                ticker=v.asset.ticker,
                asset_id=v.asset_id,
                updated_at=v.updated_at,
            )
            
            # Valuation Measures - compute effective values
            data.market_cap, data.market_cap_overridden = self._get_effective_value(
                v.market_cap, override.market_cap_override if override else None
            )
            data.enterprise_value, data.enterprise_value_overridden = self._get_effective_value(
                v.enterprise_value, override.enterprise_value_override if override else None
            )
            data.pe_trailing, data.pe_trailing_overridden = self._get_effective_value(
                v.pe_trailing, override.pe_trailing_override if override else None
            )
            data.pe_forward, data.pe_forward_overridden = self._get_effective_value(
                v.pe_forward, override.pe_forward_override if override else None
            )
            data.peg, data.peg_overridden = self._get_effective_value(
                v.peg, override.peg_override if override else None
            )
            data.price_to_sales, data.price_to_sales_overridden = self._get_effective_value(
                v.price_to_sales, override.price_to_sales_override if override else None
            )
            data.price_to_book, data.price_to_book_overridden = self._get_effective_value(
                v.price_to_book, override.price_to_book_override if override else None
            )
            data.ev_to_revenue, data.ev_to_revenue_overridden = self._get_effective_value(
                v.ev_to_revenue, override.ev_to_revenue_override if override else None
            )
            data.ev_ebitda, data.ev_ebitda_overridden = self._get_effective_value(
                v.ev_ebitda, override.ev_ebitda_override if override else None
            )
            
            # Financial Highlights - Profitability
            data.profit_margin, data.profit_margin_overridden = self._get_effective_value(
                v.profit_margin, override.profit_margin_override if override else None
            )
            data.return_on_assets, data.return_on_assets_overridden = self._get_effective_value(
                v.return_on_assets, override.return_on_assets_override if override else None
            )
            data.return_on_equity, data.return_on_equity_overridden = self._get_effective_value(
                v.return_on_equity, override.return_on_equity_override if override else None
            )
            
            # Financial Highlights - Income Statement
            data.revenue_ttm, data.revenue_ttm_overridden = self._get_effective_value(
                v.revenue_ttm, override.revenue_ttm_override if override else None
            )
            data.net_income_ttm, data.net_income_ttm_overridden = self._get_effective_value(
                v.net_income_ttm, override.net_income_ttm_override if override else None
            )
            data.diluted_eps_ttm, data.diluted_eps_ttm_overridden = self._get_effective_value(
                v.diluted_eps_ttm, override.diluted_eps_ttm_override if override else None
            )
            
            # Financial Highlights - Balance Sheet & Cash Flow
            data.total_cash, data.total_cash_overridden = self._get_effective_value(
                v.total_cash, override.total_cash_override if override else None
            )
            data.total_debt_to_equity, data.total_debt_to_equity_overridden = self._get_effective_value(
                v.total_debt_to_equity, override.total_debt_to_equity_override if override else None
            )
            data.levered_free_cash_flow, data.levered_free_cash_flow_overridden = self._get_effective_value(
                v.levered_free_cash_flow, override.levered_free_cash_flow_override if override else None
            )
            
            # Compute bands and signal
            data.pe_band = self._score_pe(data.pe_forward)
            data.peg_band = self._score_peg(data.peg)
            ev_ebitda_band = self._score_ev_ebitda(data.ev_ebitda)
            
            data.signal, data.reasons = self._determine_signal(
                data.pe_band, data.peg_band, ev_ebitda_band
            )
            
            results.append(data)
        
        return results
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get valuation analysis as DataFrame.
        
        Returns:
            DataFrame with Yahoo-aligned valuation metrics and signals.
        """
        assessments = self.run_valuation()
        
        if not assessments:
            return pd.DataFrame()
        
        rows = []
        for a in assessments:
            row = {
                "ticker": a.ticker,
                "asset_id": a.asset_id,
                "updated_at": a.updated_at,
                # Valuation Measures
                "market_cap": a.market_cap,
                "enterprise_value": a.enterprise_value,
                "pe_trailing": a.pe_trailing,
                "pe_forward": a.pe_forward,
                "peg": a.peg,
                "price_to_sales": a.price_to_sales,
                "price_to_book": a.price_to_book,
                "ev_to_revenue": a.ev_to_revenue,
                "ev_ebitda": a.ev_ebitda,
                # Financial Highlights - Profitability
                "profit_margin": a.profit_margin,
                "return_on_assets": a.return_on_assets,
                "return_on_equity": a.return_on_equity,
                # Financial Highlights - Income Statement
                "revenue_ttm": a.revenue_ttm,
                "net_income_ttm": a.net_income_ttm,
                "diluted_eps_ttm": a.diluted_eps_ttm,
                # Financial Highlights - Balance Sheet & Cash Flow
                "total_cash": a.total_cash,
                "total_debt_to_equity": a.total_debt_to_equity,
                "levered_free_cash_flow": a.levered_free_cash_flow,
                # Signal
                "valuation_action": a.signal.value,
                "reasons": "; ".join(a.reasons),
                # Override flags
                "market_cap_overridden": a.market_cap_overridden,
                "enterprise_value_overridden": a.enterprise_value_overridden,
                "pe_trailing_overridden": a.pe_trailing_overridden,
                "pe_forward_overridden": a.pe_forward_overridden,
                "peg_overridden": a.peg_overridden,
                "price_to_sales_overridden": a.price_to_sales_overridden,
                "price_to_book_overridden": a.price_to_book_overridden,
                "ev_to_revenue_overridden": a.ev_to_revenue_overridden,
                "ev_ebitda_overridden": a.ev_ebitda_overridden,
                "profit_margin_overridden": a.profit_margin_overridden,
                "return_on_assets_overridden": a.return_on_assets_overridden,
                "return_on_equity_overridden": a.return_on_equity_overridden,
                "revenue_ttm_overridden": a.revenue_ttm_overridden,
                "net_income_ttm_overridden": a.net_income_ttm_overridden,
                "diluted_eps_ttm_overridden": a.diluted_eps_ttm_overridden,
                "total_cash_overridden": a.total_cash_overridden,
                "total_debt_to_equity_overridden": a.total_debt_to_equity_overridden,
                "levered_free_cash_flow_overridden": a.levered_free_cash_flow_overridden,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def run_valuation() -> pd.DataFrame:
    """
    Convenience function for valuation analysis.
    
    Returns:
        DataFrame with Yahoo-aligned valuation metrics and signals.
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
        print("\n📐 Yahoo-Aligned Valuation Output")
        display_cols = [
            "ticker", "market_cap", "pe_trailing", "pe_forward", 
            "peg", "ev_ebitda", "profit_margin", "valuation_action"
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
