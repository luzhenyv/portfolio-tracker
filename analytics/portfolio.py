"""
Portfolio analytics module.

FR-4: Portfolio Metrics
- Market value per position (long and short)
- Unrealized P&L (long and short)
- Portfolio allocation weights (gross and net exposure)
- Total portfolio value (long, short, net, gross)
- Realized P&L tracking
"""

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from db import get_db
from db.repositories import PositionRepository, PriceRepository, TradeRepository


@dataclass
class PositionMetrics:
    """Metrics for a single position (long/short)."""
    ticker: str
    asset_id: int
    long_shares: float
    long_avg_cost: float
    short_shares: float
    short_avg_price: float
    current_price: float
    # Long exposure
    long_cost: float
    long_market_value: float
    long_unrealized_pnl: float
    # Short exposure
    short_cost: float  # Average entry price * shares
    short_market_value: float  # Absolute market value
    short_unrealized_pnl: float
    # Net/Gross
    net_shares: float
    net_exposure: float
    gross_exposure: float
    total_unrealized_pnl: float
    realized_pnl: float
    # Weights
    gross_weight: float
    net_weight: float


@dataclass
class PortfolioSummary:
    """Aggregated portfolio-level metrics."""
    # Long exposure
    long_total_cost: float
    long_market_value: float
    long_unrealized_pnl: float
    # Short exposure
    short_total_cost: float
    short_market_value: float
    short_unrealized_pnl: float
    # Net/Gross/Total
    gross_exposure: float
    net_exposure: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float  # Realized + Unrealized
    position_count: int


class PortfolioAnalyzer:
    """
    Analyzes portfolio positions and calculates metrics.
    
    Supports long and short positions with:
    - Average cost method for P&L
    - Gross and net exposure calculations
    - Realized and unrealized P&L tracking
    """
    
    def compute_portfolio(self) -> tuple[list[PositionMetrics], PortfolioSummary]:
        """
        Compute portfolio metrics for all active positions (long/short).
        
        Returns:
            Tuple of (list of position metrics, portfolio summary).
            
        Raises:
            ValueError: If no positions found.
        """
        db = get_db()
        
        with db.session() as session:
            position_repo = PositionRepository(session)
            price_repo = PriceRepository(session)
            trade_repo = TradeRepository(session)
            
            # Get position state data
            position_data = position_repo.get_position_summary()
            
            if not position_data:
                raise ValueError("No positions found in portfolio.")
            
            # Get latest prices
            asset_ids = [p["asset_id"] for p in position_data]
            latest_prices = price_repo.get_latest_prices_for_assets(asset_ids)
            
            # Get realized P&L summary
            realized_summary = trade_repo.get_realized_pnl_summary()
        
        # Calculate metrics per position
        positions: list[PositionMetrics] = []
        
        # Accumulators
        total_gross_exposure = 0.0
        long_mv_sum = 0.0
        short_mv_sum = 0.0
        long_cost_sum = 0.0
        short_cost_sum = 0.0
        long_unrl_sum = 0.0
        short_unrl_sum = 0.0
        
        for pos in position_data:
            asset_id = pos["asset_id"]
            price_record = latest_prices.get(asset_id)
            
            # Use latest price or fall back to avg cost for long, avg price for short
            if price_record:
                current_price = price_record.close
            else:
                # Fallback: use long avg cost if long, or short avg price if only short
                if pos["long_shares"] > 0:
                    current_price = pos["long_avg_cost"]
                elif pos["short_shares"] > 0:
                    current_price = pos["short_avg_price"]
                else:
                    current_price = 0.0
            
            # Long calculations
            long_shares = pos["long_shares"]
            long_avg_cost = pos["long_avg_cost"]
            long_cost = long_shares * long_avg_cost if long_shares > 0 else 0.0
            long_mv = long_shares * current_price
            long_unrl = long_mv - long_cost
            
            # Short calculations
            short_shares = pos["short_shares"]
            short_avg_price = pos["short_avg_price"]
            short_cost = short_shares * short_avg_price if short_shares > 0 else 0.0
            short_mv = short_shares * current_price  # Absolute exposure
            short_unrl = short_cost - short_mv  # Profit when price drops
            
            # Net and gross
            net_shares = long_shares - short_shares
            net_exposure = long_mv - short_mv
            gross_exposure = long_mv + short_mv
            total_unrl = long_unrl + short_unrl
            
            # Accumulate for weights
            total_gross_exposure += gross_exposure
            long_mv_sum += long_mv
            short_mv_sum += short_mv
            long_cost_sum += long_cost
            short_cost_sum += short_cost
            long_unrl_sum += long_unrl
            short_unrl_sum += short_unrl
            
            positions.append(PositionMetrics(
                ticker=pos["ticker"],
                asset_id=asset_id,
                long_shares=long_shares,
                long_avg_cost=long_avg_cost,
                short_shares=short_shares,
                short_avg_price=short_avg_price,
                current_price=current_price,
                long_cost=long_cost,
                long_market_value=long_mv,
                long_unrealized_pnl=long_unrl,
                short_cost=short_cost,
                short_market_value=short_mv,
                short_unrealized_pnl=short_unrl,
                net_shares=net_shares,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                total_unrealized_pnl=total_unrl,
                realized_pnl=pos["realized_pnl"],
                gross_weight=0.0,  # Calculated below
                net_weight=0.0,
            ))
        
        # Calculate weights
        for pos in positions:
            pos.gross_weight = pos.gross_exposure / total_gross_exposure if total_gross_exposure else 0.0
            # Net weight normalized by gross to avoid divide-by-small-net issues
            pos.net_weight = pos.net_exposure / total_gross_exposure if total_gross_exposure else 0.0
        
        summary = PortfolioSummary(
            long_total_cost=long_cost_sum,
            long_market_value=long_mv_sum,
            long_unrealized_pnl=long_unrl_sum,
            short_total_cost=short_cost_sum,
            short_market_value=short_mv_sum,
            short_unrealized_pnl=short_unrl_sum,
            gross_exposure=total_gross_exposure,
            net_exposure=long_mv_sum - short_mv_sum,
            total_unrealized_pnl=long_unrl_sum + short_unrl_sum,
            total_realized_pnl=realized_summary["net_realized_pnl"],
            total_pnl=long_unrl_sum + short_unrl_sum + realized_summary["net_realized_pnl"],
            position_count=len(positions),
        )
        
        return positions, summary
    
    def to_dataframe(self) -> tuple[pd.DataFrame, dict]:
        """
        Get portfolio as DataFrame for display/analysis.
        
        Returns:
            Tuple of (positions DataFrame, summary dict).
        """
        positions, summary = self.compute_portfolio()
        
        df = pd.DataFrame([
            {
                "ticker": p.ticker,
                "asset_id": p.asset_id,
                "long_shares": p.long_shares,
                "short_shares": p.short_shares,
                "net_shares": p.net_shares,
                "close": p.current_price,
                "long_mv": p.long_market_value,
                "short_mv": p.short_market_value,
                "gross": p.gross_exposure,
                "net": p.net_exposure,
                "pnl": p.total_unrealized_pnl,
                "realized_pnl": p.realized_pnl,
                "gross_weight": p.gross_weight,
                "net_weight": p.net_weight,
            }
            for p in positions
        ])
        
        # Calculate total cost and market value
        total_cost = summary.long_total_cost + summary.short_total_cost
        total_market_value = summary.long_market_value - summary.short_market_value  # Net value
        total_pnl_pct = (summary.total_pnl / total_cost) if total_cost > 0 else 0.0
        
        summary_dict = {
            "long_cost": summary.long_total_cost,
            "long_mv": summary.long_market_value,
            "long_pnl": summary.long_unrealized_pnl,
            "short_cost": summary.short_total_cost,
            "short_mv": summary.short_market_value,
            "short_pnl": summary.short_unrealized_pnl,
            "gross_exposure": summary.gross_exposure,
            "net_exposure": summary.net_exposure,
            "total_unrealized_pnl": summary.total_unrealized_pnl,
            "total_realized_pnl": summary.total_realized_pnl,
            "total_pnl": summary.total_pnl,
            # Additional keys for UI compatibility
            "total_cost": total_cost,
            "total_market_value": total_market_value,
            "total_pnl_pct": total_pnl_pct,
        }
        
        return df, summary_dict


def compute_portfolio() -> tuple[pd.DataFrame, dict]:
    """
    Convenience function for computing portfolio metrics.
    
    Returns:
        Tuple of (positions DataFrame, summary dict).
    """
    analyzer = PortfolioAnalyzer()
    return analyzer.to_dataframe()


def load_positions() -> pd.DataFrame:
    """
    Load position data as DataFrame.
    
    Returns:
        DataFrame with position details.
    """
    df, _ = compute_portfolio()
    return df


def load_latest_prices() -> pd.DataFrame:
    """
    Load latest prices for all owned assets.
    
    Returns:
        DataFrame with asset_id, date, close columns.
    """
    db = get_db()
    
    with db.session() as session:
        state_repo = PositionRepository(session)
        price_repo = PriceRepository(session)
        
        position_data = state_repo.get_position_summary()
        asset_ids = [p["asset_id"] for p in position_data]
        latest_prices = price_repo.get_latest_prices_for_assets(asset_ids)
    
    records = [
        {
            "asset_id": asset_id,
            "date": price.date,
            "close": price.close,
        }
        for asset_id, price in latest_prices.items()
    ]
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    from db import init_db
    init_db()
    
    try:
        df, summary = compute_portfolio()
        
        print("\n📊 Portfolio Summary")
        print(f"  Long MV: ${summary['long_mv']:,.2f}")
        print(f"  Short MV: ${summary['short_mv']:,.2f}")
        print(f"  Gross Exposure: ${summary['gross_exposure']:,.2f}")
        print(f"  Net Exposure: ${summary['net_exposure']:,.2f}")
        print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:+,.2f}")
        print(f"  Realized P&L: ${summary['total_realized_pnl']:+,.2f}")
        print(f"  Total P&L: ${summary['total_pnl']:+,.2f}")
        
        print("\n📈 Positions")
        print(df[[
            "ticker", "long_shares", "short_shares", "net_shares",
            "close", "gross", "pnl", "realized_pnl", "gross_weight"
        ]].to_string(index=False))
        
    except ValueError as e:
        print(f"⚠️ {e}")
