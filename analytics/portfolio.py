"""
Portfolio analytics module.

FR-4: Portfolio Metrics
- Market value per position
- Unrealized P&L
- Portfolio allocation weight (%)
- Total portfolio value
"""

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from db import get_db
from db.repositories import PositionRepository, PriceRepository


@dataclass
class PositionMetrics:
    """Metrics for a single position."""
    ticker: str
    asset_id: int
    shares: float
    buy_price: float
    current_price: float
    cost: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float  # Portfolio allocation weight


@dataclass
class PortfolioSummary:
    """Aggregated portfolio-level metrics."""
    total_cost: float
    total_market_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    position_count: int


class PortfolioAnalyzer:
    """
    Analyzes portfolio positions and calculates metrics.
    
    FR-4 Implementation:
    - Computes market value, P&L, allocation weights
    - Handles missing price data gracefully
    """
    
    def compute_portfolio(self) -> tuple[list[PositionMetrics], PortfolioSummary]:
        """
        Compute portfolio metrics for all owned positions.
        
        Returns:
            Tuple of (list of position metrics, portfolio summary).
            
        Raises:
            ValueError: If no positions found.
        """
        db = get_db()
        
        with db.session() as session:
            position_repo = PositionRepository(session)
            price_repo = PriceRepository(session)
            
            # Get aggregated position data
            position_data = position_repo.get_position_summary()
            
            if not position_data:
                raise ValueError("No positions found in portfolio.")
            
            # Get latest prices
            asset_ids = [p["asset_id"] for p in position_data]
            latest_prices = price_repo.get_latest_prices_for_assets(asset_ids)
        
        # Calculate metrics
        positions: list[PositionMetrics] = []
        total_market_value = 0.0
        
        for pos in position_data:
            asset_id = pos["asset_id"]
            price_record = latest_prices.get(asset_id)
            current_price = price_record.close if price_record else pos["avg_buy_price"]
            
            market_value = pos["total_shares"] * current_price
            total_market_value += market_value
            
            positions.append(PositionMetrics(
                ticker=pos["ticker"],
                asset_id=asset_id,
                shares=pos["total_shares"],
                buy_price=pos["avg_buy_price"],
                current_price=current_price,
                cost=pos["total_cost"],
                market_value=market_value,
                unrealized_pnl=0.0,  # Calculated after weights
                unrealized_pnl_pct=0.0,
                weight=0.0,
            ))
        
        # Calculate P&L and weights
        total_cost = sum(p.cost for p in positions)
        total_pnl = 0.0
        
        for pos in positions:
            pos.weight = pos.market_value / total_market_value if total_market_value else 0.0
            pos.unrealized_pnl = pos.market_value - pos.cost
            pos.unrealized_pnl_pct = pos.unrealized_pnl / pos.cost if pos.cost else 0.0
            total_pnl += pos.unrealized_pnl
        
        summary = PortfolioSummary(
            total_cost=total_cost,
            total_market_value=total_market_value,
            total_unrealized_pnl=total_pnl,
            total_unrealized_pnl_pct=total_pnl / total_cost if total_cost else 0.0,
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
                "shares": p.shares,
                "buy_price": p.buy_price,
                "close": p.current_price,
                "cost": p.cost,
                "market_value": p.market_value,
                "pnl": p.unrealized_pnl,
                "pnl_pct": p.unrealized_pnl_pct,
                "weight": p.weight,
            }
            for p in positions
        ])
        
        summary_dict = {
            "total_cost": summary.total_cost,
            "total_market_value": summary.total_market_value,
            "total_pnl": summary.total_unrealized_pnl,
            "total_pnl_pct": summary.total_unrealized_pnl_pct,
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
        position_repo = PositionRepository(session)
        price_repo = PriceRepository(session)
        
        position_data = position_repo.get_position_summary()
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
        for k, v in summary.items():
            if "pct" in k:
                print(f"  {k}: {v:.1%}")
            else:
                print(f"  {k}: ${v:,.2f}")
        
        print("\n📈 Positions")
        print(df[[
            "ticker", "shares", "buy_price", "close",
            "market_value", "pnl", "pnl_pct", "weight"
        ]].to_string(index=False))
        
    except ValueError as e:
        print(f"⚠️ {e}")
