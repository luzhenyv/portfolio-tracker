"""
Performance analytics module.

Provides return calculations and performance attribution.
Designed for long-term holding review (quarters or longer).

NAV Computation:
- Daily net assets = securities market value + cash balance
- Securities value uses close prices (not adjusted_close) since dividends
  are tracked separately in the cash ledger
- Weekends/holidays forward-fill from last known price
- Position history reconstructed from Trade ledger
- Cash history reconstructed from CashTransaction ledger
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from analytics.portfolio import PortfolioAnalyzer
from analytics.risk import RiskAnalyzer
from db import get_db, AssetStatus
from db.models import TradeAction
from db.repositories import (
    AssetRepository,
    CashRepository,
    PriceRepository,
    TradeRepository,
)
from config import config


@dataclass
class PerformanceMetrics:
    """Performance metrics for a position or portfolio."""
    total_return: float
    annualized_return: float | None
    holding_period_days: int
    start_value: float
    end_value: float


@dataclass
class AssetPerformance:
    """Performance data for a single asset."""
    ticker: str
    asset_id: int
    total_return: float
    annualized_return: float | None
    holding_period_days: int
    first_price: float | None
    last_price: float | None


@dataclass
class DailyNAV:
    """
    Daily net asset value snapshot.
    
    NAV = long_value - short_value + cash
    """
    date: str
    nav: float
    long_value: float
    short_value: float
    cash: float
    # Per-asset breakdown: {ticker: {shares, price, value}}
    positions: dict = field(default_factory=dict)


@dataclass
class NAVSeries:
    """
    Time series of daily NAV with derived metrics.
    """
    daily: list[DailyNAV]
    start_date: str
    end_date: str
    start_nav: float
    end_nav: float
    total_return: float
    annualized_return: float | None
    holding_days: int


class PerformanceAnalyzer:
    """
    Calculates performance metrics for portfolio and individual positions.
    
    Focus on long-term returns appropriate for the investment philosophy.
    """
    
    def __init__(self):
        self.config = config.risk
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
    
    def _annualize_return(
        self,
        total_return: float,
        days: int,
    ) -> float | None:
        """
        Annualize a total return.
        
        Args:
            total_return: Total return (e.g., 0.25 for 25%)
            days: Holding period in days
            
        Returns:
            Annualized return or None if insufficient data.
        """
        if days < 30:  # Less than a month is not meaningful
            return None
        
        years = days / 365.0
        
        if total_return <= -1:  # Total loss
            return -1.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def get_price_returns(
        self,
        asset_id: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """
        Get price return series for an asset.
        
        Args:
            asset_id: Asset database ID.
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).
            
        Returns:
            Series of daily returns indexed by date.
        """
        db = get_db()
        
        with db.session() as session:
            price_repo = PriceRepository(session)
            prices = price_repo.get_price_history(asset_id, start_date, end_date)
        
        if not prices:
            return pd.Series(dtype=float)
        
        # Use raw close prices for returns (dividends tracked separately in cash ledger)
        df = pd.DataFrame([
            {"date": p.date, "close": p.close}
            for p in prices
            if p.close
        ])
        
        if df.empty:
            return pd.Series(dtype=float)
        
        df = df.sort_values("date")
        df["return"] = df["close"].pct_change()
        
        return df.set_index("date")["return"].dropna()
    
    def compute_asset_performance(
        self,
        asset_id: int,
        start_date: str | None = None,
    ) -> AssetPerformance | None:
        """
        Compute performance metrics for a single asset.
        
        Args:
            asset_id: Asset database ID.
            start_date: Optional start date for calculation.
            
        Returns:
            AssetPerformance or None if insufficient data.
        """
        db = get_db()
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            price_repo = PriceRepository(session)
            
            asset = asset_repo.get_by_id(asset_id)
            if not asset:
                return None
            
            prices = price_repo.get_price_history(asset_id, start_date)
        
        if len(prices) < 2:
            return None
        
        # Use raw close prices for performance (dividends tracked separately)
        first_price = prices[0].close
        last_price = prices[-1].close
        
        if not first_price or not last_price:
            return None
        
        total_return = (last_price - first_price) / first_price
        
        first_date = datetime.strptime(prices[0].date, "%Y-%m-%d")
        last_date = datetime.strptime(prices[-1].date, "%Y-%m-%d")
        holding_days = (last_date - first_date).days
        
        return AssetPerformance(
            ticker=asset.ticker,
            asset_id=asset_id,
            total_return=total_return,
            annualized_return=self._annualize_return(total_return, holding_days),
            holding_period_days=holding_days,
            first_price=first_price,
            last_price=last_price,
        )
    
    def compute_portfolio_performance(
        self,
        lookback_days: int | None = None,
    ) -> PerformanceMetrics | None:
        """
        Compute overall portfolio performance.
        
        Args:
            lookback_days: Optional lookback period (default: all history).
            
        Returns:
            PerformanceMetrics or None if insufficient data.
        """
        # Get weights
        weights = self.portfolio_analyzer.get_portfolio_weights()
        if weights.empty:
            return None
        
        # Determine date range
        start_date = None
        if lookback_days:
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Load price history
        db = get_db()
        with db.session() as session:
            price_repo = PriceRepository(session)
            records = price_repo.get_price_history_for_assets(status=AssetStatus.OWNED)
            prices = pd.DataFrame(records)

        if prices.empty:
            return None
        
        if start_date:
            prices = prices[prices["date"] >= start_date]
        
        returns = self.risk_analyzer._compute_returns(prices)
        if returns.empty:
            return None
        
        # Compute portfolio returns
        portfolio_returns = self.risk_analyzer.compute_portfolio_returns(returns, weights)
        
        # Calculate metrics
        cumulative_return = (1 + portfolio_returns).prod() - 1
        holding_days = len(portfolio_returns)
        
        return PerformanceMetrics(
            total_return=cumulative_return,
            annualized_return=self._annualize_return(cumulative_return, holding_days),
            holding_period_days=holding_days,
            start_value=1.0,  # Normalized
            end_value=1.0 + cumulative_return,
        )
    
    def get_period_returns(
        self,
        periods: list[int] = [30, 90, 180, 365],
    ) -> dict[str, float | None]:
        """
        Get portfolio returns for multiple lookback periods.
        
        Args:
            periods: List of lookback periods in days.
            
        Returns:
            Dict mapping period label to return value.
        """
        results = {}
        
        for days in periods:
            perf = self.compute_portfolio_performance(lookback_days=days)
            label = f"{days}d"
            if days >= 365:
                label = f"{days // 365}y"
            elif days >= 30:
                label = f"{days // 30}m"
            
            results[label] = perf.total_return if perf else None
        
        return results
    
    # =========================================================================
    # NAV-Based Performance (accounts for cash + position changes over time)
    # =========================================================================
    
    def _build_position_history(
        self,
        trades: list,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Reconstruct daily position (shares) from trade ledger.
        
        Returns DataFrame with columns: date, asset_id, ticker, long_shares, short_shares
        Rows represent end-of-day state after all trades on that date.
        
        For days without trades, positions carry forward from prior day.
        """
        if not trades:
            return pd.DataFrame(columns=["date", "asset_id", "ticker", "long_shares", "short_shares"])
        
        # Track position state per asset: {asset_id: {long, short, ticker}}
        positions: dict[int, dict] = {}
        # Records: list of (date, asset_id, ticker, long, short) snapshots
        records: list[tuple] = []
        
        for trade in trades:
            asset_id = trade.asset_id
            ticker = trade.asset.ticker if trade.asset else f"ID:{asset_id}"
            
            if asset_id not in positions:
                positions[asset_id] = {"long": 0.0, "short": 0.0, "ticker": ticker}
            
            pos = positions[asset_id]
            
            if trade.action == TradeAction.BUY:
                # Cover short first, then add to long
                if pos["short"] > 0:
                    cover = min(pos["short"], trade.shares)
                    pos["short"] -= cover
                    remaining = trade.shares - cover
                else:
                    remaining = trade.shares
                pos["long"] += remaining
                
            elif trade.action == TradeAction.SELL:
                # Reduce long first, then go short
                if trade.shares <= pos["long"]:
                    pos["long"] -= trade.shares
                else:
                    excess = trade.shares - pos["long"]
                    pos["long"] = 0.0
                    pos["short"] += excess
                    
            elif trade.action == TradeAction.SHORT:
                pos["short"] += trade.shares
                
            elif trade.action == TradeAction.COVER:
                pos["short"] = max(0.0, pos["short"] - trade.shares)
            
            # Record snapshot after this trade
            records.append((
                trade.trade_date,
                asset_id,
                ticker,
                pos["long"],
                pos["short"],
            ))
        
        df = pd.DataFrame(records, columns=["date", "asset_id", "ticker", "long_shares", "short_shares"])
        
        # Keep only end-of-day state per asset
        df = df.groupby(["date", "asset_id", "ticker"], as_index=False).last()
        
        return df
    
    def _build_daily_calendar(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DatetimeIndex:
        """
        Build a daily date range including weekends/holidays.
        """
        return pd.date_range(start=start_date, end=end_date, freq="D")
    
    def _forward_fill_prices(
        self,
        prices_df: pd.DataFrame,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Forward-fill prices to cover weekends/holidays.
        
        Input: DataFrame with columns [date, ticker, close]
        Output: DataFrame with all calendar dates, forward-filled prices
        """
        if prices_df.empty:
            return pd.DataFrame(columns=["date", "ticker", "close"])
        
        # Pivot to wide format: date x ticker
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        wide = prices_df.pivot(index="date", columns="ticker", values="close")
        
        # Reindex to full calendar and forward-fill
        wide = wide.reindex(calendar).ffill()
        
        # Melt back to long format
        wide = wide.reset_index().rename(columns={"index": "date"})
        long = wide.melt(id_vars=["date"], var_name="ticker", value_name="close")
        long["date"] = long["date"].dt.strftime("%Y-%m-%d")
        
        return long.dropna(subset=["close"])
    
    def _forward_fill_positions(
        self,
        position_df: pd.DataFrame,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Forward-fill positions to cover all calendar dates.
        
        Input: DataFrame with columns [date, asset_id, ticker, long_shares, short_shares]
        Output: Same columns for all calendar dates
        """
        if position_df.empty:
            return pd.DataFrame(columns=["date", "asset_id", "ticker", "long_shares", "short_shares"])
        
        # Create multi-index (date, asset_id) and forward-fill
        position_df["date"] = pd.to_datetime(position_df["date"])
        
        # Get all unique assets
        assets = position_df[["asset_id", "ticker"]].drop_duplicates()
        
        # For each asset, create full date range and forward-fill
        result_dfs = []
        for _, row in assets.iterrows():
            asset_id = row["asset_id"]
            ticker = row["ticker"]
            
            asset_df = position_df[position_df["asset_id"] == asset_id].copy()
            asset_df = asset_df.set_index("date")[["long_shares", "short_shares"]]
            
            # Reindex to full calendar and forward-fill (0 before first trade)
            asset_df = asset_df.reindex(calendar).ffill().fillna(0.0)
            asset_df = asset_df.reset_index().rename(columns={"index": "date"})
            asset_df["asset_id"] = asset_id
            asset_df["ticker"] = ticker
            
            result_dfs.append(asset_df)
        
        if not result_dfs:
            return pd.DataFrame(columns=["date", "asset_id", "ticker", "long_shares", "short_shares"])
        
        result = pd.concat(result_dfs, ignore_index=True)
        result["date"] = result["date"].dt.strftime("%Y-%m-%d")
        return result
    
    def _forward_fill_cash(
        self,
        cash_by_date: dict[str, float],
        calendar: pd.DatetimeIndex,
        initial_balance: float = 0.0,
    ) -> pd.DataFrame:
        """
        Forward-fill cash balance to cover all calendar dates.
        
        Args:
            cash_by_date: Dict mapping date string to EOD balance
            calendar: Full date range
            initial_balance: Balance before first transaction in range
            
        Returns:
            DataFrame with columns [date, cash]
        """
        if not cash_by_date:
            # No transactions - constant balance
            return pd.DataFrame({
                "date": [d.strftime("%Y-%m-%d") for d in calendar],
                "cash": [initial_balance] * len(calendar),
            })
        
        # Create series and reindex
        dates = pd.to_datetime(list(cash_by_date.keys()))
        values = list(cash_by_date.values())
        
        cash_series = pd.Series(values, index=dates)
        cash_series = cash_series.reindex(calendar).ffill()
        
        # Fill dates before first transaction with initial balance
        cash_series = cash_series.fillna(initial_balance)
        
        df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in calendar],
            "cash": cash_series.values,
        })
        return df
    
    def compute_nav_series(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> NAVSeries | None:
        """
        Compute daily NAV time series.
        
        NAV(t) = Σ(long_shares * price) - Σ(short_shares * price) + cash
        
        Uses raw close prices (not adjusted_close) since dividends are
        tracked separately in the cash ledger.
        
        For weekends/holidays, forward-fills from last known price.
        
        Args:
            start_date: Optional start date (YYYY-MM-DD). Defaults to first trade.
            end_date: Optional end date (YYYY-MM-DD). Defaults to today.
            
        Returns:
            NAVSeries with daily snapshots and aggregate metrics,
            or None if insufficient data.
        """
        db = get_db()
        
        with db.session() as session:
            trade_repo = TradeRepository(session)
            cash_repo = CashRepository(session)
            price_repo = PriceRepository(session)
            
            # Load all trades chronologically
            trades = list(trade_repo.list_all_chronological(start_date, end_date))
            
            # Determine date range
            if not start_date:
                if trades:
                    start_date = trades[0].trade_date
                else:
                    # No trades - check cash transactions
                    cash_txns = cash_repo.list_all_chronological()
                    if cash_txns:
                        start_date = cash_txns[0].transaction_date
                    else:
                        return None
            
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Build daily calendar (including weekends)
            calendar = self._build_daily_calendar(start_date, end_date)
            if len(calendar) == 0:
                return None
            
            # Build position history from trades
            position_df = self._build_position_history(trades, start_date, end_date)
            position_filled = self._forward_fill_positions(position_df, calendar)
            
            # Get unique asset_ids for price lookup
            asset_ids = list(position_filled["asset_id"].unique()) if not position_filled.empty else []
            
            # Load price history for owned assets
            price_records = price_repo.get_price_history_for_assets(asset_ids=asset_ids)
            prices_df = pd.DataFrame(price_records)
            
            # Filter to date range
            if not prices_df.empty:
                prices_df = prices_df[
                    (prices_df["date"] >= start_date) &
                    (prices_df["date"] <= end_date)
                ]
            
            # Forward-fill prices for weekends
            prices_filled = self._forward_fill_prices(prices_df, calendar)
            
            # Get cash balances
            # First get balance before start_date
            initial_cash = cash_repo.get_balance(as_of_date=start_date)
            # Subtract any transactions on start_date to get opening balance
            start_txns = cash_repo.list_all_chronological(start_date, start_date)
            for tx in start_txns:
                initial_cash -= tx.amount
            
            cash_by_date = cash_repo.get_daily_balances(start_date, end_date)
            cash_filled = self._forward_fill_cash(cash_by_date, calendar, initial_cash)
        
        # Merge positions with prices
        if not position_filled.empty and not prices_filled.empty:
            merged = position_filled.merge(
                prices_filled,
                on=["date", "ticker"],
                how="left",
            )
        else:
            merged = position_filled.copy()
            if not merged.empty:
                merged["close"] = np.nan
        
        # Compute daily NAV
        daily_navs: list[DailyNAV] = []
        
        for date_str in sorted(cash_filled["date"].unique()):
            cash_row = cash_filled[cash_filled["date"] == date_str]
            cash = cash_row["cash"].values[0] if not cash_row.empty else 0.0
            
            # Get positions for this date
            date_positions = merged[merged["date"] == date_str] if not merged.empty else pd.DataFrame()
            
            long_value = 0.0
            short_value = 0.0
            pos_details = {}
            
            for _, pos in date_positions.iterrows():
                ticker = pos["ticker"]
                long_shares = pos.get("long_shares", 0.0) or 0.0
                short_shares = pos.get("short_shares", 0.0) or 0.0
                price = pos.get("close", None)
                
                if price is not None and not np.isnan(price):
                    long_mv = long_shares * price
                    short_mv = short_shares * price
                    long_value += long_mv
                    short_value += short_mv
                    
                    if long_shares > 0 or short_shares > 0:
                        pos_details[ticker] = {
                            "long_shares": long_shares,
                            "short_shares": short_shares,
                            "price": price,
                            "long_value": long_mv,
                            "short_value": short_mv,
                        }
            
            nav = long_value - short_value + cash
            
            daily_navs.append(DailyNAV(
                date=date_str,
                nav=nav,
                long_value=long_value,
                short_value=short_value,
                cash=cash,
                positions=pos_details,
            ))
        
        if not daily_navs:
            return None
        
        # Compute aggregate metrics
        start_nav = daily_navs[0].nav
        end_nav = daily_navs[-1].nav
        holding_days = len(daily_navs)
        
        if start_nav > 0:
            total_return = (end_nav - start_nav) / start_nav
        else:
            total_return = 0.0
        
        return NAVSeries(
            daily=daily_navs,
            start_date=daily_navs[0].date,
            end_date=daily_navs[-1].date,
            start_nav=start_nav,
            end_nav=end_nav,
            total_return=total_return,
            annualized_return=self._annualize_return(total_return, holding_days),
            holding_days=holding_days,
        )
    
    def compute_nav_returns(
        self,
        lookback_days: int | None = None,
    ) -> PerformanceMetrics | None:
        """
        Compute portfolio performance from NAV series.
        
        This is a NAV-based alternative to compute_portfolio_performance()
        that accounts for actual position changes and cash flows over time.
        
        Args:
            lookback_days: Optional lookback period (default: all history)
            
        Returns:
            PerformanceMetrics or None if insufficient data
        """
        start_date = None
        if lookback_days:
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        nav_series = self.compute_nav_series(start_date=start_date)
        
        if not nav_series or len(nav_series.daily) < 2:
            return None
        
        return PerformanceMetrics(
            total_return=nav_series.total_return,
            annualized_return=nav_series.annualized_return,
            holding_period_days=nav_series.holding_days,
            start_value=nav_series.start_nav,
            end_value=nav_series.end_nav,
        )
    
    def get_nav_period_returns(
        self,
        periods: list[int] = [30, 90, 180, 365],
    ) -> dict[str, float | None]:
        """
        Get NAV-based returns for multiple lookback periods.
        
        Args:
            periods: List of lookback periods in days.
            
        Returns:
            Dict mapping period label to return value.
        """
        results = {}
        
        for days in periods:
            perf = self.compute_nav_returns(lookback_days=days)
            label = f"{days}d"
            if days >= 365:
                label = f"{days // 365}y"
            elif days >= 30:
                label = f"{days // 30}m"
            
            results[label] = perf.total_return if perf else None
        
        return results

    def get_dividend_anomalies(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """
        Identify dividend entries that may be anomalous.
        
        Anomaly checks:
        1. No holdings on dividend date (couldn't have received dividend)
        2. Implied yield is extremely high (>20% quarterly)
        3. Amount is negative (should always be positive inflow)
        4. Asset not found (orphaned dividend)
        
        Args:
            start_date: Start of period to check (YYYY-MM-DD)
            end_date: End of period to check (YYYY-MM-DD)
            
        Returns:
            List of dicts with dividend details and anomaly flags:
            - id: transaction ID
            - ticker: asset ticker (or None if orphaned)
            - amount: dividend amount
            - date: transaction date
            - anomalies: list of anomaly descriptions
            - implied_yield: computed yield vs close price (if available)
        """
        from db.repositories import CashRepository, TradeRepository
        from db.models import CashTransactionType, TradeAction
        
        db = get_db()
        anomalies = []
        
        with db.session() as session:
            asset_repo = AssetRepository(session)
            price_repo = PriceRepository(session)
            cash_repo = CashRepository(session)
            trade_repo = TradeRepository(session)
            
            # Get all dividends in range
            dividends = cash_repo.get_dividends(
                start_date=start_date,
                end_date=end_date,
            )
            
            for div in dividends:
                anomaly_list = []
                ticker = None
                implied_yield = None
                
                # Check 1: Amount must be positive
                if div.amount <= 0:
                    anomaly_list.append(f"Negative amount: ${div.amount:.2f}")
                
                # Check 2: Asset should exist
                if div.asset_id is None:
                    anomaly_list.append("No asset attribution (orphaned dividend)")
                else:
                    asset = asset_repo.get_by_id(div.asset_id)
                    if not asset:
                        anomaly_list.append(f"Asset ID {div.asset_id} not found")
                    else:
                        ticker = asset.ticker
                        
                        # Check 3: Holdings on dividend date
                        trades = trade_repo.get_trades_for_asset(
                            asset_id=div.asset_id,
                            end_date=div.transaction_date,
                        )
                        
                        shares = 0.0
                        for trade in reversed(list(trades)):
                            if trade.action == TradeAction.BUY:
                                shares += trade.shares
                            elif trade.action == TradeAction.SELL:
                                shares -= trade.shares
                            elif trade.action == TradeAction.SHORT:
                                shares -= trade.shares
                            elif trade.action == TradeAction.COVER:
                                shares += trade.shares
                        
                        if shares <= 0:
                            anomaly_list.append(
                                f"No holdings on {div.transaction_date} "
                                f"(position: {shares:.2f} shares)"
                            )
                        
                        # Check 4: Implied yield
                        prices = price_repo.get_price_history(
                            asset_id=div.asset_id,
                            start_date=div.transaction_date,
                            end_date=div.transaction_date,
                        )
                        
                        if not prices:
                            # Get most recent price before date
                            all_prices = price_repo.get_price_history(
                                asset_id=div.asset_id,
                                end_date=div.transaction_date,
                            )
                            if all_prices:
                                prices = [all_prices[-1]]
                        
                        if prices and prices[0].close:
                            price = prices[0].close
                            # Per-share yield (assuming amount is total dividend)
                            if shares > 0 and price > 0:
                                per_share_div = div.amount / shares
                                implied_yield = per_share_div / price
                                
                                # Flag extreme yields (>20% quarterly = 80% annual)
                                if implied_yield > 0.20:
                                    anomaly_list.append(
                                        f"Extreme yield: {implied_yield:.1%} per share "
                                        f"(${per_share_div:.4f} / ${price:.2f})"
                                    )
                
                # Only include if there are anomalies
                if anomaly_list:
                    anomalies.append({
                        "id": div.id,
                        "ticker": ticker,
                        "amount": div.amount,
                        "date": div.transaction_date,
                        "description": div.description,
                        "anomalies": anomaly_list,
                        "implied_yield": implied_yield,
                    })
        
        return anomalies


def compute_performance_metrics(lookback_days: int | None = None) -> PerformanceMetrics | None:
    """
    Convenience function for portfolio performance.
    
    Args:
        lookback_days: Optional lookback period.
        
    Returns:
        PerformanceMetrics or None.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.compute_portfolio_performance(lookback_days)


def get_period_returns() -> dict[str, float | None]:
    """
    Get returns for standard periods (1m, 3m, 6m, 1y).
    
    Returns:
        Dict of period returns.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.get_period_returns()


def get_dividend_anomalies(
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Get dividend entries with potential issues.
    
    Checks for:
    - No holdings on dividend date
    - Extremely high implied yields
    - Negative amounts
    - Missing asset attribution
    
    Args:
        start_date: Start of period (YYYY-MM-DD)
        end_date: End of period (YYYY-MM-DD)
        
    Returns:
        List of anomalous dividend entries with details.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.get_dividend_anomalies(start_date, end_date)


# =============================================================================
# NAV Convenience Functions
# =============================================================================

def compute_nav_series(
    start_date: str | None = None,
    end_date: str | None = None,
) -> NAVSeries | None:
    """
    Compute daily NAV time series.
    
    NAV = securities market value + cash balance
    
    Args:
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        NAVSeries with daily snapshots, or None if insufficient data.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.compute_nav_series(start_date, end_date)


def compute_nav_returns(lookback_days: int | None = None) -> PerformanceMetrics | None:
    """
    Compute portfolio performance from NAV series.
    
    Args:
        lookback_days: Optional lookback period.
        
    Returns:
        PerformanceMetrics or None.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.compute_nav_returns(lookback_days)


def get_nav_period_returns() -> dict[str, float | None]:
    """
    Get NAV-based returns for standard periods (1m, 3m, 6m, 1y).
    
    Returns:
        Dict of period returns.
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.get_nav_period_returns()


if __name__ == "__main__":
    from db import init_db
    init_db()
    
    analyzer = PerformanceAnalyzer()
    
    print("\n" + "=" * 60)
    print("📈 NAV-BASED PORTFOLIO PERFORMANCE")
    print("=" * 60)
    
    # NAV Series
    nav_series = analyzer.compute_nav_series()
    
    if nav_series:
        print(f"\n📅 Period: {nav_series.start_date} to {nav_series.end_date}")
        print(f"   Start NAV: ${nav_series.start_nav:,.2f}")
        print(f"   End NAV:   ${nav_series.end_nav:,.2f}")
        print(f"   Total Return: {nav_series.total_return:.2%}")
        if nav_series.annualized_return is not None:
            print(f"   Annualized: {nav_series.annualized_return:.2%}")
        print(f"   Days: {nav_series.holding_days}")
        
        # Show last few days
        print("\n📊 Last 5 Days:")
        for nav in nav_series.daily[-5:]:
            print(f"   {nav.date}: NAV=${nav.nav:,.2f} "
                  f"(Long=${nav.long_value:,.2f}, Cash=${nav.cash:,.2f})")
            if nav.positions:
                for ticker, pos in nav.positions.items():
                    print(f"      {ticker}: {pos['long_shares']:.2f} shares @ ${pos['price']:.2f}")
    else:
        print("\n  No NAV data available (no trades or cash transactions)")
    
    print("\n📈 NAV Period Returns:")
    nav_returns = analyzer.get_nav_period_returns()
    for period, ret in nav_returns.items():
        if ret is not None:
            print(f"   {period}: {ret:.2%}")
        else:
            print(f"   {period}: N/A")
    
    print("\n" + "-" * 60)
    print("📊 LEGACY WEIGHT-BASED PERFORMANCE (for comparison)")
    print("-" * 60)
    
    perf = analyzer.compute_portfolio_performance()
    
    if perf:
        print(f"  Total Return: {perf.total_return:.2%}")
        if perf.annualized_return is not None:
            print(f"  Annualized Return: {perf.annualized_return:.2%}")
        print(f"  Holding Period: {perf.holding_period_days} days")
    else:
        print("  Insufficient data")
    
    print("\n📊 Period Returns (weight-based):")
    period_returns = analyzer.get_period_returns()
    for period, ret in period_returns.items():
        if ret is not None:
            print(f"   {period}: {ret:.2%}")
        else:
            print(f"   {period}: N/A")
    
    print("\n🔍 Dividend Anomalies")
    anomalies = analyzer.get_dividend_anomalies()
    if anomalies:
        for a in anomalies:
            print(f"  ⚠️ {a['ticker'] or 'Unknown'} on {a['date']}: ${a['amount']:.2f}")
            for issue in a['anomalies']:
                print(f"     - {issue}")
    else:
        print("  No anomalies found")

