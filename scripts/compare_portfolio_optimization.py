"""
Portfolio Optimization: Monte Carlo vs Mathematical Optimization

This script demonstrates two approaches to finding the optimal portfolio:
1. Monte Carlo Simulation - Random sampling of portfolio weights
2. scipy.optimize - Mathematical optimization using SLSQP

Both methods use Modern Portfolio Theory (MPT) to maximize the Sharpe Ratio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import optimize

from db import init_db, get_db
from db.repositories import PriceRepository


# =============================================================================
# Configuration
# =============================================================================
TICKERS = ["AVGO", "GOOGL", "HIMS", "META", "MRVL", "TSLA", "NVDA"]
YEAR_BACKS = 3  # Match config.data_fetcher.default_lookback_days (3 years)
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0366  # 3.66% annual risk-free rate
NUM_PORTFOLIOS = 10000   # Number of Monte Carlo simulations

# Toggle between data sources
USE_LOCAL_DB = False  # Set to False to use yfinance directly


# =============================================================================
# Data Loading
# =============================================================================
def load_stock_data_from_db(tickers: list[str]) -> pd.DataFrame:
    """
    Load stock price data from local database and compute log returns.
    
    Uses the same data source as the web app for consistent results.
    Uses raw 'close' prices (not adjusted_close) matching the web app behavior.
    """
    db = get_db()
    with db.session() as session:
        price_repo = PriceRepository(session)
        # Get price history for specific tickers
        records = price_repo.get_price_history_for_assets()
        
    prices_df = pd.DataFrame(records)
    
    if prices_df.empty:
        raise ValueError("No price data found in database")
    
    # Filter to only requested tickers
    prices_df = prices_df[prices_df["ticker"].isin(tickers)]
    
    # Pivot: date as index, tickers as columns, close prices as values
    pivot = prices_df.pivot(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()
    
    # Only keep tickers that have data
    available_tickers = [t for t in tickers if t in pivot.columns]
    pivot = pivot[available_tickers]
    
    print(f"📥 Loading {len(available_tickers)} stocks from local database...")
    print(f"   Date range: {pivot.index[0]} to {pivot.index[-1]}")
    
    # Compute log returns
    log_returns = np.log(pivot / pivot.shift(1)).dropna()
    
    print(f"✅ Loaded {len(log_returns)} trading days of data\n")
    return log_returns


def load_stock_data_from_yfinance(tickers: list[str], years: int) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance and compute log returns.
    
    Log returns are preferred over simple returns because:
    - They are time-additive (can sum across periods)
    - They are approximately normally distributed
    - They handle compounding correctly
    
    NOTE: yfinance returns adjusted close by default (auto_adjust=True).
    To match local DB behavior, we use auto_adjust=False and fetch raw Close.
    """
    start_date = datetime.today() - timedelta(days=365 * years)
    end_date = datetime.today()
    
    print(f"📥 Downloading {len(tickers)} stocks from {start_date.date()} to {end_date.date()}...")
    
    # Use auto_adjust=False to get raw prices like the local DB stores
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    prices = data["Close"]  # Raw close prices (not adjusted)
    
    # Use LOG RETURNS instead of simple percentage returns
    # log_return = ln(P_t / P_{t-1})
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    print(f"✅ Loaded {len(log_returns)} trading days of data\n")
    return log_returns


def load_stock_data(tickers: list[str], years: int) -> pd.DataFrame:
    """
    Load stock data from configured source (local DB or yfinance).
    """
    if USE_LOCAL_DB:
        return load_stock_data_from_db(tickers)
    else:
        return load_stock_data_from_yfinance(tickers, years)


# =============================================================================
# Portfolio Statistics
# =============================================================================
def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, 
                    cov_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate portfolio expected return, volatility, and Sharpe ratio.
    
    Args:
        weights: Portfolio weights (must sum to 1)
        mean_returns: Annualized expected returns for each asset
        cov_matrix: Annualized covariance matrix
    
    Returns:
        Tuple of (expected_return, volatility, sharpe_ratio)
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


def negative_sharpe(weights: np.ndarray, mean_returns: np.ndarray, 
                    cov_matrix: np.ndarray) -> float:
    """Objective function: negative Sharpe ratio (for minimization)."""
    _, _, sharpe = portfolio_stats(weights, mean_returns, cov_matrix)
    return -sharpe


# =============================================================================
# Method 1: Monte Carlo Simulation
# =============================================================================
def monte_carlo_optimization(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                              num_assets: int, num_portfolios: int) -> dict:
    """
    Find optimal portfolio using Monte Carlo simulation.
    
    Generates random portfolio weights and tracks the best Sharpe ratio found.
    This is a brute-force approach that samples the solution space.
    
    Pros:
    - Easy to understand and implement
    - Visualizes the entire efficient frontier
    - No gradient requirements
    
    Cons:
    - May miss the true optimal (depends on sample size)
    - Computationally expensive for high precision
    - Results vary between runs (unless seeded)
    """
    print("🎲 Running Monte Carlo Simulation...")
    
    port_returns = np.zeros(num_portfolios)
    port_volatility = np.zeros(num_portfolios)
    sharpe_ratios = np.zeros(num_portfolios)
    all_weights = np.zeros((num_portfolios, num_assets))
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_portfolios):
        # Generate random weights using Dirichlet distribution (ensures sum = 1)
        weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
        all_weights[i, :] = weights
        
        ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix)
        port_returns[i] = ret
        port_volatility[i] = vol
        sharpe_ratios[i] = sharpe
    
    # Find best portfolio from simulation
    max_sharpe_idx = np.argmax(sharpe_ratios)
    optimal_weights = all_weights[max_sharpe_idx, :]
    optimal_return = port_returns[max_sharpe_idx]
    optimal_volatility = port_volatility[max_sharpe_idx]
    optimal_sharpe = sharpe_ratios[max_sharpe_idx]
    
    print(f"   Best Sharpe found: {optimal_sharpe:.4f}")
    print(f"   Portfolios sampled: {num_portfolios:,}\n")
    
    return {
        "weights": optimal_weights,
        "return": optimal_return,
        "volatility": optimal_volatility,
        "sharpe": optimal_sharpe,
        "all_returns": port_returns,
        "all_volatility": port_volatility,
        "all_sharpe": sharpe_ratios,
    }


# =============================================================================
# Method 2: Mathematical Optimization (scipy.optimize)
# =============================================================================
def scipy_optimization(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        num_assets: int) -> dict:
    """
    Find optimal portfolio using scipy.optimize.minimize.
    
    Uses Sequential Least Squares Programming (SLSQP) to find the
    mathematically optimal weights that maximize the Sharpe ratio.
    
    Pros:
    - Finds the TRUE optimal solution
    - Fast and efficient
    - Deterministic (same result every time)
    
    Cons:
    - May find local optima (though Sharpe optimization is typically convex)
    - Requires smooth objective function
    - Doesn't visualize the frontier by itself
    """
    print("📐 Running Mathematical Optimization (scipy.optimize)...")
    
    # Initial guess: equal weights
    init_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Constraint: weights must sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    
    # Bounds: long-only portfolio (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Minimize negative Sharpe ratio = Maximize Sharpe ratio
    result = optimize.minimize(
        negative_sharpe,
        init_weights,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    optimal_weights = result.x
    optimal_return, optimal_volatility, optimal_sharpe = portfolio_stats(
        optimal_weights, mean_returns, cov_matrix
    )
    
    print(f"   Optimal Sharpe: {optimal_sharpe:.4f}")
    print(f"   Optimization converged: {result.success}\n")
    
    return {
        "weights": optimal_weights,
        "return": optimal_return,
        "volatility": optimal_volatility,
        "sharpe": optimal_sharpe,
    }


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Initialize database connection if using local DB
    if USE_LOCAL_DB:
        init_db()
    
    # Load data and compute statistics
    returns = load_stock_data(TICKERS, YEAR_BACKS)
    
    # Use only tickers that have data (column names from returns DataFrame)
    available_tickers = returns.columns.tolist()
    
    mean_returns = returns.mean() * TRADING_DAYS  # Annualize
    cov_matrix = returns.cov() * TRADING_DAYS      # Annualize
    num_assets = len(available_tickers)
    
    # Run both optimization methods
    mc_result = monte_carlo_optimization(mean_returns.values, cov_matrix.values, 
                                          num_assets, NUM_PORTFOLIOS)
    scipy_result = scipy_optimization(mean_returns.values, cov_matrix.values, 
                                        num_assets)
    
    # ==========================================================================
    # Results Comparison
    # ==========================================================================
    print("=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        "Stock": available_tickers,
        "Monte Carlo": mc_result["weights"],
        "scipy.optimize": scipy_result["weights"],
        "Difference": scipy_result["weights"] - mc_result["weights"],
    })
    comparison_df["Monte Carlo"] = comparison_df["Monte Carlo"].apply(lambda x: f"{x:.1%}")
    comparison_df["scipy.optimize"] = comparison_df["scipy.optimize"].apply(lambda x: f"{x:.1%}")
    comparison_df["Difference"] = comparison_df["Difference"].apply(lambda x: f"{x:+.1%}")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'Monte Carlo':>15} {'scipy.optimize':>15}")
    print("-" * 60)
    print(f"{'Expected Return':<25} {mc_result['return']:>14.1%} {scipy_result['return']:>14.1%}")
    print(f"{'Volatility':<25} {mc_result['volatility']:>14.1%} {scipy_result['volatility']:>14.1%}")
    print(f"{'Sharpe Ratio':<25} {mc_result['sharpe']:>15.4f} {scipy_result['sharpe']:>15.4f}")
    print("-" * 60)
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    plt.figure(figsize=(12, 7))
    
    # Plot all Monte Carlo portfolios
    scatter = plt.scatter(
        mc_result["all_volatility"], 
        mc_result["all_returns"], 
        c=mc_result["all_sharpe"], 
        cmap="viridis", 
        marker="o",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter, label="Sharpe Ratio")
    
    # Mark Monte Carlo optimal
    plt.scatter(
        mc_result["volatility"],
        mc_result["return"],
        c="red",
        marker="*",
        s=300,
        edgecolors="white",
        linewidths=1.5,
        label=f"Monte Carlo (Sharpe: {mc_result['sharpe']:.3f})",
        zorder=5,
    )
    
    # Mark scipy optimal
    plt.scatter(
        scipy_result["volatility"],
        scipy_result["return"],
        c="lime",
        marker="D",
        s=150,
        edgecolors="black",
        linewidths=1.5,
        label=f"scipy.optimize (Sharpe: {scipy_result['sharpe']:.3f})",
        zorder=5,
    )
    
    plt.xlabel("Volatility (Annual Risk)", fontsize=12)
    plt.ylabel("Expected Return (Annual)", fontsize=12)
    plt.title("Efficient Frontier: Monte Carlo vs Mathematical Optimization", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("efficient_frontier_comparison.png", dpi=150)
    plt.show()
    
    print("\n✅ Chart saved as 'efficient_frontier_comparison.png'")
