"""
Application configuration settings.

Centralizes all configuration parameters for the portfolio tracker.
Supports environment-based configuration and sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""
    path: Path = field(default_factory=lambda: Path("db/portfolio.db"))
    
    @property
    def url(self) -> str:
        """SQLAlchemy connection URL."""
        return f"sqlite:///{self.path}"


@dataclass(frozen=True)
class DataFetcherConfig:
    """Data fetcher configuration."""
    # Yahoo Finance settings
    yfinance_timeout: int = 30
    yfinance_max_retries: int = 3
    
    # Default lookback period for initial fetch (days)
    default_lookback_days: int = 365 * 2  # 2 years
    
    # Rate limiting (seconds between requests)
    request_delay: float = 0.5


@dataclass(frozen=True)
class RiskConfig:
    """Risk analytics configuration."""
    # Annualization factor (trading days per year)
    trading_days_per_year: int = 252
    
    # Minimum data points for reliable metrics
    min_price_history_days: int = 60


@dataclass(frozen=True)
class DecisionConfig:
    """Decision engine thresholds."""
    # Allocation thresholds
    concentration_warning_pct: float = 0.30  # 30%
    concentration_danger_pct: float = 0.40   # 40%
    concentration_extreme_pct: float = 0.60  # 60%
    
    # Volatility thresholds (annualized)
    high_volatility_threshold: float = 0.50  # 50%
    
    # Drawdown thresholds
    severe_drawdown_threshold: float = -0.50  # -50%
    moderate_drawdown_threshold: float = -0.30  # -30%
    
    # Valuation bands
    pe_cheap_threshold: float = 15.0
    pe_fair_threshold: float = 25.0
    pe_expensive_threshold: float = 35.0
    
    peg_cheap_threshold: float = 1.0
    peg_fair_threshold: float = 1.5
    peg_expensive_threshold: float = 2.0
    
    ev_ebitda_cheap_threshold: float = 10.0
    ev_ebitda_fair_threshold: float = 15.0


@dataclass(frozen=True)
class UIConfig:
    """Dashboard UI configuration."""
    page_title: str = "Portfolio Review"
    layout: str = "wide"
    
    # Number formatting
    decimal_places: int = 2
    percentage_decimal_places: int = 1
    
    # Admin features (trading and cash management UI)
    enable_admin_ui: bool = True


@dataclass
class Config:
    """
    Main configuration container.
    
    Usage:
        from config import config
        db_path = config.database.path
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_fetcher: DataFetcherConfig = field(default_factory=DataFetcherConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Base paths
    project_root: ClassVar[Path] = Path(__file__).parent
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Create config from environment variables.
        
        Supports overrides via:
        - PORTFOLIO_DB_PATH: Custom database path
        """
        db_path_env = os.getenv("PORTFOLIO_DB_PATH")
        db_config = DatabaseConfig(
            path=Path(db_path_env) if db_path_env else DatabaseConfig().path
        )
        
        return cls(database=db_config)


# Global config instance
config = Config.from_env()
