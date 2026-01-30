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
        # Prefer direct URL if provided in environment
        env_url = os.environ.get("PORTFOLIO_DB_URL") or os.environ.get("DATABASE_URL")
        if env_url:
            return env_url
            
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
class IndexConfig:
    """
    Configuration for a single market index with multi-source symbol mapping.
    
    Supports different symbols across data providers:
    - yahoo: Yahoo Finance symbol (e.g., ^GSPC, ^RUT)
    - alpha_vantage: Alpha Vantage symbol
    - finviz: Finviz symbol
    - other providers can be added as needed
    """
    symbol: str  # Internal canonical symbol (e.g., SPX, RUT)
    name: str  # Human-readable name
    description: str = ""
    category: str = "EQUITY"  # EQUITY, VOLATILITY, COMMODITY, BOND, CURRENCY
    sources: dict = field(default_factory=dict)  # Provider -> symbol mapping
    
    def get_symbol(self, provider: str) -> str:
        """Get the symbol for a specific data provider, falling back to canonical symbol."""
        return self.sources.get(provider, self.symbol)


@dataclass(frozen=True)
class MarketIndicesConfig:
    """
    Configuration for market benchmark indices.
    
    Centralizes index tracking with multi-source symbol mapping.
    Each index can have different symbols across data providers.
    
    Example usage:
        config.market_indices.get_index("SPX")
        config.market_indices.get_yahoo_symbol("SPX")  # Returns "^GSPC"
    """
    tracked_indices: tuple[IndexConfig, ...] = field(default_factory=lambda: (
        IndexConfig(
            symbol="SPX",
            name="S&P 500",
            description="Large-cap US equity market benchmark tracking 500 leading companies",
            category="EQUITY",
            sources={
                "yahoo": "^GSPC",
                "alpha_vantage": "SPX",
                "finviz": "SPX",
            },
        ),
        IndexConfig(
            symbol="RUT",
            name="Russell 2000",
            description="Small-cap US equity market benchmark tracking 2000 small companies",
            category="EQUITY",
            sources={
                "yahoo": "^RUT",
                "alpha_vantage": "RUT",
                "finviz": "RUT",
            },
        ),
        IndexConfig(
            symbol="VIX",
            name="CBOE Volatility Index",
            description="Market volatility expectation derived from S&P 500 options",
            category="VOLATILITY",
            sources={
                "yahoo": "^VIX",
                "alpha_vantage": "VIX",
                "finviz": "VIX",
            },
        ),
        IndexConfig(
            symbol="DJI",
            name="Dow Jones Industrial Average",
            description="Price-weighted index of 30 large-cap US blue-chip companies",
            category="EQUITY",
            sources={
                "yahoo": "^DJI",
                "alpha_vantage": "DJI",
                "finviz": "DJI",
            },
        ),
        IndexConfig(
            symbol="IXIC",
            name="NASDAQ Composite",
            description="Tech-heavy index tracking all NASDAQ-listed stocks",
            category="EQUITY",
            sources={
                "yahoo": "^IXIC",
                "alpha_vantage": "IXIC",
                "finviz": "IXIC",
            },
        ),
        IndexConfig(
            symbol="GOLD",
            name="Gold Futures",
            description="Gold commodity price benchmark",
            category="COMMODITY",
            sources={
                "yahoo": "GC=F",
                "alpha_vantage": "GOLD",
                "finviz": "GC",
            },
        ),
    ))
    
    def get_index(self, symbol: str) -> IndexConfig | None:
        """Get index configuration by canonical symbol."""
        for idx in self.tracked_indices:
            if idx.symbol == symbol:
                return idx
        return None
    
    def get_yahoo_symbol(self, symbol: str) -> str:
        """Get Yahoo Finance symbol for an index."""
        idx = self.get_index(symbol)
        return idx.get_symbol("yahoo") if idx else symbol
    
    def get_alpha_vantage_symbol(self, symbol: str) -> str:
        """Get Alpha Vantage symbol for an index."""
        idx = self.get_index(symbol)
        return idx.get_symbol("alpha_vantage") if idx else symbol
    
    def get_finviz_symbol(self, symbol: str) -> str:
        """Get Finviz symbol for an index."""
        idx = self.get_index(symbol)
        return idx.get_symbol("finviz") if idx else symbol
    
    @property
    def symbols(self) -> list[str]:
        """Get list of all tracked index symbols."""
        return [idx.symbol for idx in self.tracked_indices]
    
    @property
    def equity_indices(self) -> list[IndexConfig]:
        """Get all equity indices."""
        return [idx for idx in self.tracked_indices if idx.category == "EQUITY"]


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
    market_indices: MarketIndicesConfig = field(default_factory=MarketIndicesConfig)
    
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
