"""
Technical Indicators Module.

Provides functions for calculating common technical analysis indicators:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

All functions use pandas for efficient rolling window calculations.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple


@st.cache_data(ttl=3600)
def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Series of price data (typically close prices).
        period: Number of periods for the moving average (default: 20).
        
    Returns:
        Series containing SMA values.
    """
    return prices.rolling(window=period, min_periods=1).mean()


@st.cache_data(ttl=3600)
def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Uses the standard EMA formula with span parameter.
    
    Args:
        prices: Series of price data (typically close prices).
        period: Number of periods for the EMA (default: 20).
        
    Returns:
        Series containing EMA values.
    """
    return prices.ewm(span=period, adjust=False).mean()


@st.cache_data(ttl=3600)
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    
    Args:
        prices: Series of price data (typically close prices).
        period: Number of periods for RSI calculation (default: 14).
        
    Returns:
        Series containing RSI values (0-100 range).
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gains and losses using EMA (Wilder's smoothing)
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (when avg_losses is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)
    rsi = rsi.fillna(50)  # Neutral RSI when insufficient data
    
    return rsi


@st.cache_data(ttl=3600)
def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line
    
    Args:
        prices: Series of price data (typically close prices).
        fast_period: Period for fast EMA (default: 12).
        slow_period: Period for slow EMA (default: 26).
        signal_period: Period for signal line EMA (default: 9).
        
    Returns:
        Tuple of (macd_line, signal_line, histogram) Series.
    """
    # Calculate EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd_line = fast_ema - slow_ema
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@st.cache_data(ttl=3600)
def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Middle Band = SMA
    Upper Band = SMA + (std_dev * num_std)
    Lower Band = SMA - (std_dev * num_std)
    
    Args:
        prices: Series of price data (typically close prices).
        period: Number of periods for SMA and std dev (default: 20).
        num_std: Number of standard deviations for bands (default: 2.0).
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band) Series.
    """
    # Middle band (SMA)
    middle_band = prices.rolling(window=period, min_periods=1).mean()
    
    # Standard deviation
    rolling_std = prices.rolling(window=period, min_periods=1).std()
    
    # Upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame with OHLCV data.
    
    Adds the following columns:
    - sma_20, sma_50, sma_200: Simple Moving Averages
    - ema_12, ema_26: Exponential Moving Averages
    - rsi: Relative Strength Index (14-period)
    - macd, macd_signal, macd_histogram: MACD components
    - bb_upper, bb_middle, bb_lower: Bollinger Bands
    
    Args:
        df: DataFrame with 'close' column (and optionally 'high', 'low', 'open', 'volume').
        
    Returns:
        DataFrame with indicator columns added.
    """
    result = df.copy()
    close = result["close"]
    
    # SMAs
    result["sma_20"] = calculate_sma(close, 20)
    result["sma_50"] = calculate_sma(close, 50)
    result["sma_200"] = calculate_sma(close, 200)
    
    # EMAs
    result["ema_12"] = calculate_ema(close, 12)
    result["ema_26"] = calculate_ema(close, 26)
    
    # RSI
    result["rsi"] = calculate_rsi(close, 14)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(close)
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = histogram
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(close)
    result["bb_upper"] = upper
    result["bb_middle"] = middle
    result["bb_lower"] = lower
    
    return result
