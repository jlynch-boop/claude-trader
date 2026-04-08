"""
indicators.py — Technical indicator calculations.

All functions accept pandas Series or DataFrames and return the same.
Calculations are done from scratch so you can understand the math,
with comments explaining what each indicator measures.

Indicators provided:
  - sma(series, period)          — Simple Moving Average
  - ema(series, period)          — Exponential Moving Average
  - rsi(series, period)          — Relative Strength Index
  - bollinger_bands(series, ...) — Bollinger Bands (upper, middle, lower)
  - atr(high, low, close, ...)   — Average True Range
  - macd(series, ...)            — MACD line, signal line, histogram

Usage:
    import pandas as pd
    from src.indicators import sma, rsi, bollinger_bands

    df = pd.DataFrame(...)  # OHLCV DataFrame
    df["sma_20"] = sma(df["close"], 20)
    df["rsi_14"] = rsi(df["close"], 14)
    upper, middle, lower = bollinger_bands(df["close"], 20, 2.0)
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average — the average closing price over the last N candles.

    What it tells you: The average "fair value" over the period.
    Values above SMA = price is above average (trending up).
    Values below SMA = price is below average (trending down).

    Args:
        series: Price series (typically close prices).
        period: Number of candles to average.

    Returns:
        Series of SMA values. First (period-1) values will be NaN.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average — like SMA but gives more weight to recent prices.

    What it tells you: Same as SMA but reacts faster to recent price changes.
    Commonly used for faster trend detection.

    Math: EMA = price * multiplier + prev_EMA * (1 - multiplier)
          where multiplier = 2 / (period + 1)

    Args:
        series: Price series (typically close prices).
        period: Lookback period.

    Returns:
        Series of EMA values.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index — measures momentum on a 0-100 scale.

    What it tells you:
      - RSI > 70: Overbought (price may be due for a pullback)
      - RSI < 30: Oversold (price may be due for a bounce)
      - RSI ~50: Neutral

    Math: RSI = 100 - (100 / (1 + RS))
          where RS = average gain / average loss over the period

    Args:
        series: Price series (typically close prices).
        period: Lookback period (default 14, the standard).

    Returns:
        Series of RSI values (0-100). First `period` values will be NaN.
    """
    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Calculate average gains and losses using Wilder's smoothing method
    # (This is what Welles Wilder, the creator, specified)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Calculate RSI; when avg_loss = 0, price only rises → RSI = 100 by convention
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)

    return rsi_values


def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands — a volatility channel around a moving average.

    What it tells you:
      - Upper band: Price at unusually high level (2 std deviations above average)
      - Lower band: Price at unusually low level (2 std deviations below average)
      - Touch of lower band = potential buy signal (price unusually cheap)
      - Touch of upper band = potential sell signal (price unusually expensive)
      - Band width expands in volatile markets, contracts in calm markets

    Math:
      Middle = SMA(period)
      Upper  = Middle + (std_dev * rolling_std)
      Lower  = Middle - (std_dev * rolling_std)

    Args:
        series:  Price series (typically close prices).
        period:  SMA lookback period (default 20).
        std_dev: Number of standard deviations (default 2.0).

    Returns:
        Tuple of (upper, middle, lower) Series.
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()

    upper = middle + (std_dev * rolling_std)
    lower = middle - (std_dev * rolling_std)

    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Average True Range — measures market volatility.

    What it tells you: How much the price typically moves per candle.
    Used for placing stop losses that adapt to market conditions.
    High ATR = volatile market. Low ATR = calm market.

    Math: True Range = max of:
      - Current high - current low
      - |Current high - previous close|
      - |Current low - previous close|
    ATR = Average True Range over `period` candles

    Args:
        high:   Series of high prices.
        low:    Series of low prices.
        close:  Series of close prices.
        period: Lookback period (default 14).

    Returns:
        Series of ATR values.
    """
    prev_close = close.shift(1)

    # True range is the largest of these three values
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's ATR smoothing (same as RSI's avg gain/loss smoothing)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD — Moving Average Convergence/Divergence.

    What it tells you:
      - MACD line crossing above signal line = bullish momentum (buy signal)
      - MACD line crossing below signal line = bearish momentum (sell signal)
      - Histogram shows the gap between MACD and signal (positive = bullish)

    Math:
      MACD line   = EMA(fast) - EMA(slow)
      Signal line = EMA(MACD line, signal_period)
      Histogram   = MACD line - Signal line

    Args:
        series: Price series (typically close prices).
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) Series.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def percent_b(series: pd.Series, upper: pd.Series,
              lower: pd.Series) -> pd.Series:
    """
    %B — Shows where price is within the Bollinger Bands.

    What it tells you:
      - %B = 1.0: Price is at the upper band
      - %B = 0.5: Price is at the middle band
      - %B = 0.0: Price is at the lower band
      - %B > 1.0: Price is above the upper band (very overbought)
      - %B < 0.0: Price is below the lower band (very oversold)

    Args:
        series: Price series.
        upper:  Upper Bollinger Band series.
        lower:  Lower Bollinger Band series.

    Returns:
        Series of %B values.
    """
    band_width = upper - lower
    return (series - lower) / band_width.replace(0, np.nan)
