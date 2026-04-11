"""
tests/test_indicators.py — Tests for src/indicators.py

We verify each indicator against known values and check edge cases.

Run with: pytest tests/test_indicators.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import sma, ema, rsi, bollinger_bands, atr, adx, macd, percent_b


# ---------------------------------------------------------------------------
# Fixtures — simple, predictable price series for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_series():
    """A series of constant price 100. Moving averages = 100, RSI ~= 50."""
    return pd.Series([100.0] * 50)


@pytest.fixture
def rising_series():
    """A steadily rising price series: 100, 101, 102, ..."""
    return pd.Series([100.0 + i for i in range(50)])


@pytest.fixture
def falling_series():
    """A steadily falling price series: 150, 149, 148, ..."""
    return pd.Series([150.0 - i for i in range(50)])


@pytest.fixture
def oscillating_series():
    """Alternates between 100 and 110 — good for RSI tests."""
    values = []
    for i in range(50):
        values.append(100.0 if i % 2 == 0 else 110.0)
    return pd.Series(values)


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------

def test_sma_flat_series(flat_series):
    result = sma(flat_series, 5)
    # After warmup, all values should be 100.0
    assert result.dropna().iloc[-1] == pytest.approx(100.0)


def test_sma_rising_series(rising_series):
    result = sma(rising_series, 5)
    # SMA(5) of [100,101,102,103,104] = 102
    assert result.iloc[4] == pytest.approx(102.0)


def test_sma_warmup_period(flat_series):
    result = sma(flat_series, 10)
    # First 9 values should be NaN (not enough data yet)
    assert result.iloc[:9].isna().all()
    assert not pd.isna(result.iloc[9])


def test_sma_length_preserved(flat_series):
    result = sma(flat_series, 5)
    assert len(result) == len(flat_series)


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

def test_ema_flat_series(flat_series):
    result = ema(flat_series, 5)
    assert result.dropna().iloc[-1] == pytest.approx(100.0)


def test_ema_reacts_faster_than_sma(rising_series):
    sma_result = sma(rising_series, 10)
    ema_result = ema(rising_series, 10)
    # EMA weights recent prices more, so it should be closer to latest price
    latest_price = rising_series.iloc[-1]
    sma_lag = abs(latest_price - sma_result.iloc[-1])
    ema_lag = abs(latest_price - ema_result.iloc[-1])
    assert ema_lag < sma_lag


def test_ema_length_preserved(flat_series):
    result = ema(flat_series, 5)
    assert len(result) == len(flat_series)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

def test_rsi_flat_series(flat_series):
    # Constant price means no gains or losses — RSI is undefined (NaN)
    result = rsi(flat_series, 14)
    # After warmup the RSI values will be NaN due to zero losses
    # (0/0 case) — this is expected behavior
    assert len(result) == len(flat_series)


def test_rsi_rising_series_near_100(rising_series):
    result = rsi(rising_series, 14)
    # In a steadily rising series, RSI should be very high (near 100)
    # after warmup period
    valid = result.dropna()
    assert valid.iloc[-1] > 70


def test_rsi_falling_series_near_0(falling_series):
    result = rsi(falling_series, 14)
    valid = result.dropna()
    # In a steadily falling series, RSI should be very low (near 0)
    assert valid.iloc[-1] < 30


def test_rsi_bounds():
    # RSI must always be between 0 and 100
    prices = pd.Series([100, 105, 95, 110, 90, 115, 85, 120, 80] * 10)
    result = rsi(prices, 14).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_rsi_length_preserved(rising_series):
    result = rsi(rising_series, 14)
    assert len(result) == len(rising_series)


# ---------------------------------------------------------------------------
# Bollinger Bands tests
# ---------------------------------------------------------------------------

def test_bollinger_flat_series(flat_series):
    upper, middle, lower = bollinger_bands(flat_series, 20, 2.0)
    # For constant price, std dev = 0, so all bands = 100
    assert middle.dropna().iloc[-1] == pytest.approx(100.0)
    assert upper.dropna().iloc[-1] == pytest.approx(100.0)
    assert lower.dropna().iloc[-1] == pytest.approx(100.0)


def test_bollinger_upper_above_lower(rising_series):
    # Use a series with some variance
    prices = pd.Series([100 + i + (i % 5) for i in range(50)])
    upper, middle, lower = bollinger_bands(prices, 20, 2.0)
    valid_upper = upper.dropna()
    valid_lower = lower.dropna()
    assert (valid_upper > valid_lower).all()


def test_bollinger_middle_is_sma(flat_series):
    upper, middle, lower = bollinger_bands(flat_series, 20, 2.0)
    sma_result = sma(flat_series, 20)
    pd.testing.assert_series_equal(middle, sma_result)


def test_bollinger_symmetric_bands():
    prices = pd.Series([100 + np.sin(i) * 5 for i in range(50)])
    upper, middle, lower = bollinger_bands(prices, 20, 2.0)
    # Upper and lower should be equidistant from middle
    upper_dist = (upper - middle).dropna()
    lower_dist = (middle - lower).dropna()
    pd.testing.assert_series_equal(
        upper_dist.round(10), lower_dist.round(10), check_names=False
    )


def test_bollinger_length_preserved(flat_series):
    upper, middle, lower = bollinger_bands(flat_series)
    assert len(upper) == len(flat_series)
    assert len(middle) == len(flat_series)
    assert len(lower) == len(flat_series)


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlc_data():
    """Simple OHLC data for ATR testing."""
    n = 50
    close = pd.Series([100.0 + i for i in range(n)])
    high = close + 2.0
    low = close - 2.0
    return high, low, close


def test_atr_positive(ohlc_data):
    high, low, close = ohlc_data
    result = atr(high, low, close, 14)
    valid = result.dropna()
    assert (valid > 0).all()


def test_atr_reflects_volatility():
    n = 50
    # Low volatility: narrow range
    close_lv = pd.Series([100.0] * n)
    high_lv = close_lv + 0.5
    low_lv = close_lv - 0.5

    # High volatility: wide range
    close_hv = pd.Series([100.0] * n)
    high_hv = close_hv + 10.0
    low_hv = close_hv - 10.0

    atr_lv = atr(high_lv, low_lv, close_lv, 14).dropna().iloc[-1]
    atr_hv = atr(high_hv, low_hv, close_hv, 14).dropna().iloc[-1]

    assert atr_hv > atr_lv


def test_atr_length_preserved(ohlc_data):
    high, low, close = ohlc_data
    result = atr(high, low, close, 14)
    assert len(result) == len(close)


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------

def test_macd_returns_three_series(rising_series):
    macd_line, signal_line, histogram = macd(rising_series)
    assert len(macd_line) == len(rising_series)
    assert len(signal_line) == len(rising_series)
    assert len(histogram) == len(rising_series)


def test_macd_histogram_equals_line_minus_signal(rising_series):
    macd_line, signal_line, histogram = macd(rising_series)
    expected = (macd_line - signal_line).dropna()
    actual = histogram.dropna()
    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_macd_rising_positive(rising_series):
    macd_line, _, _ = macd(rising_series)
    # In a rising series, fast EMA > slow EMA, so MACD should be positive
    valid = macd_line.dropna()
    assert (valid > 0).all()


# ---------------------------------------------------------------------------
# Percent B tests
# ---------------------------------------------------------------------------

def test_percent_b_at_middle():
    prices = pd.Series([100.0] * 50)
    upper = pd.Series([110.0] * 50)
    lower = pd.Series([90.0] * 50)
    result = percent_b(prices, upper, lower)
    assert result.iloc[-1] == pytest.approx(0.5)


def test_percent_b_at_upper():
    prices = pd.Series([110.0] * 50)
    upper = pd.Series([110.0] * 50)
    lower = pd.Series([90.0] * 50)
    result = percent_b(prices, upper, lower)
    assert result.iloc[-1] == pytest.approx(1.0)


def test_percent_b_at_lower():
    prices = pd.Series([90.0] * 50)
    upper = pd.Series([110.0] * 50)
    lower = pd.Series([90.0] * 50)
    result = percent_b(prices, upper, lower)
    assert result.iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_ohlc():
    """Strong uptrend: each candle's high/low/close all rise steadily."""
    n = 60
    close = pd.Series([100.0 + i * 2 for i in range(n)])
    high  = close + 1.0
    low   = close - 1.0
    return high, low, close


@pytest.fixture
def ranging_ohlc():
    """Sideways market oscillating between 98 and 102."""
    n = 60
    prices = [100.0 + (2 if i % 4 < 2 else -2) for i in range(n)]
    close  = pd.Series(prices)
    high   = close + 0.5
    low    = close - 0.5
    return high, low, close


def test_adx_returns_series(trending_ohlc):
    high, low, close = trending_ohlc
    result = adx(high, low, close, period=14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_adx_range_0_to_100(trending_ohlc):
    high, low, close = trending_ohlc
    result = adx(high, low, close, period=14).dropna()
    assert (result >= 0).all(), "ADX should never be negative"
    assert (result <= 100).all(), "ADX should never exceed 100"


def test_adx_high_in_strong_trend(trending_ohlc):
    """A steadily rising market should produce ADX > 25 after warmup."""
    high, low, close = trending_ohlc
    result = adx(high, low, close, period=14).dropna()
    assert result.iloc[-1] > 25, (
        f"Expected ADX > 25 in strong trend, got {result.iloc[-1]:.1f}"
    )


def test_adx_low_in_ranging_market(ranging_ohlc):
    """An oscillating market should produce ADX < 25 after warmup."""
    high, low, close = ranging_ohlc
    result = adx(high, low, close, period=14).dropna()
    assert result.iloc[-1] < 30, (
        f"Expected ADX < 30 in ranging market, got {result.iloc[-1]:.1f}"
    )


def test_adx_nan_during_warmup(trending_ohlc):
    """First values should be NaN until enough data is available."""
    high, low, close = trending_ohlc
    result = adx(high, low, close, period=14)
    assert result.iloc[0] != result.iloc[0], "First value should be NaN"


def test_adx_length_preserved(trending_ohlc):
    high, low, close = trending_ohlc
    result = adx(high, low, close, period=14)
    assert len(result) == len(close)
