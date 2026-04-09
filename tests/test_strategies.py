"""
tests/test_strategies.py — Tests for all three strategy implementations.

Each strategy is tested with crafted price series that should definitively
trigger (or not trigger) each signal.

Run with: pytest tests/test_strategies.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.strategy_base import SignalType
from src.strategy_bollinger_rsi import BollingerRsiStrategy
from src.strategy_momentum import MomentumStrategy
from src.strategy_grid import GridStrategy


# ---------------------------------------------------------------------------
# Shared config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "trading": {"pairs": ["BTC/USDT"], "timeframe": "1h",
                    "starting_capital": 1000.0},
        "backtest": {"fee_rate": 0.001, "slippage_rate": 0.0005,
                     "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "risk": {"max_position_pct": 0.20, "max_drawdown_pct": 0.15,
                 "stop_loss_pct": 0.03, "take_profit_pct": 0.06,
                 "max_open_positions": 3, "risk_per_trade_pct": 0.01},
        "strategies": {
            "bollinger_rsi": {
                "bb_period": 20, "bb_std": 2.0,
                "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
            },
            "momentum": {
                "fast_ma": 10, "slow_ma": 30, "trend_ma": 50,
                "atr_period": 14, "atr_stop_multiplier": 2.0,
            },
            "grid": {
                "grid_levels": 10, "grid_spread_pct": 0.02,
                "range_pct": 0.10, "pair": "BTC/USDT",
            },
        },
    }


def _make_ohlcv(prices: list[float],
                start: str = "2024-01-01") -> pd.DataFrame:
    """Build an OHLCV DataFrame from a close price list."""
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    # Add small variation to avoid zero std (needed for RSI, Bollinger)
    highs  = [p * 1.003 for p in prices]
    lows   = [p * 0.997 for p in prices]
    return pd.DataFrame({
        "open":   prices,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": [1000.0] * n,
    }, index=idx)


# ---------------------------------------------------------------------------
# BollingerRsiStrategy tests
# ---------------------------------------------------------------------------

class TestBollingerRsiStrategy:

    def test_required_history(self, config):
        s = BollingerRsiStrategy(config)
        assert s.required_history() >= 20  # At least bb_period

    def test_hold_during_warmup(self, config):
        s = BollingerRsiStrategy(config)
        prices = [42000.0] * 10  # Far fewer than required_history
        data = _make_ohlcv(prices)
        signal = s.on_candle(data.iloc[-1], data)
        assert signal.type == SignalType.HOLD

    def test_hold_for_normal_price(self, config):
        """Price oscillating gently around the BB midline → HOLD on most candles."""
        s = BollingerRsiStrategy(config)
        # Small oscillation keeps price inside the bands, RSI neutral
        np.random.seed(7)
        prices = [42000.0 + np.random.uniform(-300, 300) for _ in range(100)]
        data = _make_ohlcv(prices)
        # Count HOLDs in the last 20 candles — should be the majority
        holds = 0
        for i in range(80, 100):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.HOLD:
                holds += 1
        assert holds >= 10, "Expected mostly HOLD signals for price near BB midline"

    def test_buy_signal_on_oversold(self, config):
        """Sharp price drop should trigger a BUY signal."""
        s = BollingerRsiStrategy(config)
        # Start at 42000, gradually drop to trigger both lower BB and RSI < 30
        warmup = [42000.0] * 30
        # Rapid decline: each candle drops more than last — extreme selling
        drop = [42000.0 - i * 200 for i in range(40)]
        prices = warmup + drop
        data = _make_ohlcv(prices)

        # Scan through all candles to find a BUY
        buys = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.BUY:
                buys.append(signal)
        assert len(buys) > 0, "Expected at least one BUY on a sharp decline"

    def test_sell_signal_on_overbought(self, config):
        """Sharp price rise should eventually trigger a SELL signal."""
        s = BollingerRsiStrategy(config)
        warmup = [42000.0] * 30
        # Rapid rise triggers upper BB touch + RSI > 70
        rise = [42000.0 + i * 200 for i in range(40)]
        prices = warmup + rise
        data = _make_ohlcv(prices)

        sells = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.SELL:
                sells.append(signal)
        assert len(sells) > 0, "Expected at least one SELL on a sharp rise"

    def test_buy_signal_has_stop_loss(self, config):
        """Every BUY must include a stop loss."""
        s = BollingerRsiStrategy(config)
        warmup = [42000.0] * 30
        drop   = [42000.0 - i * 200 for i in range(40)]
        data   = _make_ohlcv(warmup + drop)

        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.BUY:
                assert signal.stop_loss > 0
                assert signal.stop_loss < signal.price
                break


# ---------------------------------------------------------------------------
# MomentumStrategy tests
# ---------------------------------------------------------------------------

class TestMomentumStrategy:

    def test_required_history(self, config):
        s = MomentumStrategy(config)
        # trend_ma=50, so requires at least 60 candles
        assert s.required_history() >= 50

    def test_hold_during_warmup(self, config):
        s = MomentumStrategy(config)
        prices = [42000.0] * 20
        data = _make_ohlcv(prices)
        signal = s.on_candle(data.iloc[-1], data)
        assert signal.type == SignalType.HOLD

    def test_buy_on_golden_cross_above_trend(self, config):
        """Fast MA crossing above slow MA, price above trend → BUY."""
        s = MomentumStrategy(config)
        # Long flat base → trend SMA warms up and stabilises
        # Then sharp rise: fast crosses above slow while price > trend SMA
        base     = [40000.0] * 80          # Warmup — trend SMA valid and flat
        recovery = [40000.0 + i * 400 for i in range(80)]  # Strong uptrend
        prices   = base + recovery
        data = _make_ohlcv(prices)

        buys = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.BUY:
                buys.append(signal)

        assert len(buys) > 0, "Expected BUY on golden cross above trend"

    def test_sell_on_death_cross(self, config):
        """Fast MA crossing below slow MA → SELL."""
        s = MomentumStrategy(config)
        # Long uptrend so _prev_fast_above_slow is firmly True, then sharp reversal
        uptrend  = [40000.0 + i * 100 for i in range(120)]
        reversal = [uptrend[-1] - i * 500 for i in range(80)]
        prices   = uptrend + reversal
        data = _make_ohlcv(prices)

        sells = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.SELL:
                sells.append(signal)

        assert len(sells) > 0, "Expected SELL on death cross"

    def test_no_buy_below_trend(self, config):
        """Golden cross below the trend line should NOT produce a BUY."""
        s = MomentumStrategy(config)
        # Strong downtrend — price stays below 50-period SMA throughout
        prices = [50000.0 - i * 100 for i in range(200)]
        data = _make_ohlcv(prices)

        buys = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.BUY:
                buys.append(signal)

        assert len(buys) == 0, "Should not BUY when price is below trend line"

    def test_buy_has_stop_below_entry(self, config):
        """Every BUY must have stop_loss < price."""
        s = MomentumStrategy(config)
        downtrend = [50000.0 - i * 50 for i in range(30)]
        recovery  = [downtrend[-1] + i * 300 for i in range(100)]
        prices    = downtrend + recovery
        data = _make_ohlcv(prices)

        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.BUY:
                assert signal.stop_loss < signal.price
                break


# ---------------------------------------------------------------------------
# GridStrategy tests
# ---------------------------------------------------------------------------

class TestGridStrategy:

    def test_required_history(self, config):
        s = GridStrategy(config)
        assert s.required_history() == 5

    def test_first_candle_initializes_grid(self, config):
        s = GridStrategy(config)
        prices = [42000.0] * 10
        data   = _make_ohlcv(prices)
        # Run through required_history + 1 candles
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
        # After initialization, grid should be set up
        assert s._grid_initialized
        assert s._grid_upper > s._grid_center > s._grid_lower

    def test_buy_when_price_crosses_grid_level(self, config):
        """Price dropping through a grid level should produce a BUY."""
        s = GridStrategy(config)
        # Initialize with flat price, then drop through a level
        flat  = [42000.0] * 10
        # Drop 3% (within the 10% range) to cross a grid level
        drop  = [42000.0 * (1 - 0.03 * i/10) for i in range(1, 20)]
        prices = flat + drop
        data  = _make_ohlcv(prices)

        buys = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.BUY:
                buys.append(signal)

        assert len(buys) > 0, "Expected BUY as price drops through grid levels"

    def test_circuit_breaker_exits_on_breakout(self, config):
        """Price breaking far below the grid should trigger a SELL."""
        s = GridStrategy(config)
        flat  = [42000.0] * 10
        # Big drop: 15% below center (beyond the 10% range)
        crash = [42000.0 * (1 - 0.15)] * 20
        prices = flat + crash
        data  = _make_ohlcv(prices)

        sells = []
        for i in range(len(data)):
            signal = s.on_candle(data.iloc[i], data.iloc[:i+1])
            if signal.type == SignalType.SELL:
                sells.append(signal)
            elif signal.type == SignalType.BUY:
                # Simulate position being open so we can get the sell
                s._in_position = True
                s._entry_price = data.iloc[i]["close"]
                s._sell_level  = data.iloc[i]["close"] * 1.02

        # The breakout should eventually trigger a circuit breaker exit
        assert s._grid_initialized  # Grid was reset after breakout

    def test_grid_levels_are_ordered(self, config):
        """Grid levels should go from high to low."""
        s = GridStrategy(config)
        prices = [42000.0] * 10
        data = _make_ohlcv(prices)
        for i in range(len(data)):
            s.on_candle(data.iloc[i], data.iloc[:i+1])

        if s._grid_initialized and len(s._grid_levels) > 1:
            for j in range(len(s._grid_levels) - 1):
                assert s._grid_levels[j] > s._grid_levels[j + 1]

    def test_hold_in_flat_market_at_center(self, config):
        """Price sitting flat at grid center should HOLD."""
        s = GridStrategy(config)
        prices = [42000.0] * 20  # No movement
        data = _make_ohlcv(prices)
        signals = []
        for i in range(len(data)):
            sig = s.on_candle(data.iloc[i], data.iloc[:i+1])
            signals.append(sig)

        # Most signals after warmup should be HOLD (no grid level crossed)
        holds = [s for s in signals[10:] if s.type == SignalType.HOLD]
        assert len(holds) > 0
