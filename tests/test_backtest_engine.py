"""
tests/test_backtest_engine.py — Tests for the backtest engine and metrics.

We use simple, known price series so we can calculate expected outcomes
by hand and verify the engine matches.

Run with: pytest tests/test_backtest_engine.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.backtest_engine import BacktestEngine, BacktestResult, Trade
from src.backtest_metrics import BacktestMetrics
from src.strategy_base import Strategy, Signal, SignalType, Portfolio
from src.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Minimal test config
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "trading": {
        "pairs": ["BTC/USDT"],
        "starting_capital": 1000.0,
        "timeframe": "1h",
    },
    "backtest": {
        "fee_rate": 0.001,       # 0.1%
        "slippage_rate": 0.0005, # 0.05%
        "start_date": "2024-01-01",
        "end_date":   "2024-12-31",
    },
    "risk": {
        "max_position_pct": 0.20,
        "max_drawdown_pct": 0.15,
        "stop_loss_pct":    0.03,
        "take_profit_pct":  0.06,
        "max_open_positions": 3,
        "risk_per_trade_pct": 0.01,
    },
}


# ---------------------------------------------------------------------------
# Helper strategies for testing
# ---------------------------------------------------------------------------

def _make_ohlcv(prices: list[float],
                start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close price list."""
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":   prices,
        "high":   [p * 1.005 for p in prices],
        "low":    [p * 0.995 for p in prices],
        "close":  prices,
        "volume": [100.0] * n,
    }, index=idx)


class BuyOnFirstSignal(Strategy):
    """Buys on candle 5, sells on candle 10. Simple deterministic strategy."""
    def required_history(self) -> int:
        return 4

    def on_candle(self, candle, history) -> Signal:
        i = len(history)
        if i == 5:
            return Signal.buy(
                pair="BTC/USDT",
                price=candle["close"],
                stop_loss=candle["close"] * 0.90,   # 10% stop
                take_profit=candle["close"] * 1.20,
                reason="test buy at candle 5",
            )
        if i == 10:
            return Signal.sell(
                pair="BTC/USDT",
                price=candle["close"],
                reason="test sell at candle 10",
            )
        return Signal.hold()


class NeverTrades(Strategy):
    """Always returns HOLD — useful for testing baseline."""
    def required_history(self) -> int:
        return 1

    def on_candle(self, candle, history) -> Signal:
        return Signal.hold()


class AlwaysBuys(Strategy):
    """Tries to buy every candle — tests max position enforcement."""
    def required_history(self) -> int:
        return 1

    def on_candle(self, candle, history) -> Signal:
        return Signal.buy(
            pair="BTC/USDT",
            price=candle["close"],
            stop_loss=candle["close"] * 0.90,
            reason="always buy",
        )


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

def test_no_trades_with_hold_strategy():
    prices = [100.0] * 50
    data = _make_ohlcv(prices)
    engine = BacktestEngine(NeverTrades(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    # End-of-data close-out produces 0 trades (no open positions)
    assert len(result.trades) == 0


def test_equity_curve_length_matches_data():
    prices = [100.0] * 50
    data = _make_ohlcv(prices)
    engine = BacktestEngine(NeverTrades(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    assert len(result.equity_curve) == len(data)


def test_equity_flat_with_no_trades():
    prices = [100.0] * 50
    data = _make_ohlcv(prices)
    engine = BacktestEngine(NeverTrades(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    # Equity should stay at starting capital throughout
    assert result.equity_curve.iloc[-1] == pytest.approx(1000.0)


def test_buy_and_sell_produces_trade():
    prices = [100.0] * 20
    data = _make_ohlcv(prices)
    engine = BacktestEngine(BuyOnFirstSignal(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    # Should have exactly 1 completed trade (buy at 5, sell at 10)
    assert len(result.trades) == 1


def test_profitable_trade_increases_equity():
    # Rising prices: buy at 100, sell at 110 (+10%)
    prices = [100.0] * 5 + [100.0] + [101.0, 102.0, 103.0, 104.0, 110.0] + [110.0] * 9
    data = _make_ohlcv(prices)
    engine = BacktestEngine(BuyOnFirstSignal(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    assert result.equity_curve.iloc[-1] > 1000.0


def test_losing_trade_decreases_equity():
    # Falling prices: buy at 100, sell at 90 (-10%)
    prices = [100.0] * 5 + [100.0] + [99.0, 98.0, 97.0, 96.0, 90.0] + [90.0] * 9
    data = _make_ohlcv(prices)
    engine = BacktestEngine(BuyOnFirstSignal(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    assert result.equity_curve.iloc[-1] < 1000.0


def test_fees_are_charged():
    prices = [100.0] * 20
    data = _make_ohlcv(prices)
    engine = BacktestEngine(BuyOnFirstSignal(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    if result.trades:
        trade = result.trades[0]
        # Fees should be positive
        assert trade.entry_fee > 0 or trade.exit_fee > 0


def test_stop_loss_triggers():
    """Price drops below stop — position should be closed automatically."""
    # Buy at candle 5 (price=100, stop=90), then price drops to 85
    prices = [100.0] * 5 + [100.0, 100.0, 100.0, 85.0] + [85.0] * 11
    data = _make_ohlcv(prices)

    class BuyAndHold(Strategy):
        def required_history(self): return 4
        def on_candle(self, candle, history):
            if len(history) == 5:
                return Signal.buy("BTC/USDT", candle["close"],
                                  stop_loss=90.0, reason="buy")
            return Signal.hold()

    engine = BacktestEngine(BuyAndHold(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    # Should have a trade that was stopped out
    assert len(result.trades) >= 1
    stop_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
    assert len(stop_trades) == 1


def test_take_profit_triggers():
    """Price rises above take profit — position should close."""
    prices = [100.0] * 5 + [100.0, 100.0, 100.0, 125.0] + [125.0] * 11

    class BuyWithTP(Strategy):
        def required_history(self): return 4
        def on_candle(self, candle, history):
            if len(history) == 5:
                return Signal.buy("BTC/USDT", candle["close"],
                                  stop_loss=90.0, take_profit=120.0,
                                  reason="buy")
            return Signal.hold()

    data = _make_ohlcv(prices)
    engine = BacktestEngine(BuyWithTP(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    tp_trades = [t for t in result.trades if t.exit_reason == "take_profit"]
    assert len(tp_trades) == 1


def test_max_one_position_per_pair():
    """AlwaysBuys tries every candle but should only ever have 1 BTC/USDT position."""
    prices = [100.0] * 30
    data = _make_ohlcv(prices)
    engine = BacktestEngine(AlwaysBuys(TEST_CONFIG), data, TEST_CONFIG)
    result = engine.run()
    # Should have at most 1 position opened (subsequent buys rejected)
    buys = [t for t in result.trades if t.entry_price > 0]
    assert len(buys) <= 1


def test_insufficient_data_raises():
    prices = [100.0] * 3  # Too few candles for BuyOnFirstSignal (needs 5)
    data = _make_ohlcv(prices)
    with pytest.raises(ValueError):
        engine = BacktestEngine(BuyOnFirstSignal(TEST_CONFIG), data, TEST_CONFIG)
        engine.run()


# ---------------------------------------------------------------------------
# BacktestMetrics tests
# ---------------------------------------------------------------------------

def _make_result_with_equity(equity_values: list[float],
                              trades: list = None) -> BacktestResult:
    """Helper to build a BacktestResult with a given equity curve."""
    idx = pd.date_range("2024-01-01", periods=len(equity_values),
                        freq="1h", tz="UTC")
    return BacktestResult(
        strategy_name="TestStrategy",
        pair="BTC/USDT",
        start_date=idx[0],
        end_date=idx[-1],
        starting_capital=1000.0,
        trades=trades or [],
        equity_curve=pd.Series(equity_values, index=idx),
    )


def test_total_return_zero_for_flat_equity():
    result = _make_result_with_equity([1000.0] * 100)
    m = BacktestMetrics(result)
    assert m.total_return() == pytest.approx(0.0)


def test_total_return_positive():
    result = _make_result_with_equity([1000.0, 1100.0])
    m = BacktestMetrics(result)
    assert m.total_return() == pytest.approx(0.10)


def test_total_return_negative():
    result = _make_result_with_equity([1000.0, 900.0])
    m = BacktestMetrics(result)
    assert m.total_return() == pytest.approx(-0.10)


def test_max_drawdown_flat_is_zero():
    result = _make_result_with_equity([1000.0] * 100)
    m = BacktestMetrics(result)
    assert m.max_drawdown() == pytest.approx(0.0)


def test_max_drawdown_calculation():
    # Peak=1000, trough=800 → 20% drawdown
    equity = [1000.0, 1000.0, 800.0, 900.0, 950.0]
    result = _make_result_with_equity(equity)
    m = BacktestMetrics(result)
    assert m.max_drawdown() == pytest.approx(0.20)


def test_sharpe_zero_for_flat_equity():
    result = _make_result_with_equity([1000.0] * 100)
    m = BacktestMetrics(result)
    assert m.sharpe_ratio() == pytest.approx(0.0, abs=0.01)


def test_win_rate_all_winners():
    trades = [
        Trade("s", "BTC/USDT", pd.Timestamp("2024-01-01"),
              pd.Timestamp("2024-01-02"), "long",
              100.0, 110.0, 1.0, 0.1, 0.11, 9.79, 0.0979, "signal")
        for _ in range(10)
    ]
    result = _make_result_with_equity([1000.0] * 100, trades)
    m = BacktestMetrics(result)
    assert m.win_rate() == pytest.approx(1.0)


def test_win_rate_all_losers():
    trades = [
        Trade("s", "BTC/USDT", pd.Timestamp("2024-01-01"),
              pd.Timestamp("2024-01-02"), "long",
              100.0, 90.0, 1.0, 0.1, 0.09, -10.19, -0.1019, "stop_loss")
        for _ in range(10)
    ]
    result = _make_result_with_equity([1000.0] * 100, trades)
    m = BacktestMetrics(result)
    assert m.win_rate() == pytest.approx(0.0)


def test_profit_factor_greater_than_one_for_winners():
    wins  = [Trade("s","BTC/USDT",pd.Timestamp("2024-01-01"),
                   pd.Timestamp("2024-01-02"),"long",100,110,1,0.1,0.11,9.79,0.1,"signal")
             for _ in range(6)]
    loses = [Trade("s","BTC/USDT",pd.Timestamp("2024-01-01"),
                   pd.Timestamp("2024-01-02"),"long",100,95,1,0.1,0.095,-5.195,-.05,"stop_loss")
             for _ in range(4)]
    result = _make_result_with_equity([1000.0]*100, wins + loses)
    m = BacktestMetrics(result)
    assert m.profit_factor() > 1.0


def test_passes_gate_false_with_no_trades():
    result = _make_result_with_equity([1000.0] * 8760)
    m = BacktestMetrics(result)
    assert not m.passes_gate()


def test_summary_returns_dict():
    result = _make_result_with_equity([1000.0] * 100)
    m = BacktestMetrics(result)
    s = m.summary()
    assert isinstance(s, dict)
    assert "sharpe_ratio" in s
    assert "max_drawdown_pct" in s
    assert "passes_gate" in s
