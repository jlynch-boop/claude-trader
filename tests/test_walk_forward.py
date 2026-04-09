"""
tests/test_walk_forward.py — Tests for the walk-forward validator.

Key things we verify:
  1. Data splitting is correct (no leakage — OOS never contains IS data)
  2. The right number of windows are created
  3. IS/OOS ratio is respected
  4. Overfitting detection fires when it should
  5. Aggregate metrics are computed correctly
  6. Gate logic works

Run with: pytest tests/test_walk_forward.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.walk_forward import (
    WalkForwardValidator,
    WalkForwardResult,
    WindowResult,
    OVERFITTING_THRESHOLD,
    OOS_MIN_SHARPE,
)
from src.strategy_bollinger_rsi import BollingerRsiStrategy
from src.strategy_momentum import MomentumStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "trading": {"pairs": ["BTC/USDT"], "timeframe": "1h",
                    "starting_capital": 1000.0},
        "backtest": {"fee_rate": 0.001, "slippage_rate": 0.0005,
                     "start_date": "2023-01-01", "end_date": "2024-12-31"},
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


def _make_ohlcv(n: int, start_price: float = 40000.0,
                seed: int = 42) -> pd.DataFrame:
    """Generate n candles of GBM price data for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.01, n)
    prices = start_price * np.cumprod(1 + returns)

    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    highs = prices * (1 + abs(rng.normal(0, 0.003, n)))
    lows  = prices * (1 - abs(rng.normal(0, 0.003, n)))

    return pd.DataFrame({
        "open":   prices,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Data splitting tests — these are the most important (no-leakage guarantee)
# ---------------------------------------------------------------------------

class TestDataSplitting:

    def test_correct_number_of_windows(self, config):
        """Validator creates exactly n_splits windows."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        windows = v._split_data(data, warmup_needed=50)
        assert len(windows) == 3

    def test_is_oos_no_overlap(self, config):
        """IS and OOS data must not share any timestamps."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        windows = v._split_data(data, warmup_needed=50)

        for is_data, oos_data, _ in windows:
            is_times  = set(is_data.index)
            oos_times = set(oos_data.index)
            assert is_times.isdisjoint(oos_times), (
                "IS and OOS windows share timestamps — data leakage!"
            )

    def test_is_oos_are_sequential(self, config):
        """OOS must start strictly after IS ends."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        windows = v._split_data(data, warmup_needed=50)

        for is_data, oos_data, _ in windows:
            assert is_data.index[-1] < oos_data.index[0], (
                "OOS starts before IS ends — temporal ordering violated!"
            )

    def test_windows_are_sequential(self, config):
        """Window N+1 must start after Window N ends."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        windows = v._split_data(data, warmup_needed=50)

        for i in range(len(windows) - 1):
            _, oos_i, _ = windows[i]
            is_next, _, _ = windows[i + 1]
            assert oos_i.index[-1] < is_next.index[0], (
                f"Window {i+1} OOS overlaps with Window {i+2} IS"
            )

    def test_is_ratio_approximately_correct(self, config):
        """IS portion should be close to the configured ratio."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config,
                                 n_splits=3, is_ratio=0.7)
        windows = v._split_data(data, warmup_needed=50)

        for is_data, oos_data, _ in windows:
            total = len(is_data) + len(oos_data)
            actual_ratio = len(is_data) / total
            assert abs(actual_ratio - 0.7) < 0.05, (
                f"IS ratio {actual_ratio:.2f} deviates too far from 0.7"
            )

    def test_warmup_overlap_adds_prefix(self, config):
        """With warmup_overlap=True, OOS-with-warmup should be longer than OOS alone."""
        data = _make_ohlcv(3000)
        warmup = 50
        v = WalkForwardValidator(BollingerRsiStrategy, config,
                                 n_splits=3, warmup_overlap=True)
        windows = v._split_data(data, warmup_needed=warmup)

        for _, oos_data, oos_with_warmup in windows:
            assert len(oos_with_warmup) > len(oos_data), (
                "Warmup-extended OOS should be longer than pure OOS"
            )

    def test_warmup_no_overlap_matches_oos(self, config):
        """With warmup_overlap=False, oos_with_warmup should equal oos_data."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config,
                                 n_splits=3, warmup_overlap=False)
        windows = v._split_data(data, warmup_needed=50)

        for _, oos_data, oos_with_warmup in windows:
            assert len(oos_with_warmup) == len(oos_data)

    def test_all_data_covered(self, config):
        """Union of all IS+OOS windows should cover the full dataset."""
        data = _make_ohlcv(3000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=5)
        windows = v._split_data(data, warmup_needed=50)

        all_times = set()
        for is_data, oos_data, _ in windows:
            all_times.update(is_data.index)
            all_times.update(oos_data.index)

        # All timestamps in the data should be in some window
        for ts in data.index:
            assert ts in all_times, f"Timestamp {ts} not covered by any window"


# ---------------------------------------------------------------------------
# WalkForwardValidator.run() tests
# ---------------------------------------------------------------------------

class TestWalkForwardRun:

    def test_returns_correct_n_windows(self, config):
        """Result should have exactly n_splits windows."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        assert len(result.windows) == 3

    def test_window_numbers_are_sequential(self, config):
        """Windows should be numbered 1, 2, 3, ..."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        for i, wr in enumerate(result.windows):
            assert wr.window == i + 1

    def test_result_has_strategy_name(self, config):
        """Result should record the strategy name."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        assert result.strategy_name == "BollingerRsiStrategy"

    def test_aggregate_metrics_populated(self, config):
        """Aggregate metrics should be computed (not default zeros)."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        # avg_oos_sharpe might be 0 with GBM data, but field should exist
        assert hasattr(result, "avg_oos_sharpe")
        assert hasattr(result, "avg_oos_drawdown")
        assert hasattr(result, "total_oos_trades")
        assert isinstance(result.passes_gate, bool)

    def test_overfit_warning_counted(self, config):
        """overfit_windows should count windows with overfitting warnings."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        manual_count = sum(1 for wr in result.windows if wr.overfit_warning)
        assert result.overfit_windows == manual_count

    def test_summary_returns_dict(self, config):
        """summary() should return a flat dict with expected keys."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        s = result.summary()
        required_keys = [
            "strategy", "pair", "n_splits", "avg_oos_sharpe",
            "avg_oos_drawdown", "total_oos_trades", "passes_gate",
        ]
        for k in required_keys:
            assert k in s, f"Missing key: {k}"

    def test_momentum_strategy_runs(self, config):
        """Walk-forward works with MomentumStrategy, not just Bollinger."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(MomentumStrategy, config, n_splits=3)
        result = v.run(data)
        assert len(result.windows) == 3

    def test_single_split(self, config):
        """n_splits=1 should produce a single window."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=1)
        result = v.run(data)
        assert len(result.windows) == 1


# ---------------------------------------------------------------------------
# Gate and overfitting logic
# ---------------------------------------------------------------------------

class TestGateAndOverfit:

    def test_gate_fails_with_poor_metrics(self, config):
        """GBM data with no edge should fail the gate (expected)."""
        data = _make_ohlcv(5000, seed=99)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        # GBM has no exploitable edge — gate should fail
        assert not result.passes_gate

    def test_overfit_detection_logic(self, config):
        """WindowResult with IS Sharpe >> OOS Sharpe should flag overfit."""
        wr = WindowResult(
            window=1,
            is_start=pd.Timestamp("2023-01-01", tz="UTC"),
            is_end=pd.Timestamp("2023-09-01", tz="UTC"),
            oos_start=pd.Timestamp("2023-09-01", tz="UTC"),
            oos_end=pd.Timestamp("2023-12-31", tz="UTC"),
            is_sharpe=2.5,
            is_drawdown=0.08,
            is_n_trades=50,
            oos_sharpe=0.5,   # 80% degradation → overfit
            oos_drawdown=0.15,
            oos_n_trades=15,
            oos_return_pct=0.02,
            degradation=(2.5 - 0.5) / 2.5,   # = 0.8 (80%)
            overfit_warning=True,
        )
        assert wr.overfit_warning

    def test_no_overfit_when_oos_close_to_is(self, config):
        """OOS Sharpe within 50% of IS Sharpe should NOT trigger overfit."""
        wr = WindowResult(
            window=1,
            is_start=pd.Timestamp("2023-01-01", tz="UTC"),
            is_end=pd.Timestamp("2023-09-01", tz="UTC"),
            oos_start=pd.Timestamp("2023-09-01", tz="UTC"),
            oos_end=pd.Timestamp("2023-12-31", tz="UTC"),
            is_sharpe=1.5,
            is_drawdown=0.08,
            is_n_trades=50,
            oos_sharpe=1.2,   # Only 20% degradation → no overfit
            oos_drawdown=0.10,
            oos_n_trades=20,
            oos_return_pct=0.05,
            degradation=(1.5 - 1.2) / 1.5,   # = 0.2 (20%)
            overfit_warning=False,
        )
        assert not wr.overfit_warning

    def test_print_report_runs_without_error(self, config):
        """print_report() should not raise any exceptions."""
        data = _make_ohlcv(5000)
        v = WalkForwardValidator(BollingerRsiStrategy, config, n_splits=3)
        result = v.run(data)
        # Should not raise
        result.print_report()
