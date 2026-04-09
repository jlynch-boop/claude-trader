"""
walk_forward.py — Walk-forward out-of-sample validation.

Walk-forward testing is the gold standard for proving (or disproving)
that a trading strategy has a real edge — not just a curve fit to
historical data.

**The problem with regular backtests:**
A backtest on a single period can be deceiving. The strategy might have
learned patterns specific to that period. When you trade it live, it
fails because those patterns don't repeat.

**How walk-forward solves this:**
1. Split the full data into N sequential time windows (default 5)
2. For each window:
   a. In-sample (IS): first 70% — strategy "trains" (we confirm it generates signals)
   b. Out-of-sample (OOS): last 30% — blindly test performance
3. Compare IS vs OOS metrics. If OOS is much worse than IS → the strategy
   is overfit to the training data and will likely fail live.

**The key test:**
   OOS Sharpe should be ≥ 50% of IS Sharpe.
   If OOS degrades by more than 50%, that's an overfit warning.

**Important note on "training":**
Since our strategies have no tunable ML parameters (they use fixed rules),
the IS period mainly confirms the strategy generates enough trades to
evaluate. The real test is always OOS.

Usage:
    from src.walk_forward import WalkForwardValidator

    validator = WalkForwardValidator(BollingerRsiStrategy, config)
    result = validator.run(data)
    result.print_report()
"""

from dataclasses import dataclass, field
from typing import Type

import pandas as pd
import numpy as np

from src.strategy_base import Strategy
from src.backtest_engine import BacktestEngine
from src.backtest_metrics import BacktestMetrics, GATE_MIN_SHARPE, GATE_MAX_DRAWDOWN
from src.utils import get_logger

logger = get_logger(__name__)

# Overfitting warning: OOS Sharpe < IS Sharpe × (1 - OVERFITTING_THRESHOLD)
OVERFITTING_THRESHOLD = 0.50   # 50% degradation triggers warning
OOS_MIN_SHARPE        = 1.0    # Gate: OOS Sharpe must exceed this
OOS_MAX_DRAWDOWN      = 0.20   # Gate: OOS max drawdown must be below this


@dataclass
class WindowResult:
    """Results for one IS/OOS split window."""
    window:          int           # Window number (1-based)
    is_start:        pd.Timestamp
    is_end:          pd.Timestamp
    oos_start:       pd.Timestamp
    oos_end:         pd.Timestamp
    is_sharpe:       float
    is_drawdown:     float         # Max drawdown (decimal)
    is_n_trades:     int
    oos_sharpe:      float
    oos_drawdown:    float         # Max drawdown (decimal)
    oos_n_trades:    int
    oos_return_pct:  float         # OOS total return (decimal)
    degradation:     float         # (IS_sharpe - OOS_sharpe) / |IS_sharpe|
    overfit_warning: bool          # True if degradation > 50%


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward windows."""
    strategy_name:     str
    pair:              str
    n_splits:          int
    is_ratio:          float
    windows:           list[WindowResult] = field(default_factory=list)

    # Aggregate OOS statistics
    avg_oos_sharpe:    float = 0.0
    avg_oos_drawdown:  float = 0.0
    avg_oos_return:    float = 0.0
    total_oos_trades:  int   = 0
    overfit_windows:   int   = 0  # Number of windows with overfitting warning
    passes_gate:       bool  = False

    def print_report(self) -> None:
        """Print a formatted walk-forward report."""
        w = 64
        print("\n" + "=" * w)
        print(f"  WALK-FORWARD REPORT: {self.strategy_name} — {self.pair}")
        print("=" * w)
        print(f"  Splits: {self.n_splits}  |  IS ratio: {self.is_ratio:.0%}"
              f"  |  OOS ratio: {1 - self.is_ratio:.0%}")
        print("-" * w)

        # Per-window table
        header = (f"  {'Win':>3}  {'IS Period':>22}  "
                  f"{'IS Sharpe':>9}  {'OOS Sharpe':>10}  "
                  f"{'OOS DD':>6}  {'Trades':>6}  {'Overfit':>7}")
        print(header)
        print("  " + "-" * (w - 2))

        for wr in self.windows:
            is_period = f"{wr.is_start.date()} → {wr.oos_end.date()}"
            flag = "  WARN" if wr.overfit_warning else ""
            print(
                f"  {wr.window:>3}  {is_period:>22}  "
                f"{wr.is_sharpe:>+9.3f}  {wr.oos_sharpe:>+10.3f}  "
                f"{wr.oos_drawdown:>5.1%}  {wr.oos_n_trades:>6}{flag}"
            )

        print("-" * w)
        gate_str = ("PASSES — OOS edge confirmed"
                    if self.passes_gate else "FAILS  — no stable OOS edge")
        print(f"  Avg OOS Sharpe:    {self.avg_oos_sharpe:>+8.3f}"
              f"  {'✓' if self.avg_oos_sharpe >= OOS_MIN_SHARPE else '✗'}"
              f" (need > {OOS_MIN_SHARPE})")
        print(f"  Avg OOS Drawdown:  {self.avg_oos_drawdown:>7.1%}"
              f"  {'✓' if self.avg_oos_drawdown <= OOS_MAX_DRAWDOWN else '✗'}"
              f" (need < {OOS_MAX_DRAWDOWN:.0%})")
        print(f"  Avg OOS Return:    {self.avg_oos_return:>+7.1%}")
        print(f"  Total OOS Trades:  {self.total_oos_trades:>8}")
        print(f"  Overfit Warnings:  {self.overfit_windows}/{self.n_splits}")
        print("=" * w)
        print(f"  Gate check:        {gate_str}")
        print("=" * w + "\n")

    def summary(self) -> dict:
        """Return all aggregate metrics as a flat dict."""
        return {
            "strategy":          self.strategy_name,
            "pair":              self.pair,
            "n_splits":          self.n_splits,
            "avg_oos_sharpe":    round(self.avg_oos_sharpe, 3),
            "avg_oos_drawdown":  round(self.avg_oos_drawdown * 100, 2),
            "avg_oos_return":    round(self.avg_oos_return * 100, 2),
            "total_oos_trades":  self.total_oos_trades,
            "overfit_windows":   self.overfit_windows,
            "passes_gate":       self.passes_gate,
        }


class WalkForwardValidator:
    """
    Splits historical data into sequential IS/OOS windows and runs
    backtests on each. Detects overfitting by comparing IS vs OOS metrics.

    Args:
        strategy_class: The Strategy subclass to test (not an instance —
                        we instantiate it fresh for each window).
        config:         Full config dict (from config.yaml).
        n_splits:       Number of windows to split data into (default 5).
        is_ratio:       Fraction of each window used for in-sample (default 0.7).
        warmup_overlap: If True, prepend IS warmup candles to OOS window so the
                        strategy doesn't waste OOS data on indicator warmup.
    """

    def __init__(
        self,
        strategy_class: Type[Strategy],
        config:         dict,
        n_splits:       int   = 5,
        is_ratio:       float = 0.70,
        warmup_overlap: bool  = True,
    ):
        self.strategy_class = strategy_class
        self.config         = config
        self.n_splits       = n_splits
        self.is_ratio       = is_ratio
        self.warmup_overlap = warmup_overlap

    def run(self, data: pd.DataFrame) -> WalkForwardResult:
        """
        Execute walk-forward validation on the full dataset.

        Args:
            data: Full OHLCV DataFrame (from db.get_ohlcv). Must be sorted
                  chronologically (oldest first).

        Returns:
            WalkForwardResult with per-window and aggregate metrics.
        """
        # Instantiate once to get required_history and name; discard afterwards
        _temp_strategy  = self.strategy_class(self.config)
        warmup_needed   = _temp_strategy.required_history()
        strategy_name   = _temp_strategy.name
        pair            = self._detect_pair()

        logger.info(
            f"Walk-forward: {_temp_strategy.name} on {pair}, "
            f"{self.n_splits} splits, IS={self.is_ratio:.0%} / "
            f"OOS={1 - self.is_ratio:.0%}, "
            f"{len(data)} total candles"
        )

        windows = self._split_data(data, warmup_needed)
        if not windows:
            raise ValueError(
                "Not enough data to create walk-forward windows. "
                f"Need at least {warmup_needed * 2} candles per split."
            )

        window_results: list[WindowResult] = []

        for idx, (is_data, oos_data, oos_data_with_warmup) in enumerate(windows):
            win_num = idx + 1
            logger.info(
                f"  Window {win_num}/{self.n_splits}: "
                f"IS {is_data.index[0].date()} → {is_data.index[-1].date()} "
                f"({len(is_data)} candles)  |  "
                f"OOS {oos_data.index[0].date()} → {oos_data.index[-1].date()} "
                f"({len(oos_data)} candles)"
            )

            # --- Run IS backtest ---
            try:
                is_strategy = self.strategy_class(self.config)
                is_engine   = BacktestEngine(is_strategy, is_data, self.config)
                is_result   = is_engine.run()
                is_m        = BacktestMetrics(is_result)
                is_sharpe   = is_m.sharpe_ratio()
                is_dd       = is_m.max_drawdown()
                is_trades   = len(is_result.trades)
            except ValueError as e:
                logger.warning(f"  IS window {win_num} skipped: {e}")
                is_sharpe = 0.0
                is_dd     = 0.0
                is_trades = 0

            # --- Run OOS backtest ---
            # Use warmup-extended dataset so indicators are hot from candle 1
            oos_run_data = oos_data_with_warmup if self.warmup_overlap else oos_data

            try:
                oos_strategy = self.strategy_class(self.config)
                oos_engine   = BacktestEngine(oos_strategy, oos_run_data, self.config)
                oos_full_result = oos_engine.run()

                # Trim equity curve and trades to the true OOS period
                oos_result  = self._trim_to_oos(oos_full_result, oos_data.index[0])
                oos_m       = BacktestMetrics(oos_result)
                oos_sharpe  = oos_m.sharpe_ratio()
                oos_dd      = oos_m.max_drawdown()
                oos_trades  = len(oos_result.trades)
                oos_return  = oos_m.total_return()
            except ValueError as e:
                logger.warning(f"  OOS window {win_num} skipped: {e}")
                oos_sharpe = 0.0
                oos_dd     = 0.0
                oos_trades = 0
                oos_return = 0.0

            # --- Compute degradation ---
            # Degradation = how much OOS Sharpe fell relative to IS
            # A negative IS Sharpe makes the ratio misleading, so we
            # use absolute IS Sharpe as denominator (floored to avoid /0)
            abs_is = abs(is_sharpe) if abs(is_sharpe) > 0.001 else 0.001
            degradation = (is_sharpe - oos_sharpe) / abs_is
            overfit_warn = degradation > OVERFITTING_THRESHOLD

            window_results.append(WindowResult(
                window=win_num,
                is_start=is_data.index[0],
                is_end=is_data.index[-1],
                oos_start=oos_data.index[0],
                oos_end=oos_data.index[-1],
                is_sharpe=round(is_sharpe, 4),
                is_drawdown=round(is_dd, 4),
                is_n_trades=is_trades,
                oos_sharpe=round(oos_sharpe, 4),
                oos_drawdown=round(oos_dd, 4),
                oos_n_trades=oos_trades,
                oos_return_pct=round(oos_return, 4),
                degradation=round(degradation, 4),
                overfit_warning=overfit_warn,
            ))

            logger.info(
                f"    IS Sharpe={is_sharpe:+.3f}  OOS Sharpe={oos_sharpe:+.3f}  "
                f"Degradation={degradation:+.1%}  "
                f"{'OVERFIT WARNING' if overfit_warn else 'OK'}"
            )

        # --- Aggregate OOS metrics ---
        valid_oos = [wr for wr in window_results if wr.oos_n_trades > 0]

        avg_oos_sharpe   = (np.mean([wr.oos_sharpe   for wr in valid_oos])
                            if valid_oos else 0.0)
        avg_oos_drawdown = (np.mean([wr.oos_drawdown  for wr in valid_oos])
                            if valid_oos else 0.0)
        avg_oos_return   = (np.mean([wr.oos_return_pct for wr in valid_oos])
                            if valid_oos else 0.0)
        total_oos_trades = sum(wr.oos_n_trades for wr in window_results)
        overfit_windows  = sum(1 for wr in window_results if wr.overfit_warning)

        passes_gate = (
            avg_oos_sharpe   >= OOS_MIN_SHARPE
            and avg_oos_drawdown <= OOS_MAX_DRAWDOWN
            and total_oos_trades >= 10   # At least some OOS trades
        )

        wf_result = WalkForwardResult(
            strategy_name=strategy_name,
            pair=pair,
            n_splits=self.n_splits,
            is_ratio=self.is_ratio,
            windows=window_results,
            avg_oos_sharpe=round(float(avg_oos_sharpe), 4),
            avg_oos_drawdown=round(float(avg_oos_drawdown), 4),
            avg_oos_return=round(float(avg_oos_return), 4),
            total_oos_trades=total_oos_trades,
            overfit_windows=overfit_windows,
            passes_gate=passes_gate,
        )

        logger.info(
            f"Walk-forward complete: avg OOS Sharpe={avg_oos_sharpe:+.3f}, "
            f"gate={'PASS' if passes_gate else 'FAIL'}"
        )
        return wf_result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _split_data(
        self, data: pd.DataFrame, warmup_needed: int
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Divide data into n_splits windows; each window is split into IS/OOS.

        Returns a list of (is_data, oos_data, oos_data_with_warmup) tuples.
        oos_data_with_warmup prepends the last `warmup_needed` IS candles
        so the strategy's indicators are warm from the first OOS bar.
        """
        n = len(data)
        window_size = n // self.n_splits

        # Require each IS portion to be at least 2× the warmup
        min_is_size = warmup_needed * 2
        if int(window_size * self.is_ratio) < min_is_size:
            logger.warning(
                f"IS window size {int(window_size * self.is_ratio)} is less than "
                f"2× warmup ({min_is_size}). Results may be unreliable."
            )

        windows = []
        for i in range(self.n_splits):
            start_idx = i * window_size
            # Last window takes any remainder
            end_idx   = (i + 1) * window_size if i < self.n_splits - 1 else n

            window = data.iloc[start_idx:end_idx]
            is_end = int(len(window) * self.is_ratio)

            is_data  = window.iloc[:is_end]
            oos_data = window.iloc[is_end:]

            if len(is_data) < warmup_needed + 1:
                logger.warning(
                    f"Window {i + 1}: IS too small ({len(is_data)} candles, "
                    f"need {warmup_needed + 1}). Skipping."
                )
                continue
            if len(oos_data) < warmup_needed + 1:
                logger.warning(
                    f"Window {i + 1}: OOS too small ({len(oos_data)} candles, "
                    f"need {warmup_needed + 1}). Skipping."
                )
                continue

            # Build warmup-extended OOS dataset
            warmup_prefix = is_data.iloc[-warmup_needed:] if self.warmup_overlap else pd.DataFrame()
            oos_data_with_warmup = pd.concat([warmup_prefix, oos_data])

            windows.append((is_data, oos_data, oos_data_with_warmup))

        return windows

    def _trim_to_oos(
        self, result, oos_start: pd.Timestamp
    ):
        """
        Trim a BacktestResult so equity curve and trades only cover
        the true OOS period (removing the IS warmup prefix).

        When warmup_overlap=True, the backtest ran on IS-warmup + OOS data,
        so we need to strip out the warmup portion of results.
        """
        from src.backtest_engine import BacktestResult

        if not self.warmup_overlap:
            return result

        # Trim equity curve to OOS start
        trimmed_equity = result.equity_curve[result.equity_curve.index >= oos_start]

        # Recalibrate the starting capital for this OOS slice
        # Use the equity value at OOS start as the effective starting capital
        if trimmed_equity.empty:
            oos_starting_cap = result.starting_capital
        else:
            oos_starting_cap = float(trimmed_equity.iloc[0])

        # Trim trades to those that entered during OOS period
        oos_trades = [t for t in result.trades if t.entry_time >= oos_start]

        return BacktestResult(
            strategy_name=result.strategy_name,
            pair=result.pair,
            start_date=oos_start,
            end_date=result.end_date,
            starting_capital=oos_starting_cap,
            trades=oos_trades,
            equity_curve=trimmed_equity,
        )

    def _detect_pair(self) -> str:
        pairs = self.config.get("trading", {}).get("pairs", ["BTC/USDT"])
        return pairs[0] if pairs else "BTC/USDT"
