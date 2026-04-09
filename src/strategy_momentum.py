"""
strategy_momentum.py — Trend-following strategy using dual moving average crossover.

**The idea:**
When a fast-moving average crosses above a slow-moving average (golden cross),
it signals the start of an uptrend. We buy and hold until the fast MA crosses
back below the slow MA (death cross).

A long-term trend filter (200-period SMA) prevents trading counter-trend:
  - We only buy during confirmed uptrends (price > 200 SMA)
  - This avoids buying into brief recoveries during bear markets

**Why this works in crypto:**
Crypto has strong, sustained directional moves (trends). This strategy
captures the middle of those moves — it misses the very start and end,
but the middle portion is the most reliable part.

**Trade logic:**
  BUY when:
    - Fast SMA crosses ABOVE slow SMA (golden cross)
    - Price is ABOVE the 200-period SMA (we're in an uptrend)
    - No open position in this pair

  SELL when:
    - Fast SMA crosses BELOW slow SMA (death cross)
    OR
    - Stop loss is hit (handled by engine)

**Stop loss:**
  2 × ATR below entry price. Wider than Bollinger RSI because this strategy
  holds trades longer — needs room to breathe through normal volatility.

**Config params (from config.yaml → strategies → momentum):**
  fast_ma:              10   Fast SMA period
  slow_ma:              30   Slow SMA period
  trend_ma:             200  Long-term trend filter period
  atr_period:           14   ATR lookback for stop loss
  atr_stop_multiplier:  2.0  Stop = entry - (ATR × multiplier)
"""

import pandas as pd

from src.strategy_base import Strategy, Signal
from src.indicators import sma, atr


class MomentumStrategy(Strategy):
    """
    Dual SMA crossover with 200-period trend filter.

    Buys on golden cross (fast > slow) when price is above 200 SMA.
    Sells on death cross (fast < slow).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        params = config["strategies"]["momentum"]
        self.fast_ma            = params["fast_ma"]             # e.g. 10
        self.slow_ma            = params["slow_ma"]             # e.g. 30
        self.trend_ma           = params["trend_ma"]            # e.g. 200
        self.atr_period         = params["atr_period"]          # e.g. 14
        self.atr_stop_mult      = params["atr_stop_multiplier"] # e.g. 2.0

        # Track previous crossover state to detect crossings
        self._prev_fast_above_slow: bool | None = None

    def required_history(self) -> int:
        """Need enough candles for the 200-period trend MA to warm up."""
        return self.trend_ma + 10

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        """
        Detect fast/slow SMA crossovers filtered by the 200-period trend.
        """
        close = history["close"]
        high  = history["high"]
        low   = history["low"]

        fast_values  = sma(close, self.fast_ma)
        slow_values  = sma(close, self.slow_ma)
        trend_values = sma(close, self.trend_ma)
        atr_values   = atr(high, low, close, self.atr_period)

        current_fast  = fast_values.iloc[-1]
        current_slow  = slow_values.iloc[-1]
        current_trend = trend_values.iloc[-1]
        current_atr   = atr_values.iloc[-1]
        current_price = candle["close"]

        if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(current_trend):
            return Signal.hold(reason="indicators warming up")

        fast_above_slow = current_fast > current_slow
        above_trend     = current_price > current_trend

        # Detect crossover: state changed from previous candle
        crossed_up   = fast_above_slow and (self._prev_fast_above_slow is False)
        crossed_down = (not fast_above_slow) and (self._prev_fast_above_slow is True)

        # Cast to Python bool — numpy bools fail `is False` identity checks
        self._prev_fast_above_slow = bool(fast_above_slow)

        # --- BUY: golden cross above trend line ---
        if crossed_up and above_trend:
            stop_loss = current_price - (self.atr_stop_mult * current_atr)
            stop_loss = max(stop_loss, current_price * 0.85)

            return Signal.buy(
                pair=self._get_pair(),
                price=current_price,
                stop_loss=stop_loss,
                reason=(
                    f"Golden cross: fast={current_fast:.2f} > slow={current_slow:.2f} "
                    f"above trend ({current_price:.2f} > {current_trend:.2f})"
                ),
            )

        # --- SELL: death cross ---
        if crossed_down:
            return Signal.sell(
                pair=self._get_pair(),
                price=current_price,
                reason=(
                    f"Death cross: fast={current_fast:.2f} < slow={current_slow:.2f}"
                ),
            )

        return Signal.hold()

    def _get_pair(self) -> str:
        return self.config["trading"]["pairs"][0]
