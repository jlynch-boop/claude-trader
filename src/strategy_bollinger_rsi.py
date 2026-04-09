"""
strategy_bollinger_rsi.py — Mean reversion strategy using Bollinger Bands + RSI.

**The idea:**
When price moves far from its average (touches the lower Bollinger Band)
AND momentum confirms it's oversold (RSI < 30), the price has probably
moved too far and is likely to snap back. We buy there and sell when it
snaps back to the other extreme.

**Why this works in crypto:**
Crypto assets show "mean reversion" — extreme moves tend to partially reverse.
Using BOTH Bollinger Bands and RSI as confirmation reduces false signals
(e.g. a band touch during a strong downtrend is filtered by RSI not being
extreme enough).

**Trade logic:**
  BUY when:
    - Close price ≤ lower Bollinger Band (price at statistical extreme low)
    - RSI < rsi_oversold (30) — momentum confirms oversold condition
    - No open position in this pair

  SELL when:
    - Close price ≥ upper Bollinger Band (price at statistical extreme high)
    - RSI > rsi_overbought (70) — momentum confirms overbought condition
    OR
    - Stop loss is hit (handled by the backtest engine automatically)

**Stop loss:**
  1.5 × ATR below entry price. ATR adapts to current volatility,
  so in calm markets the stop is tighter, in volatile markets it's wider.

**Config params (from config.yaml → strategies → bollinger_rsi):**
  bb_period:       20   Bollinger Band lookback
  bb_std:          2.0  Standard deviations for band width
  rsi_period:      14   RSI lookback
  rsi_oversold:    30   RSI buy threshold
  rsi_overbought:  70   RSI sell threshold
"""

import pandas as pd

from src.strategy_base import Strategy, Signal, SignalType
from src.indicators import bollinger_bands, rsi, atr


class BollingerRsiStrategy(Strategy):
    """
    Mean reversion using Bollinger Bands + RSI confirmation.

    Parameters are read from config['strategies']['bollinger_rsi'].
    """

    def __init__(self, config: dict):
        super().__init__(config)
        params = config["strategies"]["bollinger_rsi"]
        self.bb_period      = params["bb_period"]       # e.g. 20
        self.bb_std         = params["bb_std"]          # e.g. 2.0
        self.rsi_period     = params["rsi_period"]      # e.g. 14
        self.rsi_oversold   = params["rsi_oversold"]    # e.g. 30
        self.rsi_overbought = params["rsi_overbought"]  # e.g. 70
        self.atr_period     = 14
        self.atr_multiplier = 1.5  # Stop = entry - 1.5 * ATR

    def required_history(self) -> int:
        """
        Need enough candles for all indicators to warm up.
        Bollinger needs bb_period, RSI needs rsi_period + a few extra.
        """
        return max(self.bb_period, self.rsi_period) + 10

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        """
        Evaluate Bollinger Bands + RSI on the current candle.

        Returns BUY, SELL, or HOLD.
        """
        close = history["close"]
        high  = history["high"]
        low   = history["low"]

        # Calculate indicators on full history (no lookahead)
        upper, middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)
        rsi_values           = rsi(close, self.rsi_period)
        atr_values           = atr(high, low, close, self.atr_period)

        current_close = candle["close"]
        current_rsi   = rsi_values.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_atr   = atr_values.iloc[-1]

        # Need valid indicator values
        if pd.isna(current_rsi) or pd.isna(current_lower) or pd.isna(current_atr):
            return Signal.hold(reason="indicators not yet warmed up")

        # --- BUY condition ---
        # Price touched or crossed below the lower band AND RSI is oversold
        price_at_lower = current_close <= current_lower
        rsi_oversold   = current_rsi < self.rsi_oversold

        if price_at_lower and rsi_oversold:
            stop_loss = current_close - (self.atr_multiplier * current_atr)
            stop_loss = max(stop_loss, current_close * 0.85)  # Never > 15% stop

            return Signal.buy(
                pair=self._get_pair(),
                price=current_close,
                stop_loss=stop_loss,
                take_profit=current_upper,  # Target: upper band
                reason=(
                    f"BB lower touch ({current_close:.2f} <= {current_lower:.2f}) "
                    f"+ RSI oversold ({current_rsi:.1f})"
                ),
            )

        # --- SELL condition ---
        # Price touched or crossed above the upper band AND RSI is overbought
        price_at_upper  = current_close >= current_upper
        rsi_overbought  = current_rsi > self.rsi_overbought

        if price_at_upper and rsi_overbought:
            return Signal.sell(
                pair=self._get_pair(),
                price=current_close,
                reason=(
                    f"BB upper touch ({current_close:.2f} >= {current_upper:.2f}) "
                    f"+ RSI overbought ({current_rsi:.1f})"
                ),
            )

        return Signal.hold()

    def _get_pair(self) -> str:
        """Get the first configured trading pair."""
        return self.config["trading"]["pairs"][0]
