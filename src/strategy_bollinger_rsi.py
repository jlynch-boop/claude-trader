"""
strategy_bollinger_rsi.py — Mean reversion strategy using Bollinger Bands + RSI.

**The idea:**
When price moves far from its average (touches the lower Bollinger Band)
AND momentum confirms it's oversold (RSI < 30), the price has probably
moved too far and is likely to snap back. We buy there and sell when it
snaps back to the other extreme.

**Trend filter (key improvement):**
We only buy dips when we're already in an uptrend (price > trend_ma SMA).
Buying dips in a downtrend is "catching a falling knife" — the dip just
keeps going. The trend filter eliminates most of these losing trades.

**Trade logic:**
  BUY when:
    - Close price ≤ lower Bollinger Band (price at statistical extreme low)
    - RSI < rsi_oversold (30) — momentum confirms oversold condition
    - Price is ABOVE trend_ma SMA (we're in an uptrend — dip is buyable)
    - No open position in this pair

  SELL when:
    - Close price ≥ upper Bollinger Band AND RSI > rsi_overbought
    OR
    - Stop loss / take profit hit (handled by the backtest engine)

**Stop loss:**
  atr_stop_multiplier × ATR below entry. ATR adapts to current volatility.
  Default 3.0× ATR — wider than before to survive normal intra-candle noise.
  BTC hourly ATR ≈ 0.5-2% of price, so stop is ~1.5-6% below entry.

**Take profit:**
  atr_stop_multiplier × 2 × ATR above entry (2:1 reward:risk).
  Exits at a fixed distance rather than waiting for the full upper BB touch
  which may take too long during sideways markets.

**Config params (from config.yaml → strategies → bollinger_rsi):**
  bb_period:            20   Bollinger Band lookback
  bb_std:               2.0  Standard deviations for band width
  rsi_period:           14   RSI lookback
  rsi_oversold:         30   RSI buy threshold
  rsi_overbought:       70   RSI sell threshold
  atr_period:           14   ATR lookback for stop/TP sizing
  atr_stop_multiplier:  3.0  Stop = entry - (ATR × multiplier)
  trend_ma:             50   Only BUY when price > this SMA (trend filter)
"""

import pandas as pd

from src.strategy_base import Strategy, Signal, SignalType
from src.indicators import bollinger_bands, rsi, atr, sma


class BollingerRsiStrategy(Strategy):
    """
    Mean reversion using Bollinger Bands + RSI + trend filter.

    Parameters are read from config['strategies']['bollinger_rsi'].
    """

    def __init__(self, config: dict):
        super().__init__(config)
        params = config["strategies"]["bollinger_rsi"]
        self.bb_period      = params["bb_period"]                      # e.g. 20
        self.bb_std         = params["bb_std"]                         # e.g. 2.0
        self.rsi_period     = params["rsi_period"]                     # e.g. 14
        self.rsi_oversold   = params["rsi_oversold"]                   # e.g. 30
        self.rsi_overbought = params["rsi_overbought"]                 # e.g. 70
        self.atr_period     = params.get("atr_period", 14)             # e.g. 14
        self.atr_multiplier = params.get("atr_stop_multiplier", 3.0)  # e.g. 3.0
        self.trend_ma       = params.get("trend_ma", 50)               # e.g. 50

    def required_history(self) -> int:
        """
        Need enough candles for all indicators to warm up.
        The trend MA is typically the longest, so it sets the minimum.
        When trend_ma = 0 the filter is disabled and doesn't add warmup.
        """
        base = max(self.bb_period, self.rsi_period, self.atr_period)
        if self.trend_ma > 0:
            base = max(base, self.trend_ma)
        return base + 10

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        """
        Evaluate Bollinger Bands + RSI + trend filter on the current candle.

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

        # Need core indicators to be valid
        if pd.isna(current_rsi) or pd.isna(current_lower) or pd.isna(current_atr):
            return Signal.hold(reason="indicators not yet warmed up")

        # --- Trend filter (only active when trend_ma > 0) ---
        # Only buy dips when price is above the trend MA (uptrend confirmed).
        # When trend_ma = 0, the filter is disabled (buys in any direction).
        if self.trend_ma > 0:
            trend_values  = sma(close, self.trend_ma)
            current_trend = trend_values.iloc[-1]
            if pd.isna(current_trend):
                return Signal.hold(reason="trend MA not yet warmed up")
            in_uptrend = current_close > current_trend
        else:
            in_uptrend = True  # filter disabled

        # --- BUY condition ---
        price_at_lower = current_close <= current_lower
        rsi_oversold   = current_rsi < self.rsi_oversold

        if price_at_lower and rsi_oversold and in_uptrend:
            stop_loss   = current_close - (self.atr_multiplier * current_atr)
            stop_loss   = max(stop_loss, current_close * 0.85)  # Hard floor at -15%
            take_profit = current_close + (self.atr_multiplier * 2 * current_atr)

            trend_info = (
                f" [trend: {current_close:.0f} > {current_trend:.0f}]"
                if self.trend_ma > 0 else ""
            )
            return Signal.buy(
                pair=self._get_pair(),
                price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BB lower + RSI {current_rsi:.1f} oversold{trend_info}",
            )

        # --- SELL condition ---
        price_at_upper = current_close >= current_upper
        rsi_overbought = current_rsi > self.rsi_overbought

        if price_at_upper and rsi_overbought:
            return Signal.sell(
                pair=self._get_pair(),
                price=current_close,
                reason=(
                    f"BB upper + RSI {current_rsi:.1f} overbought"
                ),
            )

        return Signal.hold()

    def _get_pair(self) -> str:
        """Get the first configured trading pair."""
        return self.config["trading"]["pairs"][0]
