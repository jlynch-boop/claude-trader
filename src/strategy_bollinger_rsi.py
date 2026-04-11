"""
strategy_bollinger_rsi.py — Mean reversion strategy using Bollinger Bands + RSI.

**The idea:**
When price moves far from its average (touches the lower Bollinger Band)
AND momentum confirms it's oversold (RSI < rsi_oversold), the price has
probably moved too far and is likely to snap back to the mean.

**Entry:**
  BUY when:
    - Close price ≤ lower Bollinger Band (price at statistical extreme low)
    - RSI < rsi_oversold (30) — momentum confirms oversold condition
    - No open position in this pair

**Exit — take profit (mean reversion target):**
  The natural target for mean reversion is the MIDDLE band (the SMA of the
  Bollinger Band period). This is much more achievable than the upper band:
    - Lower band → middle band = 1 standard deviation (~5-10% on daily BTC)
    - Lower band → upper band = 2 standard deviations (~10-20% on daily BTC)
  By targeting the middle band, we exit as soon as the "snap-back" completes
  rather than waiting for a full overbought extreme that may take much longer.

**Exit — stop loss:**
  atr_stop_multiplier × ATR below entry. ATR adapts to current volatility.
  Default 2.0× ATR to keep losses tight when the mean reversion fails.

**Trend filter (optional, disabled by default):**
  When trend_ma > 0, only buy dips when price is above the trend MA.
  NOTE: This filter is often too restrictive — RSI<30 means price is already
  in a downswing, which usually puts it below any short-term MA. Use with care
  or set trend_ma = 0 (disabled) for pure mean reversion.

**Config params (from config.yaml → strategies → bollinger_rsi):**
  bb_period:            20   Bollinger Band lookback (default: days on daily data)
  bb_std:               2.0  Standard deviations for band width
  rsi_period:           14   RSI lookback
  rsi_oversold:         30   RSI buy threshold
  rsi_overbought:       70   RSI sell threshold (for the upper-band SELL signal)
  atr_period:           14   ATR lookback for stop sizing
  atr_stop_multiplier:  2.0  Stop = entry - (ATR × multiplier)
  trend_ma:             0    Only BUY when price > this SMA (0 = disabled)
"""

import pandas as pd

from src.strategy_base import Strategy, Signal, SignalType
from src.indicators import bollinger_bands, rsi, atr, sma


class BollingerRsiStrategy(Strategy):
    """
    Mean reversion using Bollinger Bands + RSI.

    Buys at the lower band when oversold; exits at the middle band (mean
    reversion confirmed) or when stop loss is hit.

    Parameters are read from config['strategies']['bollinger_rsi'].
    """

    def __init__(self, config: dict):
        super().__init__(config)
        params = config["strategies"]["bollinger_rsi"]
        self.bb_period      = params["bb_period"]                     # e.g. 20
        self.bb_std         = params["bb_std"]                        # e.g. 2.0
        self.rsi_period     = params["rsi_period"]                    # e.g. 14
        self.rsi_oversold   = params["rsi_oversold"]                  # e.g. 30
        self.rsi_overbought = params["rsi_overbought"]                # e.g. 70
        self.atr_period     = params.get("atr_period", 14)            # e.g. 14
        self.atr_multiplier = params.get("atr_stop_multiplier", 2.0)  # e.g. 2.0
        self.trend_ma       = params.get("trend_ma", 0)               # 0 = disabled

    def required_history(self) -> int:
        """
        Need enough candles for all indicators to warm up.
        When trend_ma = 0 the filter is disabled and doesn't add warmup.
        """
        base = max(self.bb_period, self.rsi_period, self.atr_period)
        if self.trend_ma > 0:
            base = max(base, self.trend_ma)
        return base + 10

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        """
        Evaluate Bollinger Bands + RSI on the current candle.

        BUY at lower band + oversold RSI.
        Take profit targets the middle band (mean reversion complete).
        Stop loss is ATR-based.

        Returns BUY, SELL, or HOLD.
        """
        close = history["close"]
        high  = history["high"]
        low   = history["low"]

        # Calculate indicators on full history (no lookahead)
        upper, middle, lower = bollinger_bands(close, self.bb_period, self.bb_std)
        rsi_values           = rsi(close, self.rsi_period)
        atr_values           = atr(high, low, close, self.atr_period)

        current_close  = candle["close"]
        current_rsi    = rsi_values.iloc[-1]
        current_upper  = upper.iloc[-1]
        current_lower  = lower.iloc[-1]
        current_middle = middle.iloc[-1]   # Mean reversion target
        current_atr    = atr_values.iloc[-1]

        # Need core indicators to be valid
        if pd.isna(current_rsi) or pd.isna(current_lower) or pd.isna(current_atr):
            return Signal.hold(reason="indicators not yet warmed up")

        # --- Optional trend filter ---
        # Only active when trend_ma > 0. When disabled, trades in any direction.
        if self.trend_ma > 0:
            trend_values  = sma(close, self.trend_ma)
            current_trend = trend_values.iloc[-1]
            if pd.isna(current_trend):
                return Signal.hold(reason="trend MA not yet warmed up")
            in_uptrend = current_close > current_trend
        else:
            in_uptrend    = True   # filter disabled
            current_trend = None

        # --- BUY condition ---
        price_at_lower = current_close <= current_lower
        rsi_oversold   = current_rsi < self.rsi_oversold

        if price_at_lower and rsi_oversold and in_uptrend:
            stop_loss   = current_close - (self.atr_multiplier * current_atr)
            stop_loss   = max(stop_loss, current_close * 0.85)  # Hard floor -15%

            # Take profit = middle band (the SMA, natural mean reversion target)
            # This is ~1 std deviation above entry — much more achievable than
            # the 6×ATR target (which could be 2-3× beyond the middle band)
            take_profit = current_middle

            trend_info = (
                f" [above {current_trend:.0f} SMA{self.trend_ma}]"
                if self.trend_ma > 0 else ""
            )
            return Signal.buy(
                pair=self._get_pair(),
                price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=(
                    f"BB lower + RSI {current_rsi:.1f} oversold"
                    f"{trend_info}"
                    f" → TP at middle band {current_middle:.0f}"
                ),
            )

        # --- SELL condition ---
        # Also generate a sell signal when price hits the upper band + overbought RSI
        # (handles the case where price overshoots the middle band target)
        price_at_upper = current_close >= current_upper
        rsi_overbought = current_rsi > self.rsi_overbought

        if price_at_upper and rsi_overbought:
            return Signal.sell(
                pair=self._get_pair(),
                price=current_close,
                reason=f"BB upper + RSI {current_rsi:.1f} overbought",
            )

        return Signal.hold()

    def _get_pair(self) -> str:
        """Get the first configured trading pair."""
        return self.config["trading"]["pairs"][0]
