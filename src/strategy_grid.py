"""
strategy_grid.py — Grid trading strategy for ranging (sideways) markets.

**The idea:**
Instead of predicting direction, we set up a grid of buy and sell orders
at fixed price intervals. Every time price drops to a grid level, we buy
a small amount. When price rises back up to the next level, we sell for
a small profit. This repeats continuously, grinding out small gains.

**Why this works:**
Crypto markets range (move sideways without a clear trend) roughly 60-70%
of the time. During those periods, price bounces between levels repeatedly.
Each bounce earns a small profit. Many small profits add up.

**The risk:**
If price breaks strongly in one direction and exits the grid range,
we stop trading and close all positions (circuit breaker). Without this,
a strong downtrend could trap many buy positions at a loss.

**Trade logic:**
  SETUP: On the first candle, define a grid:
    - Center = current price
    - Range  = center ± (range_pct / 2)   e.g. ±5%
    - Place grid_levels evenly within the range
    - Even levels: buy levels. Odd levels: sell levels.

  Each tick:
    - If price crosses below a buy level → BUY signal
    - If we have an open position and price crosses above next sell level → SELL
    - If price exits the grid range → SELL all and reset grid (circuit breaker)

**Config params (from config.yaml → strategies → grid):**
  grid_levels:      10   Number of price levels in the grid
  grid_spread_pct:  0.02  Distance between levels as % of price (2%)
  range_pct:        0.10  Total grid range as fraction (10% = ±5%)
"""

import pandas as pd

from src.strategy_base import Strategy, Signal
from src.utils import get_logger

logger = get_logger(__name__)


class GridStrategy(Strategy):
    """
    Simple grid trading strategy.

    Buys at lower grid levels, sells at upper levels.
    Exits completely if price breaks out of the grid range.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        params = config["strategies"]["grid"]
        self.grid_levels    = params["grid_levels"]      # e.g. 10
        self.grid_spread    = params["grid_spread_pct"]  # e.g. 0.02 (2%)
        self.range_pct      = params["range_pct"]        # e.g. 0.10 (10%)
        self.pair           = params.get("pair",
                              config["trading"]["pairs"][0])

        # Grid state (set up on first signal-eligible candle)
        self._grid_initialized = False
        self._grid_levels:   list[float] = []  # Price levels
        self._grid_center:   float = 0.0
        self._grid_upper:    float = 0.0       # Circuit breaker upper limit
        self._grid_lower:    float = 0.0       # Circuit breaker lower limit
        self._prev_price:    float = 0.0
        self._next_buy_level: int = 0          # Index of next buy level (from top)
        self._in_position:   bool = False
        self._entry_price:   float = 0.0
        self._sell_level:    float = 0.0       # Price at which to take profit

    def required_history(self) -> int:
        """Grid needs very little history — just 5 candles to stabilize."""
        return 5

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        """
        Check grid levels and return BUY, SELL, or HOLD.
        """
        current_price = candle["close"]

        # Initialize grid on first eligible candle
        if not self._grid_initialized:
            self._initialize_grid(current_price)
            self._prev_price = current_price
            return Signal.hold(reason="Grid initialized")

        signal = self._evaluate_grid(current_price)
        self._prev_price = current_price
        return signal

    def _initialize_grid(self, center_price: float) -> None:
        """
        Set up the grid around the current price.

        Grid levels are evenly spaced within the range.
        Example: center=42000, range=10%, spread=2%
          Upper limit = 42000 * 1.05 = $44,100
          Lower limit = 42000 * 0.95 = $39,900
          10 levels spaced 2% apart
        """
        self._grid_center = center_price
        half_range = self.range_pct / 2
        self._grid_upper = center_price * (1 + half_range)
        self._grid_lower = center_price * (1 - half_range)

        # Generate evenly spaced grid levels from upper to lower
        step = center_price * self.grid_spread
        levels = []
        level = self._grid_upper
        while level >= self._grid_lower and len(levels) < self.grid_levels:
            levels.append(round(level, 4))
            level -= step

        self._grid_levels = levels
        self._next_buy_level = len(levels) - 1  # Start at the lowest buy level
        self._grid_initialized = True

        logger.debug(
            f"Grid initialized: center={center_price:.2f}, "
            f"upper={self._grid_upper:.2f}, lower={self._grid_lower:.2f}, "
            f"{len(levels)} levels"
        )

    def _evaluate_grid(self, current_price: float) -> Signal:
        """Check price against grid levels and return appropriate signal."""

        # --- Circuit breaker: price exited the grid range ---
        if current_price > self._grid_upper or current_price < self._grid_lower:
            if self._in_position:
                self._in_position = False
                direction = "above upper" if current_price > self._grid_upper else "below lower"
                return Signal.sell(
                    pair=self.pair,
                    price=current_price,
                    reason=f"Grid circuit breaker: price {direction} range",
                )
            # Reset grid around new price if we broke out
            self._initialize_grid(current_price)
            return Signal.hold(reason="Grid reset after breakout")

        # --- SELL: if in position and price crossed above sell level ---
        if self._in_position and current_price >= self._sell_level:
            self._in_position = False
            pnl_pct = (current_price - self._entry_price) / self._entry_price
            return Signal.sell(
                pair=self.pair,
                price=current_price,
                reason=(
                    f"Grid sell at {current_price:.2f} "
                    f"(entry={self._entry_price:.2f}, "
                    f"gain={pnl_pct:.1%})"
                ),
            )

        # --- BUY: if not in position and price crossed below a buy level ---
        if not self._in_position:
            # Find the buy level the price just crossed below
            for i, level in enumerate(self._grid_levels):
                if (self._prev_price > level >= current_price
                        and i < len(self._grid_levels) - 1):
                    # Buy here, sell at the level above
                    self._sell_level = self._grid_levels[max(0, i - 1)]
                    self._entry_price = current_price
                    stop_loss = max(
                        self._grid_lower * 0.99,  # Just below grid lower
                        current_price * 0.90,      # Never more than 10% stop
                    )
                    self._in_position = True
                    return Signal.buy(
                        pair=self.pair,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=self._sell_level,
                        reason=(
                            f"Grid buy at level {i}: {current_price:.2f} "
                            f"(target: {self._sell_level:.2f})"
                        ),
                    )

        return Signal.hold()
