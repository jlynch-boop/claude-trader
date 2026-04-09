"""
strategy_base.py — Abstract base class for all trading strategies.

Every strategy in this system inherits from Strategy and implements
two methods:
  - on_candle(): receive a new candle and return a trading signal
  - required_history(): how many candles are needed before signals start

This design ensures:
  - No lookahead bias (strategy only sees candles up to the current one)
  - Consistent interface across all strategies
  - Easy swapping between strategies in the backtest engine

Usage:
    from src.strategy_base import Strategy, Signal, SignalType

    class MyStrategy(Strategy):
        def required_history(self) -> int:
            return 20

        def on_candle(self, candle, history) -> Signal:
            if some_condition:
                return Signal(
                    type=SignalType.BUY,
                    pair="BTC/USDT",
                    price=candle["close"],
                    stop_loss=candle["close"] * 0.97,
                    take_profit=candle["close"] * 1.06,
                    reason="My buy condition was met",
                )
            return Signal.hold()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class SignalType(Enum):
    """The three possible actions a strategy can signal."""
    BUY  = "BUY"   # Open a long position
    SELL = "SELL"  # Close the current position (or open short, future use)
    HOLD = "HOLD"  # Do nothing


@dataclass
class Signal:
    """
    A trading signal returned by a strategy's on_candle() method.

    The backtest engine and paper trader consume this object.
    The risk manager may modify size_pct before execution.

    Attributes:
        type:        BUY, SELL, or HOLD
        pair:        Trading pair (e.g. "BTC/USDT")
        price:       Suggested entry/exit price (usually candle close)
        stop_loss:   Price level for stop loss (required for BUY signals)
        take_profit: Price level for take profit (optional)
        size_pct:    Suggested position size as fraction of capital (0.0–1.0).
                     The risk manager will override this with proper sizing.
        reason:      Human-readable explanation — great for debugging and logs
    """
    type:        SignalType
    pair:        str         = ""
    price:       float       = 0.0
    stop_loss:   float       = 0.0
    take_profit: float       = 0.0
    size_pct:    float       = 0.0
    reason:      str         = ""

    @staticmethod
    def hold(pair: str = "", reason: str = "") -> "Signal":
        """Convenience constructor for a HOLD signal."""
        return Signal(type=SignalType.HOLD, pair=pair, reason=reason)

    @staticmethod
    def buy(pair: str, price: float, stop_loss: float,
            take_profit: float = 0.0, reason: str = "") -> "Signal":
        """Convenience constructor for a BUY signal."""
        return Signal(
            type=SignalType.BUY,
            pair=pair,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    @staticmethod
    def sell(pair: str, price: float, reason: str = "") -> "Signal":
        """Convenience constructor for a SELL signal."""
        return Signal(
            type=SignalType.SELL,
            pair=pair,
            price=price,
            reason=reason,
        )

    def is_actionable(self) -> bool:
        """Returns True if this signal requires any action (not HOLD)."""
        return self.type != SignalType.HOLD


@dataclass
class Portfolio:
    """
    Snapshot of the portfolio state passed to the risk manager.

    The backtest engine keeps this updated after every trade.

    Attributes:
        equity:          Total portfolio value (cash + positions)
        cash:            Uninvested cash available
        open_positions:  Dict of pair → position dict {amount, entry_price, stop_loss}
        peak_equity:     Highest equity value ever reached (for drawdown calc)
    """
    equity:         float = 0.0
    cash:           float = 0.0
    open_positions: dict  = field(default_factory=dict)
    peak_equity:    float = 0.0

    def position_count(self) -> int:
        """Number of currently open positions."""
        return len(self.open_positions)

    def has_position(self, pair: str) -> bool:
        """Returns True if a position is open for the given pair."""
        return pair in self.open_positions


class Strategy(ABC):
    """
    Abstract base class all strategies must inherit from.

    The strategy receives candles one at a time (chronological order)
    and returns a Signal for each one. It never sees future candles,
    which prevents lookahead bias.

    Subclasses must implement:
      - required_history(): minimum candles needed before signals start
      - on_candle(): main signal logic
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The full config dict (from config.yaml).
                    Strategy-specific params are under config["strategies"][name].
        """
        self.config = config
        self.name   = self.__class__.__name__

    @abstractmethod
    def required_history(self) -> int:
        """
        Minimum number of candles needed before this strategy can generate
        non-HOLD signals.

        The backtest engine skips the first required_history() candles to
        allow indicators to warm up. For example, a strategy using a 200-period
        SMA needs at least 200 candles before it produces reliable values.

        Returns:
            Integer number of candles.
        """
        pass

    @abstractmethod
    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        """
        Called once for every new candle. Returns a trading signal.

        Args:
            candle:  The current OHLCV candle as a pandas Series with keys:
                     open, high, low, close, volume
            history: All candles from the start up to and including the current
                     one, as a DataFrame. Shape: (n_candles, 5).
                     Never contains future data.

        Returns:
            Signal with type BUY, SELL, or HOLD.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}()"
