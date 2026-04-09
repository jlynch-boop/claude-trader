"""
risk_manager.py — Position sizing and safety guardrails.

This is the safety net between strategy signals and actual trades.
Every signal passes through here before execution.

Key responsibilities:
  1. Size positions correctly (risk 1% of equity per trade)
  2. Reject signals that would violate risk rules
  3. Trigger circuit breaker when drawdown limit is hit
  4. Enforce max position count and max position size

With $1,000 starting capital and 1% risk per trade:
  - Risk per trade = $10
  - If stop is 3% away → position size = $10 / 0.03 = $333
  - Max position (20% cap) → $200
  - So effective size = min($333, $200) = $200

This keeps losses small and manageable while learning.

Usage:
    from src.risk_manager import RiskManager
    from src.strategy_base import Portfolio

    rm = RiskManager(config)
    portfolio = Portfolio(equity=1000, cash=1000, peak_equity=1000)
    approved = rm.check_signal(signal, portfolio)
    if approved:
        size = rm.calculate_position_size(portfolio.equity, entry, stop)
"""

from src.strategy_base import Signal, SignalType, Portfolio
from src.utils import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Validates and sizes every trade before execution.

    All parameters come from config.yaml under the 'risk' section.
    """

    def __init__(self, config: dict):
        risk = config["risk"]
        self.max_position_pct  = risk["max_position_pct"]   # e.g. 0.20 (20%)
        self.max_drawdown_pct  = risk["max_drawdown_pct"]   # e.g. 0.15 (15%)
        self.stop_loss_pct     = risk["stop_loss_pct"]      # e.g. 0.03 (3%)
        self.max_open_positions = risk["max_open_positions"] # e.g. 3
        self.risk_per_trade_pct = risk["risk_per_trade_pct"] # e.g. 0.01 (1%)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def check_signal(self, signal: Signal, portfolio: Portfolio) -> Signal | None:
        """
        Validate a signal against all risk rules.

        Returns the approved signal (possibly with adjusted size_pct),
        or None if the signal is rejected.

        Rejection reasons are logged for debugging.
        """
        if signal.type == SignalType.HOLD:
            return signal  # HOLD always passes through

        if signal.type == SignalType.BUY:
            return self._check_buy(signal, portfolio)

        if signal.type == SignalType.SELL:
            return self._check_sell(signal, portfolio)

        return None

    def calculate_position_size(self, equity: float, entry_price: float,
                                stop_price: float) -> float:
        """
        Calculate position size using fixed-fractional sizing.

        We risk a fixed percentage of equity on each trade.
        Position size is determined by the distance to the stop loss.

        Formula:
            risk_amount = equity * risk_per_trade_pct
            price_risk  = entry_price - stop_price  (for longs)
            position    = risk_amount / price_risk

        Example with $1,000 equity, $42,000 entry, $40,740 stop (3% away):
            risk_amount = $1,000 * 0.01 = $10
            price_risk  = $42,000 - $40,740 = $1,260
            position    = $10 / $1,260 = 0.00794 BTC
            value       = 0.00794 * $42,000 = $333

        Then capped at max_position_pct * equity = $200.
        So final position value = $200.

        Args:
            equity:      Current total portfolio equity
            entry_price: Intended entry price
            stop_price:  Stop loss price

        Returns:
            Position value in quote currency (e.g. USDT), or 0 if invalid.
        """
        if stop_price <= 0 or entry_price <= stop_price:
            logger.warning(
                f"Invalid stop price: entry={entry_price:.4f}, stop={stop_price:.4f}"
            )
            return 0.0

        risk_amount = equity * self.risk_per_trade_pct
        price_risk  = entry_price - stop_price
        position_value = (risk_amount / price_risk) * entry_price

        # Cap at max_position_pct of equity
        max_position_value = equity * self.max_position_pct
        position_value = min(position_value, max_position_value)

        return round(position_value, 6)

    def current_drawdown(self, equity: float, peak_equity: float) -> float:
        """
        Calculate current drawdown as a fraction (0.0 to 1.0).

        Example: peak=$1000, current=$850 → drawdown=0.15 (15%)

        Args:
            equity:      Current portfolio equity
            peak_equity: Highest equity value ever reached

        Returns:
            Drawdown as a decimal (0.0 = no drawdown, 1.0 = total loss).
        """
        if peak_equity <= 0:
            return 0.0
        return max(0.0, (peak_equity - equity) / peak_equity)

    def is_circuit_breaker_active(self, equity: float,
                                  peak_equity: float) -> bool:
        """
        Returns True if drawdown has exceeded the maximum allowed.

        When the circuit breaker is active, no new BUY signals are executed.
        The trader can still SELL (close positions) to reduce exposure.

        Example: max_drawdown_pct=0.15, peak=$1000, current=$840 (16% DD)
        → circuit breaker activates → no new buys allowed.
        """
        return self.current_drawdown(equity, peak_equity) >= self.max_drawdown_pct

    # -------------------------------------------------------------------------
    # Internal checks
    # -------------------------------------------------------------------------

    def _check_buy(self, signal: Signal, portfolio: Portfolio) -> Signal | None:
        """Run all BUY-specific risk checks."""

        # 1. Circuit breaker — stop new buys if in deep drawdown
        if self.is_circuit_breaker_active(portfolio.equity, portfolio.peak_equity):
            dd = self.current_drawdown(portfolio.equity, portfolio.peak_equity)
            # Use debug (not warning) — expected behavior during drawdown periods
            logger.debug(
                f"REJECTED {signal.pair} BUY — circuit breaker active "
                f"(drawdown {dd:.1%} >= {self.max_drawdown_pct:.1%})"
            )
            return None

        # 2. Max open positions
        if portfolio.position_count() >= self.max_open_positions:
            logger.debug(
                f"REJECTED {signal.pair} BUY — max positions reached "
                f"({portfolio.position_count()}/{self.max_open_positions})"
            )
            return None

        # 3. Already have a position in this pair
        if portfolio.has_position(signal.pair):
            logger.debug(
                f"REJECTED {signal.pair} BUY — position already open"
            )
            return None

        # 4. Stop loss must be set
        if signal.stop_loss <= 0:
            logger.warning(
                f"REJECTED {signal.pair} BUY — no stop loss provided"
            )
            return None

        # 5. Stop must be below entry (we only trade longs)
        if signal.stop_loss >= signal.price:
            logger.warning(
                f"REJECTED {signal.pair} BUY — stop {signal.stop_loss:.4f} "
                f">= entry {signal.price:.4f}"
            )
            return None

        # 6. Enough cash to open a position
        min_trade_value = portfolio.equity * 0.01  # At least 1% of equity
        if portfolio.cash < min_trade_value:
            logger.debug(
                f"REJECTED {signal.pair} BUY — insufficient cash "
                f"({portfolio.cash:.2f} < {min_trade_value:.2f})"
            )
            return None

        # All checks passed — set size_pct based on position sizing
        position_value = self.calculate_position_size(
            portfolio.equity, signal.price, signal.stop_loss
        )
        signal.size_pct = min(position_value / portfolio.equity,
                              self.max_position_pct)

        logger.debug(
            f"APPROVED {signal.pair} BUY @ {signal.price:.4f} "
            f"stop={signal.stop_loss:.4f} size={signal.size_pct:.1%} "
            f"reason={signal.reason}"
        )
        return signal

    def _check_sell(self, signal: Signal, portfolio: Portfolio) -> Signal | None:
        """Run all SELL-specific risk checks."""

        # Must have an open position to sell
        if not portfolio.has_position(signal.pair):
            logger.debug(
                f"REJECTED {signal.pair} SELL — no open position"
            )
            return None

        logger.debug(
            f"APPROVED {signal.pair} SELL @ {signal.price:.4f} "
            f"reason={signal.reason}"
        )
        return signal
