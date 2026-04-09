"""
backtest_engine.py — Core backtesting simulation loop.

Simulates trading a strategy on historical OHLCV data. Every trade
includes realistic fees and slippage. No lookahead bias: the strategy
only ever sees data up to the current candle.

How it works:
  1. Load historical candles
  2. For each candle (in chronological order):
     a. Check if any open positions hit their stop loss or take profit
     b. Ask the strategy for a signal based on current + past candles
     c. Pass signal through the risk manager
     d. If approved: simulate trade execution (apply slippage + fee)
     e. Record the trade and update portfolio equity
  3. Return a BacktestResult with full trade history and equity curve

Usage:
    from src.backtest_engine import BacktestEngine
    from src.strategy_bollinger_rsi import BollingerRsiStrategy
    from src.config import load_config
    from src.database import Database

    cfg = load_config()
    db = Database()
    df = db.get_ohlcv("BTC/USDT", "1h", "2023-01-01", "2024-12-31")

    strategy = BollingerRsiStrategy(cfg)
    engine = BacktestEngine(strategy, df, cfg)
    result = engine.run()
    result.metrics.print_report()
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from src.strategy_base import Signal, SignalType, Portfolio
from src.risk_manager import RiskManager
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """A single completed trade (entry + exit pair)."""
    strategy:    str
    pair:        str
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    side:        str           # "long" (only longs for now)
    entry_price: float
    exit_price:  float
    amount:      float         # Base currency (e.g. BTC)
    entry_fee:   float         # Fee paid on entry (quote currency)
    exit_fee:    float         # Fee paid on exit (quote currency)
    pnl:         float         # Net P&L after fees (quote currency)
    pnl_pct:     float         # P&L as % of position value
    exit_reason: str           # "signal", "stop_loss", "take_profit", "end_of_data"


@dataclass
class BacktestResult:
    """Everything produced by a backtest run."""
    strategy_name:  str
    pair:           str
    start_date:     pd.Timestamp
    end_date:       pd.Timestamp
    starting_capital: float
    trades:         list[Trade]            = field(default_factory=list)
    equity_curve:   pd.Series             = field(default_factory=pd.Series)
    snapshots:      pd.DataFrame          = field(default_factory=pd.DataFrame)


class BacktestEngine:
    """
    Simulates a strategy on historical OHLCV data.

    Key design decisions:
    - Candle iteration is strictly chronological (index 0 → end)
    - Strategy receives history[:i+1] — never future candles
    - Stops and take-profits are checked at the OPEN of each candle
      (conservative: assumes worst-case fill within the candle)
    - Slippage is applied as a fixed percentage of price
    - Fees are applied round-trip (entry + exit)
    """

    def __init__(self, strategy, data: pd.DataFrame, config: dict):
        """
        Args:
            strategy: An instance of a Strategy subclass.
            data:     OHLCV DataFrame indexed by datetime (from db.get_ohlcv).
            config:   Full config dict (from config.yaml).
        """
        self.strategy     = strategy
        self.data         = data.copy()
        self.config       = config
        self.fee_rate     = config["backtest"]["fee_rate"]
        self.slippage     = config["backtest"]["slippage_rate"]
        self.starting_cap = config["trading"]["starting_capital"]
        self.risk_manager = RiskManager(config)

    def run(self) -> BacktestResult:
        """
        Execute the full backtest. Returns a BacktestResult.
        """
        if len(self.data) < self.strategy.required_history() + 1:
            raise ValueError(
                f"Not enough data: need {self.strategy.required_history()} candles "
                f"for warmup, got {len(self.data)}"
            )

        pair = self._detect_pair()
        logger.info(
            f"Starting backtest: {self.strategy.name} on {pair} "
            f"({len(self.data)} candles, ${self.starting_cap:,.0f} capital)"
        )

        # --- Portfolio state ---
        portfolio = Portfolio(
            equity=self.starting_cap,
            cash=self.starting_cap,
            open_positions={},
            peak_equity=self.starting_cap,
        )

        completed_trades: list[Trade] = []
        equity_points: list[tuple] = []  # (datetime, equity)

        warmup = self.strategy.required_history()

        for i in range(len(self.data)):
            candle = self.data.iloc[i]
            history = self.data.iloc[:i + 1]
            dt = self.data.index[i]

            # --- 1. Check stop loss / take profit for open positions ---
            exits = self._check_exits(candle, portfolio)
            for exit_trade in exits:
                completed_trades.append(exit_trade)
                self._apply_exit(exit_trade, portfolio)
                logger.debug(
                    f"  {dt.date()} EXIT {exit_trade.pair} "
                    f"pnl={exit_trade.pnl:+.2f} ({exit_trade.pnl_pct:+.1%}) "
                    f"via {exit_trade.exit_reason}"
                )

            # --- 2. Update portfolio equity (mark-to-market) ---
            portfolio.equity = self._calculate_equity(candle["close"], portfolio)
            portfolio.peak_equity = max(portfolio.peak_equity, portfolio.equity)

            # --- 3. Skip until warmup complete ---
            if i < warmup:
                equity_points.append((dt, portfolio.equity))
                continue

            # --- 4. Get signal from strategy ---
            signal = self.strategy.on_candle(candle, history)

            if signal.is_actionable():
                # Set the pair if strategy didn't
                if not signal.pair:
                    signal.pair = pair

                # --- 5. Pass signal through risk manager ---
                approved = self.risk_manager.check_signal(signal, portfolio)

                if approved and approved.type == SignalType.BUY:
                    entry_trade = self._execute_buy(approved, candle, dt, portfolio)
                    if entry_trade:
                        logger.debug(
                            f"  {dt.date()} BUY  {approved.pair} "
                            f"@ {entry_trade.entry_price:.2f} "
                            f"stop={portfolio.open_positions[approved.pair]['stop_loss']:.2f}"
                        )

                elif approved and approved.type == SignalType.SELL:
                    exit_trade = self._execute_sell(approved, candle, dt,
                                                    portfolio, "signal")
                    if exit_trade:
                        completed_trades.append(exit_trade)
                        self._apply_exit(exit_trade, portfolio)
                        logger.debug(
                            f"  {dt.date()} SELL {approved.pair} "
                            f"pnl={exit_trade.pnl:+.2f} ({exit_trade.pnl_pct:+.1%})"
                        )

            equity_points.append((dt, portfolio.equity))

        # --- 6. Close any remaining open positions at last price ---
        last_candle = self.data.iloc[-1]
        last_dt = self.data.index[-1]
        for pair_name in list(portfolio.open_positions.keys()):
            pos = portfolio.open_positions[pair_name]
            exit_price = self._apply_slippage(last_candle["close"], "sell")
            fee = exit_price * pos["amount"] * self.fee_rate
            gross_pnl = (exit_price - pos["entry_price"]) * pos["amount"]
            net_pnl = gross_pnl - pos["entry_fee"] - fee
            pnl_pct = net_pnl / (pos["entry_price"] * pos["amount"])

            t = Trade(
                strategy=self.strategy.name,
                pair=pair_name,
                entry_time=pos["entry_time"],
                exit_time=last_dt,
                side="long",
                entry_price=pos["entry_price"],
                exit_price=exit_price,
                amount=pos["amount"],
                entry_fee=pos["entry_fee"],
                exit_fee=fee,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                exit_reason="end_of_data",
            )
            completed_trades.append(t)

        # Build equity curve Series
        if equity_points:
            idx, vals = zip(*equity_points)
            equity_curve = pd.Series(vals, index=idx, name="equity")
        else:
            equity_curve = pd.Series(dtype=float)

        result = BacktestResult(
            strategy_name=self.strategy.name,
            pair=pair,
            start_date=self.data.index[0],
            end_date=self.data.index[-1],
            starting_capital=self.starting_cap,
            trades=completed_trades,
            equity_curve=equity_curve,
        )

        logger.info(
            f"Backtest complete: {len(completed_trades)} trades, "
            f"final equity=${equity_curve.iloc[-1]:,.2f}"
            if not equity_curve.empty else "Backtest complete: no trades"
        )
        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _execute_buy(self, signal: Signal, candle: pd.Series,
                     dt: pd.Timestamp, portfolio: Portfolio) -> Optional[Trade]:
        """Simulate a buy order fill with slippage and fee."""
        fill_price = self._apply_slippage(signal.price, "buy")
        position_value = portfolio.equity * signal.size_pct
        position_value = min(position_value, portfolio.cash)

        if position_value <= 0:
            return None

        amount = position_value / fill_price
        fee = position_value * self.fee_rate

        portfolio.cash -= (position_value + fee)
        portfolio.open_positions[signal.pair] = {
            "amount":      amount,
            "entry_price": fill_price,
            "entry_time":  dt,
            "entry_fee":   fee,
            "stop_loss":   signal.stop_loss,
            "take_profit": signal.take_profit,
        }

        # Return a partially-filled Trade (exit fields filled in later)
        return Trade(
            strategy=self.strategy.name,
            pair=signal.pair,
            entry_time=dt,
            exit_time=dt,      # placeholder
            side="long",
            entry_price=fill_price,
            exit_price=0.0,
            amount=amount,
            entry_fee=fee,
            exit_fee=0.0,
            pnl=0.0,
            pnl_pct=0.0,
            exit_reason="",
        )

    def _execute_sell(self, signal: Signal, candle: pd.Series,
                      dt: pd.Timestamp, portfolio: Portfolio,
                      reason: str) -> Optional[Trade]:
        """Simulate a sell order fill with slippage and fee."""
        if signal.pair not in portfolio.open_positions:
            return None

        pos = portfolio.open_positions[signal.pair]
        fill_price = self._apply_slippage(signal.price, "sell")
        fee = fill_price * pos["amount"] * self.fee_rate
        gross_pnl = (fill_price - pos["entry_price"]) * pos["amount"]
        net_pnl = gross_pnl - pos["entry_fee"] - fee
        pnl_pct = net_pnl / (pos["entry_price"] * pos["amount"])

        return Trade(
            strategy=self.strategy.name,
            pair=signal.pair,
            entry_time=pos["entry_time"],
            exit_time=dt,
            side="long",
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            amount=pos["amount"],
            entry_fee=pos["entry_fee"],
            exit_fee=fee,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )

    def _check_exits(self, candle: pd.Series,
                     portfolio: Portfolio) -> list[Trade]:
        """Check if any open positions hit stop loss or take profit."""
        exits = []
        # Use candle open as conservative fill price for stops
        check_price = candle["open"]

        for pair_name, pos in list(portfolio.open_positions.items()):
            hit_stop = pos["stop_loss"] > 0 and check_price <= pos["stop_loss"]
            hit_tp   = (pos["take_profit"] > 0
                        and check_price >= pos["take_profit"])

            if hit_stop or hit_tp:
                reason = "stop_loss" if hit_stop else "take_profit"
                fill_price = pos["stop_loss"] if hit_stop else pos["take_profit"]
                fill_price = self._apply_slippage(fill_price, "sell")
                fee = fill_price * pos["amount"] * self.fee_rate
                gross_pnl = (fill_price - pos["entry_price"]) * pos["amount"]
                net_pnl = gross_pnl - pos["entry_fee"] - fee
                pnl_pct = net_pnl / (pos["entry_price"] * pos["amount"])

                from datetime import timezone
                dt = candle.name if hasattr(candle, 'name') else pd.Timestamp.now()

                exits.append(Trade(
                    strategy=self.strategy.name,
                    pair=pair_name,
                    entry_time=pos["entry_time"],
                    exit_time=dt,
                    side="long",
                    entry_price=pos["entry_price"],
                    exit_price=fill_price,
                    amount=pos["amount"],
                    entry_fee=pos["entry_fee"],
                    exit_fee=fee,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                ))

        return exits

    def _apply_exit(self, trade: Trade, portfolio: Portfolio) -> None:
        """Update portfolio after a position is closed."""
        if trade.pair in portfolio.open_positions:
            pos = portfolio.open_positions.pop(trade.pair)
            proceeds = trade.exit_price * trade.amount - trade.exit_fee
            portfolio.cash += proceeds

    def _calculate_equity(self, current_price: float,
                          portfolio: Portfolio) -> float:
        """Total equity = cash + mark-to-market value of open positions."""
        positions_value = sum(
            pos["amount"] * current_price
            for pos in portfolio.open_positions.values()
        )
        return portfolio.cash + positions_value

    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to a fill price.
        Buys fill slightly higher, sells fill slightly lower.
        """
        if side == "buy":
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _detect_pair(self) -> str:
        """Try to get the pair name from the strategy config."""
        pairs = self.config.get("trading", {}).get("pairs", ["BTC/USDT"])
        return pairs[0] if pairs else "BTC/USDT"
