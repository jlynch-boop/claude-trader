"""
paper_trader.py — Simulated live trading using real exchange data.

Paper trading runs the strategy in real-time with real market data, but
uses simulated fills instead of real orders. This lets you verify that
the strategy actually works on live data before risking real money.

**How it works:**
  1. Load recent historical data from the database (for warmup)
  2. Wait for the next hourly candle to close
  3. Fetch that candle from the exchange
  4. Run it through the strategy + risk manager
  5. Simulate any fills (with slippage + fees)
  6. Log trades and portfolio snapshots to SQLite
  7. Print a status update and repeat

**Two modes:**
  - Live mode (default): connects to exchange, waits for real candles.
    Requires internet access and a configured exchange in config.yaml.
  - Simulate mode (--simulate): replays historical data from the database
    at configurable speed. No internet needed. Useful for testing and for
    running in environments without exchange access.

**Stopping:**
  Press Ctrl+C to stop cleanly. All positions are left open (paper positions
  are just records — no real orders to cancel).

Usage:
    from src.paper_trader import PaperTrader
    from src.strategy_bollinger_rsi import BollingerRsiStrategy

    strategy = BollingerRsiStrategy(config)
    trader   = PaperTrader(strategy, config, db)
    trader.run()
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import numpy as np

from src.strategy_base import Signal, SignalType, Portfolio
from src.risk_manager import RiskManager
from src.database import Database
from src.utils import get_logger

logger = get_logger(__name__)


class PaperTrader:
    """
    Runs a strategy on live (or replayed) market data with simulated fills.

    All trades and portfolio snapshots are logged to the SQLite database
    so you can review them with the dashboard.

    Args:
        strategy:         An instantiated Strategy subclass.
        config:           Full config dict (from config.yaml).
        db:               Database instance for logging.
        simulate_delay_s: In simulate mode, pause this many seconds between
                          candles (default 0 = as fast as possible).
    """

    def __init__(
        self,
        strategy,
        config: dict,
        db: Database,
        simulate_delay_s: float = 0.0,
    ):
        self.strategy         = strategy
        self.config           = config
        self.db               = db
        self.simulate_delay   = simulate_delay_s
        self.fee_rate         = config["backtest"]["fee_rate"]
        self.slippage         = config["backtest"]["slippage_rate"]
        self.risk_manager     = RiskManager(config)

        self.pair      = config["trading"]["pairs"][0]
        self.timeframe = config["trading"]["timeframe"]
        capital        = config["trading"]["starting_capital"]

        self.portfolio = Portfolio(
            equity=capital,
            cash=capital,
            open_positions={},
            peak_equity=capital,
        )

        # Rolling history window (for passing to strategy.on_candle)
        self.history: pd.DataFrame = pd.DataFrame()

        # Stats counters
        self.candles_processed = 0
        self.total_trades      = 0

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self, simulate: bool = False, max_candles: Optional[int] = None) -> None:
        """
        Main paper trading loop.

        Args:
            simulate:    If True, replay historical data from the DB instead
                         of fetching from the exchange.
            max_candles: Stop after this many candles (None = run forever).
                         Useful for testing and simulation.
        """
        logger.info(
            f"Paper trader starting: {self.strategy.name} on {self.pair} "
            f"({'SIMULATE' if simulate else 'LIVE'} mode)"
        )

        # Load warmup history from the database
        self._load_history()

        if simulate:
            self._run_simulate(max_candles)
        else:
            self._run_live(max_candles)

    # -------------------------------------------------------------------------
    # Mode implementations
    # -------------------------------------------------------------------------

    def _run_live(self, max_candles: Optional[int]) -> None:
        """
        Live mode: wait for real candles from the exchange.

        Waits until the next hourly candle closes, fetches it, processes it.
        Retries on exchange errors with exponential backoff.
        """
        try:
            import ccxt
        except ImportError:
            raise RuntimeError("ccxt not installed. Run: pip install ccxt")

        exchange = self._connect_exchange()

        logger.info(
            f"Connected to {self.config['exchange']['name']}. "
            f"Starting capital: ${self.portfolio.equity:,.2f}"
        )

        iterations = 0
        while max_candles is None or iterations < max_candles:
            try:
                # Wait for next candle
                self._wait_for_next_candle()

                # Fetch the just-closed candle
                candle = self._fetch_latest_candle(exchange)
                if candle is None:
                    logger.warning("Could not fetch candle, skipping")
                    continue

                # Process it
                self._process_candle(candle)
                self._log_snapshot()
                self._print_status()
                iterations += 1

            except KeyboardInterrupt:
                logger.info("\nPaper trader stopped by user (Ctrl+C).")
                logger.info(
                    f"Final equity: ${self.portfolio.equity:,.2f}  |  "
                    f"Total trades: {self.total_trades}"
                )
                break
            except Exception as e:
                logger.error(f"Error on candle {iterations}: {e}. Retrying in 60s.")
                time.sleep(60)

    def _run_simulate(self, max_candles: Optional[int]) -> None:
        """
        Simulate mode: replay historical data from the database.

        Skips the warmup portion and processes candles as fast as possible
        (or at simulate_delay_s per candle for visual inspection).
        """
        warmup = self.strategy.required_history()

        # Load the full history for simulation (not just the warmup tail)
        all_data = self.db.get_ohlcv(self.pair, self.timeframe)
        if all_data.empty:
            logger.error("No historical data in database. Run generate_seed_data.py first.")
            return

        if len(all_data) <= warmup:
            logger.error(
                f"Not enough historical data for simulation. "
                f"Need > {warmup} candles, have {len(all_data)}."
            )
            return

        # Start from after warmup so indicators are ready immediately
        sim_data = all_data.iloc[warmup:]
        if max_candles is not None:
            sim_data = sim_data.iloc[:max_candles]

        logger.info(
            f"Simulating {len(sim_data)} candles "
            f"({sim_data.index[0].date()} → {sim_data.index[-1].date()})"
        )

        try:
            for i, (dt, row) in enumerate(sim_data.iterrows()):
                # History available up to this candle
                history_slice = all_data.loc[:dt]
                self._process_candle_with_history(row, history_slice)
                self._log_snapshot()

                if (i + 1) % 100 == 0 or (i + 1) == len(sim_data):
                    self._print_status(dt)

                if self.simulate_delay > 0:
                    time.sleep(self.simulate_delay)

        except KeyboardInterrupt:
            logger.info("\nSimulation stopped by user (Ctrl+C).")

        logger.info(
            f"Simulation complete: {self.candles_processed} candles, "
            f"{self.total_trades} trades, "
            f"final equity=${self.portfolio.equity:,.2f}"
        )
        self._print_status()

    # -------------------------------------------------------------------------
    # Candle processing
    # -------------------------------------------------------------------------

    def _process_candle(self, candle: pd.Series) -> None:
        """
        Process a live candle: append to history, run strategy, handle fills.
        """
        # Append to rolling history
        self.history = pd.concat([
            self.history,
            candle.to_frame().T,
        ]).tail(self.strategy.required_history() * 3)
        self.history.index = pd.to_datetime(self.history.index, utc=True)

        self._process_candle_with_history(candle, self.history)

    def _process_candle_with_history(
        self, candle: pd.Series, history: pd.DataFrame
    ) -> None:
        """
        Core processing logic: strategy → risk manager → simulate fill → log.
        """
        # --- 1. Check stop loss / take profit for open positions ---
        self._check_exits(candle)

        # --- 2. Update equity (mark-to-market) ---
        self.portfolio.equity = self._calculate_equity(candle["close"])
        self.portfolio.peak_equity = max(
            self.portfolio.peak_equity, self.portfolio.equity
        )

        # --- 3. Get strategy signal ---
        signal = self.strategy.on_candle(candle, history)

        # --- 4. Pass through risk manager ---
        if signal.is_actionable():
            if not signal.pair:
                signal.pair = self.pair
            approved = self.risk_manager.check_signal(signal, self.portfolio)

            if approved and approved.type == SignalType.BUY:
                self._execute_buy(approved, candle)

            elif approved and approved.type == SignalType.SELL:
                self._execute_sell(approved, candle, "signal")

        self.candles_processed += 1

    def _check_exits(self, candle: pd.Series) -> None:
        """Check stop loss and take profit for all open positions."""
        check_price = candle["open"]

        for pair_name in list(self.portfolio.open_positions.keys()):
            pos = self.portfolio.open_positions[pair_name]
            hit_stop = pos["stop_loss"] > 0 and check_price <= pos["stop_loss"]
            hit_tp   = (pos["take_profit"] > 0
                        and check_price >= pos["take_profit"])

            if hit_stop or hit_tp:
                reason = "stop_loss" if hit_stop else "take_profit"
                fill_price = pos["stop_loss"] if hit_stop else pos["take_profit"]

                fake_signal = Signal(
                    type=SignalType.SELL,
                    pair=pair_name,
                    price=fill_price,
                    reason=reason,
                )
                self._execute_sell(fake_signal, candle, reason)

    def _execute_buy(self, signal: Signal, candle: pd.Series) -> None:
        """Simulate a buy fill and record it."""
        fill_price     = signal.price * (1 + self.slippage)
        position_value = min(
            self.portfolio.equity * signal.size_pct,
            self.portfolio.cash,
        )
        if position_value <= 0:
            return

        amount = position_value / fill_price
        fee    = position_value * self.fee_rate

        self.portfolio.cash -= (position_value + fee)
        self.portfolio.open_positions[signal.pair] = {
            "amount":      amount,
            "entry_price": fill_price,
            "entry_time":  candle.name if hasattr(candle, "name") else datetime.now(timezone.utc),
            "entry_fee":   fee,
            "stop_loss":   signal.stop_loss,
            "take_profit": signal.take_profit,
        }

        dt_str = str(candle.name) if hasattr(candle, "name") else ""
        logger.info(
            f"PAPER BUY  {signal.pair} @ {fill_price:,.2f}  "
            f"qty={amount:.6f}  fee=${fee:.2f}  "
            f"stop={signal.stop_loss:.2f}  {dt_str}"
        )

        self._log_trade("buy", signal.pair, fill_price, amount, fee, None, candle)
        self.total_trades += 1

    def _execute_sell(self, signal: Signal, candle: pd.Series,
                      reason: str) -> None:
        """Simulate a sell fill, compute P&L, and record it."""
        if signal.pair not in self.portfolio.open_positions:
            return

        pos        = self.portfolio.open_positions.pop(signal.pair)
        fill_price = signal.price * (1 - self.slippage)
        fee        = fill_price * pos["amount"] * self.fee_rate
        gross_pnl  = (fill_price - pos["entry_price"]) * pos["amount"]
        net_pnl    = gross_pnl - pos["entry_fee"] - fee

        proceeds = fill_price * pos["amount"] - fee
        self.portfolio.cash += proceeds

        dt_str = str(candle.name) if hasattr(candle, "name") else ""
        logger.info(
            f"PAPER SELL {signal.pair} @ {fill_price:,.2f}  "
            f"pnl={net_pnl:+.2f} ({net_pnl / (pos['entry_price'] * pos['amount']):+.1%})  "
            f"reason={reason}  {dt_str}"
        )

        self._log_trade("sell", signal.pair, fill_price, pos["amount"],
                        fee, net_pnl, candle)
        self.total_trades += 1

    # -------------------------------------------------------------------------
    # Live helpers
    # -------------------------------------------------------------------------

    def _connect_exchange(self):
        """Create and return a ccxt exchange instance."""
        import ccxt

        ex_cfg  = self.config["exchange"]
        ex_name = ex_cfg["name"]

        exchange_class = getattr(ccxt, ex_name, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {ex_name}")

        params = {"enableRateLimit": True}
        # Add API keys if present (public endpoints work without them)
        if ex_cfg.get("api_key"):
            params["apiKey"] = ex_cfg["api_key"]
        if ex_cfg.get("api_secret"):
            params["secret"] = ex_cfg["api_secret"]

        ex = exchange_class(params)

        # Never use sandbox for paper trading (we want real price data)
        return ex

    def _wait_for_next_candle(self) -> None:
        """Sleep until the next candle period starts (plus a small buffer)."""
        now       = datetime.now(timezone.utc)
        # Next full hour
        next_hour = (now + timedelta(hours=1)).replace(
            minute=0, second=5, microsecond=0
        )
        wait_s = (next_hour - now).total_seconds()

        logger.info(
            f"Next candle in {wait_s / 60:.1f} min "
            f"(at {next_hour.strftime('%H:%M UTC')})"
        )
        time.sleep(max(wait_s, 1))

    def _fetch_latest_candle(self, exchange) -> Optional[pd.Series]:
        """Fetch the most recently closed candle from the exchange."""
        try:
            raw = exchange.fetch_ohlcv(self.pair, self.timeframe, limit=2)
            if not raw or len(raw) < 1:
                return None

            # Index 0 is the most recently CLOSED candle
            row = raw[0]
            ts = pd.Timestamp(row[0], unit="ms", tz="UTC")
            candle = pd.Series({
                "open":   row[1],
                "high":   row[2],
                "low":    row[3],
                "close":  row[4],
                "volume": row[5],
            }, name=ts)

            logger.debug(
                f"Fetched candle {ts}: close={row[4]:,.2f}"
            )
            return candle

        except Exception as e:
            logger.error(f"Failed to fetch candle: {e}")
            return None

    # -------------------------------------------------------------------------
    # History and equity
    # -------------------------------------------------------------------------

    def _load_history(self) -> None:
        """
        Load recent candles from the DB to use as warmup data.
        The strategy needs at least required_history() candles to compute
        indicators before it can generate signals.
        """
        warmup  = self.strategy.required_history()
        # Load 3× warmup so the first strategy call has plenty of history
        needed  = warmup * 3
        df      = self.db.get_ohlcv(self.pair, self.timeframe)

        if df.empty:
            raise RuntimeError(
                f"No historical data for {self.pair}. "
                "Run: python scripts/generate_seed_data.py"
            )

        self.history = df.tail(needed).copy()
        logger.info(
            f"Loaded {len(self.history)} warmup candles "
            f"(up to {self.history.index[-1].date()})"
        )

    def _calculate_equity(self, current_price: float) -> float:
        """Cash + mark-to-market value of all open positions."""
        pos_value = sum(
            p["amount"] * current_price
            for p in self.portfolio.open_positions.values()
        )
        return self.portfolio.cash + pos_value

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_trade(
        self,
        side: str,
        pair: str,
        price: float,
        amount: float,
        fee: float,
        pnl: Optional[float],
        candle: pd.Series,
    ) -> None:
        """Write a trade record to the SQLite database."""
        try:
            ts = candle.name if hasattr(candle, "name") else datetime.now(timezone.utc)
            if isinstance(ts, pd.Timestamp):
                ts_ms  = int(ts.timestamp() * 1000)
                dt_str = ts.isoformat()
            else:
                ts_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
                dt_str = datetime.now(timezone.utc).isoformat()

            with self.db._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO trades
                      (strategy, pair, side, price, amount, fee,
                       timestamp, datetime, pnl, is_paper, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        self.strategy.name, pair, side, price,
                        amount, fee, ts_ms, dt_str, pnl, None,
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to log trade to DB: {e}")

    def _log_snapshot(self) -> None:
        """Write a portfolio snapshot to the SQLite database."""
        try:
            now    = datetime.now(timezone.utc)
            ts_ms  = int(now.timestamp() * 1000)
            dt_str = now.isoformat()

            pos_value = self.portfolio.equity - self.portfolio.cash
            dd        = self.risk_manager.current_drawdown(
                self.portfolio.equity, self.portfolio.peak_equity
            )

            with self.db._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO portfolio_snapshots
                      (strategy, timestamp, datetime, equity, cash,
                       positions_value, drawdown_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.strategy.name, ts_ms, dt_str,
                        self.portfolio.equity, self.portfolio.cash,
                        pos_value, dd,
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to log snapshot to DB: {e}")

    def _print_status(self, dt=None) -> None:
        """Print a one-line status summary."""
        starting = self.config["trading"]["starting_capital"]
        pnl      = self.portfolio.equity - starting
        pnl_pct  = pnl / starting * 100
        dd       = self.risk_manager.current_drawdown(
            self.portfolio.equity, self.portfolio.peak_equity
        )
        n_pos    = len(self.portfolio.open_positions)
        dt_str   = str(dt.date()) if dt is not None else "now"

        print(
            f"[{dt_str}]  equity=${self.portfolio.equity:>10,.2f}  "
            f"P&L={pnl:>+8.2f} ({pnl_pct:>+5.1f}%)  "
            f"DD={dd:.1%}  positions={n_pos}  trades={self.total_trades}"
        )
