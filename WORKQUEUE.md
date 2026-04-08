# Work Queue — Claude Trader

This file is the persistent task tracker. Claude reads this at the start of every session
and resumes from the first unchecked `[ ]` item.

**Legend:**
- `[x]` = Done
- `[ ]` = Not started / in progress
- `[-]` = Skipped / not applicable

---

## Phase 1: Foundation (Days 1-15)
**Goal:** Data pipeline, indicators, database. Everything needed to feed a backtester.

- [x] Set up git branch `claude/setup-and-overview-7vTle`
- [x] Create directory structure (`src/`, `scripts/`, `tests/`, `data/`, `logs/`, `notebooks/`)
- [x] `config.yaml` — all settings (exchange, pairs, risk params, strategy params)
- [x] `requirements.txt`
- [x] `.gitignore` + `.env.example`
- [x] `src/config.py` — config loader with validation
- [x] `src/utils.py` — logging setup, timestamp helpers
- [x] `src/database.py` — SQLite: ohlcv, trades, portfolio_snapshots tables
- [x] `tests/test_database.py` — 15 tests, all passing
- [x] `src/data_fetcher.py` — CCXT OHLCV downloader with pagination + resume
- [x] `scripts/fetch_data.py` — CLI entry point
- [x] `src/indicators.py` — SMA, EMA, RSI, Bollinger Bands, ATR, MACD
- [x] `tests/test_indicators.py` — 26 tests, all passing
- [x] `CLAUDE.md` — session resume instructions
- [x] `WORKQUEUE.md` — this file
- [ ] Fetch seed data: BTC/USDT 1h (2023-01-01 to present) and export to `data/BTC_USDT_1h.csv`
- [ ] Fetch seed data: ETH/USDT 1h (2023-01-01 to present) and export to `data/ETH_USDT_1h.csv`
- [ ] Add CSV export method to `src/database.py`
- [ ] Update `.gitignore` to allow `data/*.csv` (currently ignores all of `data/`)
- [ ] Commit seed CSVs to repo

**Phase 1 Status:** Foundation complete. Seed data fetch remaining.

---

## Phase 2: Backtesting Engine (Days 16-35)
**Goal:** Implement strategy framework, risk manager, and backtesting engine. Run all 3 strategies on seed data and compare results.

### Strategy Framework
- [ ] `src/strategy_base.py` — abstract base class
  - `SignalType` enum: BUY, SELL, HOLD
  - `Signal` dataclass: type, pair, price, stop_loss, take_profit, size_pct, reason
  - `Strategy` ABC: `on_candle(candle, history) -> Signal`, `required_history() -> int`

### Risk Manager
- [ ] `src/risk_manager.py` — position sizing + safety checks
  - `check_signal(signal, portfolio) -> Signal | None`
  - `calculate_position_size(equity, entry, stop) -> float` (fixed-fractional, 1% risk/trade)
  - `is_circuit_breaker_active(equity) -> bool` (stops at 15% drawdown from peak)
  - `current_drawdown(equity) -> float`
- [ ] `tests/test_risk_manager.py` — position sizing math, circuit breaker, max position limits

### Backtest Engine
- [ ] `src/backtest_engine.py` — core simulation loop
  - `BacktestResult` dataclass: equity curve, trade list
  - Iterates candles chronologically (no lookahead)
  - Simulates fills with slippage + fees
  - Calls risk manager on every signal
  - Records portfolio snapshots every N candles
- [ ] `src/backtest_metrics.py` — performance metrics
  - Sharpe ratio (annualized for hourly data)
  - Max drawdown
  - Win rate
  - Profit factor
  - Total return
  - Calmar ratio
  - `print_report()` — formatted terminal table
  - `plot_equity_curve()` — matplotlib chart
- [ ] `tests/test_backtest_engine.py` — feed known data, verify expected returns

### Strategies
- [ ] `src/strategy_bollinger_rsi.py` — mean reversion
  - BUY: price ≤ lower BB AND RSI < 30
  - SELL: price ≥ upper BB AND RSI > 70
  - Stop: 1.5x ATR below entry
- [ ] `src/strategy_momentum.py` — dual MA crossover
  - BUY: fast SMA crosses above slow SMA AND price > 200 SMA
  - SELL: fast SMA crosses below slow SMA
  - Stop: 2x ATR below entry
- [ ] `src/strategy_grid.py` — grid trading
  - Define price range, place grid levels
  - BUY at each lower grid level, SELL at each upper level
  - Circuit breaker: close all if price breaks out of range
- [ ] `tests/test_strategies.py` — verify signal generation for known patterns

### CLI + Analysis
- [ ] `scripts/run_backtest.py` — CLI: `python scripts/run_backtest.py --strategy bollinger_rsi`
- [ ] Compare all 3 strategies on BTC/USDT and ETH/USDT seed data
- [ ] Document results in a comment here

**Phase 2 Status:** Not started.

---

## Phase 3: Validation & Paper Trading (Days 36-60)
**Goal:** Walk-forward validation to prove (or disprove) edge. Paper trading with live data.

### Walk-Forward Validation
- [ ] `src/walk_forward.py` — walk-forward validator
  - Split data into N sequential windows (default 5)
  - Each window: 70% in-sample, 30% out-of-sample
  - Run backtest on each split, compare IS vs OOS metrics
  - Flag if OOS Sharpe degrades >50% vs IS (overfitting warning)
  - Return aggregate OOS metrics
- [ ] `tests/test_walk_forward.py` — verify data splitting, no leakage
- [ ] `scripts/run_walk_forward.py` — CLI entry point
- [ ] Run walk-forward on all 3 strategies, record results here:
  - `bollinger_rsi`: OOS Sharpe = TBD
  - `momentum`: OOS Sharpe = TBD
  - `grid`: OOS Sharpe = TBD
- [ ] **Gate:** Only advance strategies with OOS Sharpe > 1.0 and max drawdown < 20%

### Paper Trading
- [ ] `src/paper_trader.py` — live data + simulated execution
  - Fetches latest candle on each timeframe tick
  - Passes through strategy + risk manager
  - Simulates fill (with slippage/fees)
  - Logs all trades to SQLite
  - Updates portfolio snapshots
- [ ] `scripts/run_paper_trader.py` — CLI: `python scripts/run_paper_trader.py --strategy bollinger_rsi`

### Dashboard
- [ ] `src/dashboard.py` — performance monitoring
  - Terminal mode: tabulate table of current P&L, recent trades, metrics
  - Chart mode (`--chart`): equity curve, trade markers, monthly returns
- [ ] `scripts/run_dashboard.py` — CLI entry point

**Phase 3 Status:** Not started. Requires Phase 2 complete.

---

## Phase 4: Live Trading Preparation (Days 61-90)
**Goal:** Real execution layer. Deploy only if paper trading results match backtest.

### Prerequisites (must all be true before proceeding)
- [ ] Best strategy has OOS Sharpe > 1.0 across walk-forward splits
- [ ] Paper trading ran for ≥ 14 days
- [ ] Paper trading metrics within 50% of backtest (no major degradation)
- [ ] Max paper drawdown < 15%

### Live Trader
- [ ] `src/live_trader.py` — inherits PaperTrader, overrides `_execute_trade()`
  - Uses ccxt `create_market_order()` for real fills
  - Requires `--live` flag AND `CONFIRM_LIVE=yes` env var
  - Kill switch: Ctrl+C cancels all open orders
- [ ] Security audit: review all risk management code
- [ ] Test with minimum position size ($10) before scaling up
- [ ] Start with 25% of capital ($250), scale over 30 days if results hold
- [ ] Document live trading results

**Phase 4 Status:** Not started. Requires Phase 3 complete + prerequisites.

---

## Backtest Results Log
*(filled in during Phase 2)*

| Strategy | Pair | Period | Sharpe | Max DD | Win Rate | Profit Factor | Trades |
|----------|------|--------|--------|--------|----------|---------------|--------|
| TBD | | | | | | | |

## Walk-Forward Results Log
*(filled in during Phase 3)*

| Strategy | OOS Sharpe | OOS Max DD | IS vs OOS Degradation | Passes Gate? |
|----------|------------|------------|----------------------|--------------|
| TBD | | | | |

## Paper Trading Log
*(filled in during Phase 3)*

| Strategy | Start Date | Days Run | P&L | Sharpe | Max DD | Pass? |
|----------|------------|----------|-----|--------|--------|-------|
| TBD | | | | | | |
