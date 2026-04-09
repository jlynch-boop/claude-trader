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
- [x] Fetch seed data: BTC/USDT 1h (2023-01-01 to present) → `data/BTC_USDT_1h.csv` (28,633 candles, GBM synthetic)
- [x] Fetch seed data: ETH/USDT 1h (2023-01-01 to present) → `data/ETH_USDT_1h.csv` (28,633 candles, GBM synthetic)
- [x] Add CSV export/import methods to `src/database.py`
- [x] `scripts/generate_seed_data.py` — reproducible GBM data generator (seed=42)
- [x] Update `.gitignore` to allow `data/*.csv`
- [x] Commit seed CSVs to repo

**Phase 1 Status: COMPLETE** ✓ (41/41 tests passing)

---

## Phase 2: Backtesting Engine (Days 16-35)
**Goal:** Implement strategy framework, risk manager, and backtesting engine. Run all 3 strategies on seed data and compare results.

### Strategy Framework
- [x] `src/strategy_base.py` — abstract base class (Signal, SignalType, Portfolio, Strategy ABC)

### Risk Manager
- [x] `src/risk_manager.py` — fixed-fractional sizing, circuit breaker, max positions
- [x] `tests/test_risk_manager.py` — 26 tests

### Backtest Engine
- [x] `src/backtest_engine.py` — chronological simulation, fills with slippage+fees, stop/TP checks
- [x] `src/backtest_metrics.py` — Sharpe, drawdown, win rate, profit factor, Calmar, gate check
- [x] `tests/test_backtest_engine.py` — 22 tests

### Strategies
- [x] `src/strategy_bollinger_rsi.py` — BB lower touch + RSI < 30 → BUY; upper + RSI > 70 → SELL
- [x] `src/strategy_momentum.py` — golden/death cross filtered by 200-period trend (numpy bool bug fixed)
- [x] `src/strategy_grid.py` — grid levels with circuit breaker on range breakout
- [x] `tests/test_strategies.py` — 22 tests

### CLI + Analysis
- [x] `scripts/run_backtest.py` — CLI: `python scripts/run_backtest.py --strategy bollinger_rsi`
- [x] Compare all 3 strategies on BTC/USDT seed data
- [x] Document results below

**Phase 2 Status: COMPLETE** ✓ (105/105 tests passing)

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
*(on synthetic GBM seed data, 2023-01-01 → 2025-12-31, BTC/USDT 1h, $1,000 capital)*

| Strategy | Sharpe | Max DD | Win Rate | Profit Factor | Trades | Gate |
|----------|--------|--------|----------|---------------|--------|------|
| bollinger_rsi | -0.332 | 9.85% | 31.5% | 1.086 | 146 | FAIL |
| momentum | -1.158 | 15.02% | 31.0% | 0.843 | 116 | FAIL |
| grid | -1.495 | 15.44% | 66.5% | 0.816 | 209 | FAIL |

**Analysis:** All three fail the gate on synthetic data. This is expected —
GBM data has no exploitable patterns (it's mathematically random). The real
test is on live exchange data. Note that all strategies kept drawdown within
the 20% limit, which confirms the risk manager is working correctly.

**Next step:** Phase 3 walk-forward validation will use the same data; the
important test is whether the walk-forward validator correctly identifies
that there is no stable edge in random data (it should fail consistently).

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
