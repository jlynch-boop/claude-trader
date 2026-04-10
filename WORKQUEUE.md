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
- [x] `src/walk_forward.py` — walk-forward validator
  - Split data into N sequential windows (default 5)
  - Each window: 70% in-sample, 30% out-of-sample
  - Run backtest on each split, compare IS vs OOS metrics
  - Flag if OOS Sharpe degrades >50% vs IS (overfitting warning)
  - Return aggregate OOS metrics
- [x] `tests/test_walk_forward.py` — 20 tests, all passing
- [x] `scripts/run_walk_forward.py` — CLI entry point
- [x] Run walk-forward on all 3 strategies, record results here:
  - `bollinger_rsi`: OOS Sharpe = -0.315 (3/5 overfit warnings)
  - `momentum`: OOS Sharpe = -1.566 (3/5 overfit warnings)
  - `grid`: OOS Sharpe = -0.373 (2/5 overfit warnings)
- [x] **Gate:** All fail on synthetic GBM data (expected — random data has no edge)

### Paper Trading
- [x] `src/paper_trader.py` — live data + simulated execution
  - Fetches latest candle on each timeframe tick
  - Passes through strategy + risk manager
  - Simulates fill (with slippage/fees)
  - Logs all trades to SQLite
  - Updates portfolio snapshots
  - `--simulate` mode replays historical data (no internet needed)
- [x] `scripts/run_paper_trader.py` — CLI: `python scripts/run_paper_trader.py --strategy bollinger_rsi --simulate`

### Dashboard
- [x] `src/dashboard.py` — performance monitoring
  - Terminal mode: tabulate table of current P&L, recent trades, metrics
  - Chart mode (`--chart`): equity curve, trade markers, monthly returns
- [x] `scripts/run_dashboard.py` — CLI entry point

**Phase 3 Status: COMPLETE** ✓ (125/125 tests passing)

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

## Backtest Results Log (Real CryptoCompare Data)
*(BTC/USDT 1h, 2024-04-10 → 2026-04-10, $1,000 capital, Kraken fees 0.26%)*

### Hourly data — original strategies
| Strategy | Sharpe | Max DD | Win Rate | Profit Factor | Trades | Gate |
|----------|--------|--------|----------|---------------|--------|------|
| bollinger_rsi (no trend filter, 1.5×ATR stop) | -1.137 | 9.72% | 25.1% | 0.871 | 171 | FAIL |
| momentum (trend_ma=50) | -2.986 | 15.06% | 24.3% | 0.516 | 148 | FAIL |
| grid (range_pct=0.20) | -1.499 | 7.12% | 65.8% | 0.739 | 122 | FAIL |

### Daily resampled — corrected Sharpe (rfr=0%)
| Strategy | Sharpe | Max DD | Ann. Return | Profit Factor | Trades | Gate |
|----------|--------|--------|-------------|---------------|--------|------|
| bollinger_rsi (trend_ma=50, rsi<30) | — | — | — | — | 0 | FAIL (0 signals: trend filter blocks all) |
| momentum (fast=10, slow=30) | 0.272 | 4.99% | +1.09% | 1.260 | 12 | FAIL (too few trades) |
| **grid (range_pct=0.20)** | **0.733** | **3.63%** | **+3.29%** | **1.477** | **49** | **closest to passing** |

**Analysis (real data):**
- **Grid on daily BTC is the best performer**: +6.68% total over 2 years, 75.5% win rate, PF 1.477
- **Root cause of failures**: 2024-2026 BTC had a massive bull run to ATH ($100k+) then correction — challenging for all mechanical strategies
- **BollingerRSI design flaw discovered**: RSI<30 + price at lower BB is ALWAYS below the SMA trend filter (contradictory conditions). Disabled trend filter = -10.73% return. Strategy needs rethinking.
- **Momentum needs more trades**: fast_ma=10/slow_ma=30 on daily only crosses ~6 times per year. Need shorter MAs.
- **Metrics fix applied**: Corrected `annualized_return` to use calendar time (was using hardcoded `/8760`); Sharpe now auto-detects periods_per_year from equity curve index; risk-free rate set to 0% (correct for crypto)

**Key finding:** No strategy fully passes the gate on 2024-2026 BTC data. The market regime (strong bull run then correction) is hostile to mean reversion and ranging strategies. The test period is genuinely difficult — this is valuable information, not a code bug.

## Walk-Forward Results Log (Real Data, Daily Resampled)
*(BTC/USDT, 5 splits, 70% IS / 30% OOS, daily candles from 1h resample)*

| Strategy | OOS Sharpe | OOS Max DD | Overfit Warnings | Passes Gate? |
|----------|------------|------------|-----------------|--------------|
| bollinger_rsi | N/A | N/A | N/A | FAIL (0 trades) |
| momentum | N/A | N/A | N/A | FAIL (0 trades) |
| grid | -0.861 | 1.3% | 2/5 | FAIL (only 14 OOS trades total) |

**Analysis:** Walk-forward inconclusive for Grid due to too few OOS trades (14 total = ~3 per window). Window 1 shows OOS Sharpe +1.399 (promising). Subsequent windows have too few trades to be statistically meaningful.

## Improvements Made (Strategy Parameter Tuning Session)
1. **BollingerRSI**: Added ATR-based stop (3×ATR), take-profit (6×ATR = 2:1 R:R), trend filter (trend_ma param)
2. **Momentum**: Reduced trend_ma from 200 to 50 (200 hours = only 8 days, too slow to warm up)
3. **Grid**: Widened range_pct from 0.10 to 0.20 (10% range was blown out in volatile markets)
4. **Infrastructure**: Added `--resample 1D` to backtest/walk-forward scripts for daily testing
5. **Metrics fix**: Corrected annualization for any timeframe, set rfr=0% for crypto

## Next Session Priority: Strategy Redesign
[ ] Try shorter MA periods for Momentum on daily (fast_ma=5, slow_ma=15) → more trades
[ ] Redesign BollingerRSI: remove impossible trend filter, try faster exit (target middle BB instead of 6×ATR)
[ ] Consider fetching older historical data (2020-2023) via CryptoCompare for longer test period
[ ] Implement regime detection (ADX filter) for Grid: only trade when ADX < 25 (ranging market)
[ ] Try Grid on a longer historical period where BTC ranged more

## Paper Trading Log
*(requires gate passage first — currently no strategy fully passes)*

| Strategy | Start Date | Days Run | P&L | Sharpe | Max DD | Pass? |
|----------|------------|----------|-----|--------|--------|-------|
| TBD | | | | | | |
