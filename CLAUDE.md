# Claude Trader — Session Instructions

## What This Project Is

An AI crypto trading system built for a beginner user who is supplementing their income.
Goal: prove a statistically valid edge before risking real money.
Stack: Python, ccxt, pandas, SQLite, matplotlib.

## How to Resume Work

**At the start of every session:**

1. Read `WORKQUEUE.md` — it shows what's done and what's next
2. Find the first task marked `[ ]` (not yet done)
3. Resume from there without asking the user for instructions

If the user says "continue" or "keep going" — just do it. No need to re-explain the plan.

## Branch

Always develop on: `claude/setup-and-overview-7vTle`
Always push to that branch when work is complete.

## Key Constraints

- User is a **programming beginner** — write clear, well-commented code
- Starting capital is **under $1,000** — risk management is critical
- **No real money** until walk-forward validation passes AND paper trading shows consistent results
- `config.yaml` has `sandbox: true` by default — never change this without explicit user request

## Project Structure

```
claude-trader/
├── CLAUDE.md          ← you are here
├── WORKQUEUE.md       ← persistent task tracker — read this every session
├── config.yaml        ← all settings
├── requirements.txt
├── src/               ← all source code
├── scripts/           ← CLI entry points
├── tests/             ← pytest test suite
├── data/              ← SQLite DB + CSV seed data (gitignored except CSVs)
└── notebooks/         ← Jupyter exploration notebooks
```

## Running Tests

```bash
pytest tests/ -v
```
All tests must pass before committing.

## Current Phase Summary

See WORKQUEUE.md for exact status. Phases:
- **Phase 1** (Days 1-15): Data pipeline, indicators, database — DONE
- **Phase 2** (Days 16-35): Strategy framework, backtesting engine, 3 strategies
- **Phase 3** (Days 36-60): Walk-forward validation, paper trading, dashboard
- **Phase 4** (Days 61-90): Live trading preparation

## Important Files to Know

| File | What it does |
|------|-------------|
| `src/config.py` | Loads config.yaml |
| `src/database.py` | SQLite CRUD for ohlcv, trades, snapshots |
| `src/data_fetcher.py` | Downloads OHLCV from exchange via ccxt |
| `src/indicators.py` | SMA, EMA, RSI, Bollinger, ATR, MACD |
| `src/strategy_base.py` | Abstract base class for all strategies (Phase 2) |
| `src/risk_manager.py` | Position sizing + circuit breaker (Phase 2) |
| `src/backtest_engine.py` | Core simulation loop (Phase 2) |
