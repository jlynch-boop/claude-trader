"""
scripts/run_backtest.py — Run a backtest for a given strategy on seed data.

Usage:
    python scripts/run_backtest.py --strategy bollinger_rsi
    python scripts/run_backtest.py --strategy momentum --pair ETH/USDT
    python scripts/run_backtest.py --strategy grid --start 2024-01-01 --end 2024-12-31
    python scripts/run_backtest.py --all   # Run all three strategies, compare
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.database import Database
from src.backtest_engine import BacktestEngine
from src.backtest_metrics import BacktestMetrics
from src.strategy_bollinger_rsi import BollingerRsiStrategy
from src.strategy_momentum import MomentumStrategy
from src.strategy_grid import GridStrategy
from src.utils import get_logger

logger = get_logger(__name__)

STRATEGIES = {
    "bollinger_rsi": BollingerRsiStrategy,
    "momentum":      MomentumStrategy,
    "grid":          GridStrategy,
}


def run_one(strategy_name: str, pair: str, start: str, end: str,
            cfg: dict, db: Database, chart: bool = False) -> dict:
    """Run a single backtest and return the summary dict."""
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )

    df = db.get_ohlcv(pair, cfg["trading"]["timeframe"],
                      start_date=start, end_date=end)
    if df.empty:
        raise RuntimeError(
            f"No data for {pair}. "
            "Run: python scripts/generate_seed_data.py"
        )

    strategy = STRATEGIES[strategy_name](cfg)
    engine   = BacktestEngine(strategy, df, cfg)
    result   = engine.run()
    metrics  = BacktestMetrics(result)

    metrics.print_report()

    if chart:
        chart_path = f"data/{strategy_name}_{pair.replace('/', '_')}_equity.png"
        metrics.plot_equity_curve(save_path=chart_path)

    return metrics.summary()


def main():
    parser = argparse.ArgumentParser(
        description="Run a backtest on historical seed data."
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()),
        help="Strategy to backtest",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all strategies and compare results",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help='Trading pair (default: first pair in config)',
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: config backtest.start_date)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: config backtest.end_date)",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Save equity curve chart to data/",
    )

    args = parser.parse_args()

    if not args.strategy and not args.all:
        parser.error("Provide --strategy or --all")

    cfg = load_config()
    db  = Database()

    pair  = args.pair  or cfg["trading"]["pairs"][0]
    start = args.start or cfg["backtest"]["start_date"]
    end   = args.end   or cfg["backtest"]["end_date"]

    strategies_to_run = (list(STRATEGIES.keys()) if args.all
                         else [args.strategy])

    all_summaries = []
    for name in strategies_to_run:
        logger.info(f"\n{'='*56}")
        logger.info(f"Running backtest: {name} on {pair}")
        logger.info(f"{'='*56}")
        try:
            summary = run_one(name, pair, start, end, cfg, db, args.chart)
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"Backtest failed for {name}: {e}")

    # If running all strategies, print a comparison table
    if args.all and len(all_summaries) > 1:
        _print_comparison(all_summaries)


def _print_comparison(summaries: list[dict]) -> None:
    """Print a side-by-side comparison of all strategy results."""
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)
    header = f"  {'Metric':<22} " + " ".join(
        f"{s['strategy'][:14]:>14}" for s in summaries
    )
    print(header)
    print("-" * 70)

    rows = [
        ("Total return %",    "total_return_pct",  "{:>14.2f}%"),
        ("Ann. return %",     "annualized_return",  "{:>14.2f}%"),
        ("Sharpe ratio",      "sharpe_ratio",       "{:>14.3f}"),
        ("Max drawdown %",    "max_drawdown_pct",   "{:>14.2f}%"),
        ("Trades",            "n_trades",           "{:>14}"),
        ("Win rate %",        "win_rate_pct",       "{:>14.1f}%"),
        ("Profit factor",     "profit_factor",      "{:>14.3f}"),
        ("Passes gate",       "passes_gate",        "{:>14}"),
    ]

    for label, key, fmt in rows:
        vals = " ".join(fmt.format(s[key]) for s in summaries)
        print(f"  {label:<22} {vals}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
