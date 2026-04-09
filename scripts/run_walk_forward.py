"""
scripts/run_walk_forward.py — Walk-forward validation CLI.

Splits historical data into sequential in-sample / out-of-sample windows,
runs backtests on each, and reports whether OOS metrics confirm a real edge.

Usage:
    python scripts/run_walk_forward.py --strategy bollinger_rsi
    python scripts/run_walk_forward.py --strategy momentum --splits 5
    python scripts/run_walk_forward.py --all   # Run all three strategies
    python scripts/run_walk_forward.py --strategy grid --is-ratio 0.75

If a strategy's average OOS Sharpe > 1.0 and OOS drawdown < 20%, it passes
the gate and can advance to paper trading.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.database import Database
from src.walk_forward import WalkForwardValidator
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
            cfg: dict, db: Database, n_splits: int, is_ratio: float) -> dict:
    """Run walk-forward validation for one strategy. Returns summary dict."""
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )

    df = db.get_ohlcv(pair, cfg["trading"]["timeframe"],
                      start_date=start, end_date=end,
                      exchange=cfg["exchange"]["name"])
    if df.empty:
        raise RuntimeError(
            f"No data for {pair} on {cfg['exchange']['name']}. "
            "Run: python scripts/generate_seed_data.py  (synthetic)\n"
            "  OR: python scripts/fetch_data.py --pair BTC/USDT  (real data)"
        )

    strategy_cls = STRATEGIES[strategy_name]
    validator    = WalkForwardValidator(
        strategy_class=strategy_cls,
        config=cfg,
        n_splits=n_splits,
        is_ratio=is_ratio,
    )

    result = validator.run(df)
    result.print_report()
    return result.summary()


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation — prove OOS edge before paper trading."
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()),
        help="Strategy to validate",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all strategies and compare OOS results",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="Trading pair (default: first pair in config)",
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
        "--splits", "-n",
        type=int,
        default=5,
        help="Number of walk-forward windows (default: 5)",
    )
    parser.add_argument(
        "--is-ratio",
        type=float,
        default=0.70,
        help="Fraction of each window used for in-sample (default: 0.70)",
    )

    args = parser.parse_args()

    if not args.strategy and not args.all:
        parser.error("Provide --strategy or --all")

    cfg = load_config()
    db  = Database()

    pair     = args.pair  or cfg["trading"]["pairs"][0]
    start    = args.start or cfg["backtest"]["start_date"]
    end      = args.end   or cfg["backtest"]["end_date"]
    n_splits = args.splits
    is_ratio = args.is_ratio

    strategies_to_run = (list(STRATEGIES.keys()) if args.all
                         else [args.strategy])

    all_summaries = []
    for name in strategies_to_run:
        logger.info(f"\n{'='*64}")
        logger.info(f"Walk-forward: {name} on {pair}")
        logger.info(f"{'='*64}")
        try:
            summary = run_one(name, pair, start, end, cfg, db, n_splits, is_ratio)
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"Walk-forward failed for {name}: {e}")

    # Comparison table when running all strategies
    if args.all and len(all_summaries) > 1:
        _print_comparison(all_summaries)


def _print_comparison(summaries: list[dict]) -> None:
    """Print a side-by-side OOS comparison of all strategies."""
    print("\n" + "=" * 72)
    print("  WALK-FORWARD COMPARISON — OUT-OF-SAMPLE RESULTS")
    print("=" * 72)
    header = f"  {'Metric':<24} " + " ".join(
        f"{s['strategy'][:14]:>14}" for s in summaries
    )
    print(header)
    print("-" * 72)

    rows = [
        ("Avg OOS Sharpe",    "avg_oos_sharpe",   "{:>14.3f}"),
        ("Avg OOS Drawdown %","avg_oos_drawdown",  "{:>14.2f}%"),
        ("Avg OOS Return %",  "avg_oos_return",    "{:>14.2f}%"),
        ("Total OOS Trades",  "total_oos_trades",  "{:>14}"),
        ("Overfit Warnings",  "overfit_windows",   "{:>14}"),
        ("Passes Gate",       "passes_gate",        "{:>14}"),
    ]

    for label, key, fmt in rows:
        vals = " ".join(fmt.format(s[key]) for s in summaries)
        print(f"  {label:<24} {vals}")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
