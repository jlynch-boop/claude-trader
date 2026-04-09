"""
scripts/run_dashboard.py — Paper trading performance dashboard.

Shows equity, P&L, recent trades, and metrics from paper trading runs.

Usage:
    # Terminal snapshot (one-time report):
    python scripts/run_dashboard.py

    # Filter to a specific strategy:
    python scripts/run_dashboard.py --strategy bollinger_rsi

    # Live refresh (updates every 30 seconds):
    python scripts/run_dashboard.py --watch

    # Save chart to file:
    python scripts/run_dashboard.py --chart

    # Chart with custom output path:
    python scripts/run_dashboard.py --chart --output data/my_chart.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database
from src.dashboard import Dashboard
from src.strategy_bollinger_rsi import BollingerRsiStrategy
from src.strategy_momentum import MomentumStrategy
from src.strategy_grid import GridStrategy

STRATEGIES = {
    "bollinger_rsi": BollingerRsiStrategy,
    "momentum":      MomentumStrategy,
    "grid":          GridStrategy,
}


def main():
    parser = argparse.ArgumentParser(
        description="View paper trading performance dashboard."
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()),
        default=None,
        help="Filter to a specific strategy (default: show all)",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Live refresh mode — update the dashboard every N seconds",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Refresh interval in seconds for --watch mode (default: 30)",
    )
    parser.add_argument(
        "--chart", "-c",
        action="store_true",
        help="Show or save a matplotlib chart (equity curve, drawdown, monthly returns)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save chart image (default: display interactively)",
    )

    args = parser.parse_args()

    db = Database()

    # Map CLI slug ("bollinger_rsi") → class name ("BollingerRsiStrategy")
    # stored in the database, so filtering works correctly.
    strategy_name = STRATEGIES[args.strategy].__name__ if args.strategy else None

    dash = Dashboard(db, strategy_name=strategy_name)

    if args.chart:
        save_path = args.output
        if save_path is None and args.strategy:
            save_path = f"data/{args.strategy}_paper_chart.png"
        dash.plot(save_path=save_path)
    elif args.watch:
        dash.run(refresh_s=args.refresh)
    else:
        dash.print_report()


if __name__ == "__main__":
    main()
