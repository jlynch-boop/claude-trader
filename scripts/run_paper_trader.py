"""
scripts/run_paper_trader.py — Paper trading CLI.

Run a strategy on live market data with simulated fills.
All trades and snapshots are logged to SQLite for dashboard review.

Usage:
    # Simulate (replay historical data — works without internet):
    python scripts/run_paper_trader.py --strategy bollinger_rsi --simulate
    python scripts/run_paper_trader.py --strategy momentum --simulate --candles 500

    # Live (connects to exchange — requires internet + configured exchange):
    python scripts/run_paper_trader.py --strategy bollinger_rsi

Notes:
  - Press Ctrl+C to stop cleanly at any time
  - In simulate mode, runs as fast as possible through historical data
  - In live mode, waits for each real hourly candle to close
  - Trades are stored in data/trader.db — view with: python scripts/run_dashboard.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.database import Database
from src.paper_trader import PaperTrader
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


def main():
    parser = argparse.ArgumentParser(
        description="Run a strategy in paper trading mode."
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()),
        required=True,
        help="Strategy to run",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Replay historical data instead of connecting to the exchange. "
            "Works without internet. Good for testing."
        ),
    )
    parser.add_argument(
        "--candles", "-n",
        type=int,
        default=None,
        help="Stop after N candles (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to pause between candles in simulate mode (default: 0)",
    )

    args = parser.parse_args()

    cfg = load_config()
    db  = Database()

    strategy_class = STRATEGIES[args.strategy]
    strategy       = strategy_class(cfg)

    print(f"\n{'='*60}")
    print(f"  Paper Trader: {args.strategy}")
    print(f"  Mode:         {'SIMULATE (replaying historical data)' if args.simulate else 'LIVE (connecting to exchange)'}")
    print(f"  Capital:      ${cfg['trading']['starting_capital']:,.2f}")
    print(f"  Pair:         {cfg['trading']['pairs'][0]}")
    if args.simulate:
        candles_str = str(args.candles) if args.candles else "all available"
        print(f"  Candles:      {candles_str}")
    print(f"{'='*60}\n")

    if not args.simulate:
        print(
            "  NOTE: Live mode will wait for real hourly candles from the exchange.\n"
            "  Press Ctrl+C to stop at any time.\n"
            "  Ensure config.yaml has the correct exchange configured.\n"
        )

    trader = PaperTrader(
        strategy=strategy,
        config=cfg,
        db=db,
        simulate_delay_s=args.delay,
    )

    trader.run(simulate=args.simulate, max_candles=args.candles)


if __name__ == "__main__":
    main()
