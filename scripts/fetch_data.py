"""
scripts/fetch_data.py — CLI to download historical OHLCV data.

Usage examples:
    # Fetch 2 years of BTC/USDT hourly data
    python scripts/fetch_data.py --pair BTC/USDT --timeframe 1h --days 730

    # Fetch ETH/USDT from a specific start date
    python scripts/fetch_data.py --pair ETH/USDT --start-date 2023-01-01

    # Fetch all configured pairs
    python scripts/fetch_data.py --all-pairs --days 365
"""

import argparse
import sys
import os

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.database import Database
from src.data_fetcher import DataFetcher
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from a crypto exchange."
    )
    parser.add_argument(
        "--pair",
        type=str,
        help='Trading pair to fetch, e.g. "BTC/USDT"',
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help='Candle interval (default: 1h). Options: 1m, 5m, 15m, 1h, 4h, 1d',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to fetch (default: 365)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help='Explicit start date in YYYY-MM-DD format (overrides --days)',
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Fetch all pairs defined in config.yaml",
    )

    args = parser.parse_args()

    if not args.pair and not args.all_pairs:
        parser.error("Provide --pair or --all-pairs")

    cfg = load_config()
    db = Database()
    fetcher = DataFetcher(cfg, db)

    pairs = cfg["trading"]["pairs"] if args.all_pairs else [args.pair]

    total = 0
    for pair in pairs:
        logger.info(f"=== Fetching {pair} ===")
        n = fetcher.fetch_historical(
            pair=pair,
            timeframe=args.timeframe,
            days=args.days,
            start_date=args.start_date,
        )
        total += n
        logger.info(f"  {pair}: {n} new candles")

    logger.info(f"Total new candles inserted: {total}")


if __name__ == "__main__":
    main()
