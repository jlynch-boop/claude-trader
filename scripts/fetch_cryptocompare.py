"""
scripts/fetch_cryptocompare.py — Download historical OHLCV from CryptoCompare.

CryptoCompare has years of free hourly crypto data, unlike most exchange APIs
which cap at 720-1000 candles. No API key needed for basic usage.

Usage (run locally — requires internet):
    python scripts/fetch_cryptocompare.py --pair BTC/USDT --days 730
    python scripts/fetch_cryptocompare.py --pair ETH/USDT --days 730
    python scripts/fetch_cryptocompare.py --all-pairs --days 730

This downloads the data, stores it in the SQLite database, and exports a CSV
to data/ so you can push it to git for the cloud agent to use.

Requirements (install in your venv):
    pip install requests pandas pyyaml
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.database import Database
from src.utils import get_logger

logger = get_logger(__name__)

# CryptoCompare free API — no key needed, rate limit ~50 calls/min
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
CANDLES_PER_REQUEST = 2000  # CryptoCompare allows up to 2000 per call


def fetch_hourly_ohlcv(fsym: str, tsym: str, days: int) -> pd.DataFrame:
    """
    Download hourly OHLCV data from CryptoCompare.

    Paginates backwards in time, 2000 candles per request, until we have
    the requested number of days.

    Args:
        fsym: From symbol (e.g. "BTC")
        tsym: To symbol (e.g. "USDT" or "USD")
        days: Number of days of history

    Returns:
        DataFrame with columns [open, high, low, close, volume], indexed
        by UTC datetime, sorted oldest → newest.
    """
    total_candles_needed = days * 24
    all_candles = []
    to_ts = None  # None = latest data

    logger.info(
        f"Fetching {fsym}/{tsym} hourly data ({days} days = "
        f"{total_candles_needed:,} candles)..."
    )

    while len(all_candles) < total_candles_needed:
        remaining = total_candles_needed - len(all_candles)
        limit = min(remaining, CANDLES_PER_REQUEST)

        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": limit,
        }
        if to_ts is not None:
            params["toTs"] = to_ts

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("Response") == "Error":
                logger.error(f"API error: {data.get('Message', 'unknown')}")
                break

            candles = data.get("Data", {}).get("Data", [])
            if not candles:
                logger.warning("No more data available.")
                break

            # Filter out candles with zero volume (CryptoCompare sometimes
            # returns placeholder candles for dates before the coin existed)
            valid = [c for c in candles if c.get("volumeto", 0) > 0]
            all_candles = valid + all_candles  # prepend (older data first)

            # Next batch ends just before the oldest candle in this batch
            oldest_ts = candles[0]["time"]
            to_ts = oldest_ts - 1

            logger.info(
                f"  Got {len(valid)}/{len(candles)} valid candles, "
                f"total so far: {len(all_candles):,}"
            )

            # Rate limit: be polite
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    # Convert to DataFrame
    rows = []
    for c in all_candles:
        dt = datetime.fromtimestamp(c["time"], tz=timezone.utc)
        rows.append({
            "datetime": dt,
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c.get("volumefrom", 0),  # volume in base currency
        })

    df = pd.DataFrame(rows).set_index("datetime").sort_index()

    # Remove duplicates (overlapping pagination)
    df = df[~df.index.duplicated(keep="last")]

    logger.info(
        f"Downloaded {len(df):,} candles: "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV from CryptoCompare (free, no API key)."
    )
    parser.add_argument(
        "--pair",
        type=str,
        help='Trading pair, e.g. "BTC/USDT"',
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Fetch all pairs defined in config.yaml",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Days of history to fetch (default: 730 = 2 years)",
    )

    args = parser.parse_args()
    if not args.pair and not args.all_pairs:
        parser.error("Provide --pair or --all-pairs")

    cfg = load_config()
    db = Database()
    exchange_name = cfg["exchange"]["name"]

    pairs = cfg["trading"]["pairs"] if args.all_pairs else [args.pair]

    for pair in pairs:
        # Parse pair: "BTC/USDT" → fsym="BTC", tsym="USDT"
        parts = pair.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid pair format: {pair}. Use 'BTC/USDT'.")
            continue

        fsym, tsym = parts

        # CryptoCompare uses "USD" and "USDT" — both work
        df = fetch_hourly_ohlcv(fsym, tsym, args.days)

        if df.empty:
            logger.error(f"No data returned for {pair}.")
            continue

        # Convert to ccxt-style rows for database insertion
        rows = []
        for dt, row in df.iterrows():
            ts_ms = int(dt.timestamp() * 1000)
            rows.append([ts_ms, row["open"], row["high"],
                         row["low"], row["close"], row["volume"]])

        inserted = db.insert_ohlcv(exchange_name, pair, "1h", rows)
        logger.info(f"  {pair}: inserted {inserted} candles into DB")

        # Export to CSV for git
        safe_name = pair.replace("/", "_")
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )
        csv_path = os.path.join(data_dir, f"{safe_name}_1h.csv")
        exported = db.export_ohlcv_csv(pair, "1h", csv_path, exchange=exchange_name)
        logger.info(f"  Exported {exported} rows to {csv_path}")

    print(f"\nDone! Push the CSVs to git:")
    print(f"  git add data/*_1h.csv")
    print(f"  git commit -m 'Add {args.days}-day historical data from CryptoCompare'")
    print(f"  git push origin claude/setup-and-overview-7vTle")


if __name__ == "__main__":
    main()
