"""
data_fetcher.py — Download historical OHLCV data from any CCXT exchange.

Features:
  - Handles CCXT's pagination (max ~1000 candles per request)
  - Respects exchange rate limits
  - Resumes from the last stored candle (incremental updates)
  - Saves directly to SQLite via Database

Usage:
    from src.data_fetcher import DataFetcher
    from src.config import load_config
    from src.database import Database

    cfg = load_config()
    db = Database()
    fetcher = DataFetcher(cfg, db)
    fetcher.fetch_historical("BTC/USDT", "1h", days=730)
"""

import time
from datetime import datetime, timezone

import ccxt

from src.database import Database
from src.utils import get_logger, date_str_to_ts, ts_to_datetime

logger = get_logger(__name__)


class DataFetcher:
    """
    Downloads OHLCV candles from an exchange and stores them in SQLite.

    The exchange is opened in sandbox mode by default (from config).
    No API keys are needed for fetching public market data.
    """

    def __init__(self, config: dict, database: Database):
        self.config = config
        self.db = database
        self.exchange_name = config["exchange"]["name"]
        self.sandbox = config["exchange"].get("sandbox", True)
        self._exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        """
        Create and configure the CCXT exchange instance.

        Note: Sandbox mode is NOT applied here. The data fetcher only reads
        public market data (OHLCV), which doesn't require authentication or
        sandbox. Sandbox mode is only applied by the live/paper trader when
        placing orders.
        """
        exchange_class = getattr(ccxt, self.exchange_name)
        exchange = exchange_class({
            "enableRateLimit": True,  # Automatically respect rate limits
        })

        logger.info(f"Initialized exchange: {self.exchange_name} (public data only)")
        return exchange

    def fetch_historical(self, pair: str, timeframe: str,
                         days: int = 365,
                         start_date: str = None) -> int:
        """
        Fetch historical OHLCV data and save to database.

        If data already exists in the database, only fetches missing candles
        (resume/incremental mode).

        Args:
            pair:       Trading pair e.g. "BTC/USDT"
            timeframe:  Candle interval e.g. "1h", "4h", "1d"
            days:       Number of days of history to fetch (default 365).
                        Used only if start_date is not provided AND no
                        existing data is in the database.
            start_date: Optional explicit start date "YYYY-MM-DD".
                        Overrides the `days` parameter.

        Returns:
            Total number of new candles inserted into the database.
        """
        if not self._exchange.has.get("fetchOHLCV"):
            raise ValueError(
                f"{self.exchange_name} does not support fetchOHLCV"
            )

        # Determine the starting timestamp
        since_ms = self._get_since_ms(pair, timeframe, days, start_date)
        now_ms = int(time.time() * 1000)

        logger.info(
            f"Fetching {pair} {timeframe} from "
            f"{ts_to_datetime(since_ms).strftime('%Y-%m-%d')} "
            f"to {ts_to_datetime(now_ms).strftime('%Y-%m-%d')}"
        )

        total_inserted = 0
        fetch_since = since_ms

        while True:
            candles = self._fetch_batch(pair, timeframe, fetch_since)

            if not candles:
                break

            # Discard any candles that are in the future (incomplete candles)
            candles = [c for c in candles if c[0] < now_ms]

            if not candles:
                break

            inserted = self.db.insert_ohlcv(
                self.exchange_name, pair, timeframe, candles
            )
            total_inserted += inserted

            last_ts = candles[-1][0]
            last_dt = ts_to_datetime(last_ts).strftime("%Y-%m-%d %H:%M")
            logger.info(
                f"  Fetched {len(candles)} candles "
                f"(+{inserted} new), last: {last_dt}"
            )

            # If we got fewer candles than the limit, we've reached the end
            limit = self._get_limit()
            if len(candles) < limit:
                break

            # Advance to the next batch (1 ms after last candle)
            fetch_since = last_ts + 1

            # Respect rate limits
            time.sleep(self._exchange.rateLimit / 1000)

        logger.info(
            f"Done. Inserted {total_inserted} new candles for {pair} {timeframe}"
        )
        return total_inserted

    def fetch_latest_candle(self, pair: str, timeframe: str) -> list | None:
        """
        Fetch the single most recent completed candle.

        Used by the paper trader to get fresh data on each tick.

        Returns:
            [timestamp_ms, open, high, low, close, volume] or None.
        """
        candles = self._fetch_batch(pair, timeframe, since=None, limit=2)
        if not candles:
            return None
        # Return the second-to-last candle (last is still forming)
        return candles[-2] if len(candles) >= 2 else candles[-1]

    def _fetch_batch(self, pair: str, timeframe: str,
                     since: int | None, limit: int = None) -> list:
        """
        Fetch a single batch of candles from the exchange.

        Args:
            pair:      Trading pair
            timeframe: Candle interval
            since:     Start timestamp in ms (None = latest candles)
            limit:     Max candles to fetch (default: exchange max)

        Returns:
            List of [timestamp, open, high, low, close, volume] lists.
        """
        if limit is None:
            limit = self._get_limit()

        try:
            candles = self._exchange.fetch_ohlcv(
                pair,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
            return candles
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching {pair}: {e}")
            return []
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {pair}: {e}")
            raise

    def _get_since_ms(self, pair: str, timeframe: str,
                      days: int, start_date: str | None) -> int:
        """
        Determine the 'since' timestamp for fetching.

        Priority:
        1. If start_date is given, use it.
        2. If data exists in DB, resume from 1ms after the last stored candle.
        3. Otherwise, calculate `days` ago from now.
        """
        if start_date:
            return date_str_to_ts(start_date)

        latest = self.db.get_latest_timestamp(
            self.exchange_name, pair, timeframe
        )
        if latest is not None:
            logger.info(
                f"Resuming from last stored candle: "
                f"{ts_to_datetime(latest).strftime('%Y-%m-%d %H:%M')}"
            )
            return latest + 1

        # Calculate `days` ago
        now_ms = int(time.time() * 1000)
        return now_ms - (days * 24 * 60 * 60 * 1000)

    def _get_limit(self) -> int:
        """
        Get the safe candle fetch limit for this exchange.

        Most exchanges return up to 500-1500 candles per request.
        We use a conservative 500 to be safe across all exchanges.
        """
        return 500
