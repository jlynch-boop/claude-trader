"""
database.py — SQLite storage for OHLCV data, trades, and portfolio snapshots.

Three tables:
  - ohlcv:               Historical and live price candles
  - trades:              Every trade executed (backtest, paper, or live)
  - portfolio_snapshots: Periodic equity curve snapshots

Usage:
    from src.database import Database
    db = Database()
    db.insert_ohlcv("binance", "BTC/USDT", "1h", rows)
    df = db.get_ohlcv("BTC/USDT", "1h", "2024-01-01", "2024-12-31")
"""

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def _get_db_path() -> str:
    """Return the path to the SQLite database file."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, "data", "trader.db")


class Database:
    """
    Thin wrapper around SQLite.

    All timestamps are stored as milliseconds since epoch (integer).
    All datetime strings are stored as ISO 8601 UTC (text).
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _get_db_path()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_tables()
        logger.debug(f"Database initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Open a new connection. Caller is responsible for closing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange    TEXT NOT NULL,
                    pair        TEXT NOT NULL,
                    timeframe   TEXT NOT NULL,
                    timestamp   INTEGER NOT NULL,
                    datetime    TEXT NOT NULL,
                    open        REAL NOT NULL,
                    high        REAL NOT NULL,
                    low         REAL NOT NULL,
                    close       REAL NOT NULL,
                    volume      REAL NOT NULL,
                    UNIQUE(exchange, pair, timeframe, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
                    ON ohlcv(exchange, pair, timeframe, timestamp);

                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy    TEXT NOT NULL,
                    pair        TEXT NOT NULL,
                    side        TEXT NOT NULL,  -- 'buy' or 'sell'
                    price       REAL NOT NULL,
                    amount      REAL NOT NULL,  -- base currency amount (e.g. BTC)
                    fee         REAL NOT NULL,  -- fee in quote currency (e.g. USDT)
                    timestamp   INTEGER NOT NULL,
                    datetime    TEXT NOT NULL,
                    pnl         REAL,           -- realized P&L (null for open trades)
                    is_paper    INTEGER NOT NULL DEFAULT 1,  -- 1=paper/backtest, 0=live
                    notes       TEXT
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy         TEXT NOT NULL,
                    timestamp        INTEGER NOT NULL,
                    datetime         TEXT NOT NULL,
                    equity           REAL NOT NULL,  -- total portfolio value
                    cash             REAL NOT NULL,  -- uninvested cash
                    positions_value  REAL NOT NULL,  -- value of open positions
                    drawdown_pct     REAL NOT NULL   -- current drawdown from peak (0.0-1.0)
                );
            """)

    # -------------------------------------------------------------------------
    # OHLCV methods
    # -------------------------------------------------------------------------

    def insert_ohlcv(self, exchange: str, pair: str, timeframe: str,
                     rows: list[list]) -> int:
        """
        Insert OHLCV rows, ignoring duplicates.

        Args:
            exchange:  Exchange name (e.g. "binance")
            pair:      Trading pair (e.g. "BTC/USDT")
            timeframe: Candle interval (e.g. "1h")
            rows:      List of [timestamp_ms, open, high, low, close, volume]
                       as returned by ccxt.fetch_ohlcv()

        Returns:
            Number of new rows inserted (duplicates are silently skipped).
        """
        if not rows:
            return 0

        records = []
        for row in rows:
            ts_ms = int(row[0])
            dt_str = datetime.fromtimestamp(
                ts_ms / 1000, tz=timezone.utc
            ).isoformat()
            records.append((
                exchange, pair, timeframe, ts_ms, dt_str,
                float(row[1]), float(row[2]), float(row[3]),
                float(row[4]), float(row[5])
            ))

        sql = """
            INSERT OR IGNORE INTO ohlcv
              (exchange, pair, timeframe, timestamp, datetime,
               open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.executemany(sql, records)
            inserted = cursor.rowcount

        logger.debug(f"Inserted {inserted}/{len(rows)} candles for {pair} {timeframe}")
        return inserted

    def get_ohlcv(self, pair: str, timeframe: str,
                  start_date: str = None, end_date: str = None,
                  exchange: str = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data as a DataFrame.

        Args:
            pair:       Trading pair (e.g. "BTC/USDT")
            timeframe:  Candle interval (e.g. "1h")
            start_date: Optional start date string "YYYY-MM-DD" (inclusive)
            end_date:   Optional end date string "YYYY-MM-DD" (inclusive)
            exchange:   Exchange name filter (default None = any exchange).
                        Pass cfg["exchange"]["name"] to use a specific source.

        Returns:
            DataFrame with columns [open, high, low, close, volume],
            indexed by UTC datetime. Empty DataFrame if no data found.
        """
        if exchange is not None:
            sql = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE exchange = ? AND pair = ? AND timeframe = ?
            """
            params = [exchange, pair, timeframe]
        else:
            sql = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE pair = ? AND timeframe = ?
            """
            params = [pair, timeframe]

        if start_date:
            start_ts = _date_str_to_ts(start_date)
            sql += " AND timestamp >= ?"
            params.append(start_ts)

        if end_date:
            # Add one day to make end_date inclusive
            end_ts = _date_str_to_ts(end_date) + 86_400_000
            sql += " AND timestamp < ?"
            params.append(end_ts)

        sql += " ORDER BY timestamp ASC"

        with self._get_conn() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        return df

    def get_latest_timestamp(self, exchange: str, pair: str,
                             timeframe: str) -> int | None:
        """
        Get the most recent candle timestamp for a given pair/timeframe.
        Used by the data fetcher to resume from where it left off.

        Returns:
            Latest timestamp in milliseconds, or None if no data exists.
        """
        sql = """
            SELECT MAX(timestamp) as latest
            FROM ohlcv
            WHERE exchange = ? AND pair = ? AND timeframe = ?
        """
        with self._get_conn() as conn:
            row = conn.execute(sql, (exchange, pair, timeframe)).fetchone()

        return row["latest"] if row and row["latest"] is not None else None

    def export_ohlcv_csv(self, pair: str, timeframe: str,
                         output_path: str,
                         exchange: str = None) -> int:
        """
        Export OHLCV data to a CSV file for offline use and git storage.

        Args:
            pair:        Trading pair (e.g. "BTC/USDT")
            timeframe:   Candle interval (e.g. "1h")
            output_path: Full path for the output CSV file
            exchange:    Exchange name filter (default None = any exchange)

        Returns:
            Number of rows exported.
        """
        df = self.get_ohlcv(pair, timeframe, exchange=exchange)
        if df.empty:
            logger.warning(f"No data to export for {pair} {timeframe}")
            return 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        logger.info(f"Exported {len(df)} rows to {output_path}")
        return len(df)

    def import_ohlcv_csv(self, csv_path: str, exchange: str, pair: str,
                         timeframe: str) -> int:
        """
        Import OHLCV data from a CSV file into the database.

        Used to seed the database from committed CSV files without
        needing to re-fetch from the exchange.

        Args:
            csv_path:  Path to the CSV file (exported by export_ohlcv_csv)
            exchange:  Exchange name (e.g. "binance")
            pair:      Trading pair (e.g. "BTC/USDT")
            timeframe: Candle interval (e.g. "1h")

        Returns:
            Number of new rows inserted.
        """
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return 0

        # Convert datetime index back to millisecond timestamps
        rows = []
        for dt, row in df.iterrows():
            ts_ms = int(pd.Timestamp(dt).timestamp() * 1000)
            rows.append([ts_ms, row["open"], row["high"],
                         row["low"], row["close"], row["volume"]])

        inserted = self.insert_ohlcv(exchange, pair, timeframe, rows)
        logger.info(f"Imported {inserted} rows from {csv_path}")
        return inserted

    # -------------------------------------------------------------------------
    # Trade methods
    # -------------------------------------------------------------------------

    def insert_trade(self, strategy: str, pair: str, side: str, price: float,
                     amount: float, fee: float, timestamp_ms: int,
                     pnl: float = None, is_paper: bool = True,
                     notes: str = None) -> int:
        """
        Record a single trade.

        Args:
            strategy:     Strategy name (e.g. "bollinger_rsi")
            pair:         Trading pair
            side:         "buy" or "sell"
            price:        Execution price
            amount:       Amount in base currency (e.g. BTC)
            fee:          Fee in quote currency (e.g. USDT)
            timestamp_ms: Execution time in milliseconds
            pnl:          Realized P&L (None for opening trades)
            is_paper:     True for backtest/paper, False for live
            notes:        Optional human-readable notes

        Returns:
            Row ID of the inserted trade.
        """
        dt_str = datetime.fromtimestamp(
            timestamp_ms / 1000, tz=timezone.utc
        ).isoformat()

        sql = """
            INSERT INTO trades
              (strategy, pair, side, price, amount, fee, timestamp, datetime,
               pnl, is_paper, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.execute(sql, (
                strategy, pair, side, price, amount, fee,
                timestamp_ms, dt_str, pnl, int(is_paper), notes
            ))
            return cursor.lastrowid

    def get_trades(self, strategy: str = None,
                   is_paper: bool = None) -> pd.DataFrame:
        """
        Retrieve trades as a DataFrame.

        Args:
            strategy: Filter by strategy name (optional).
            is_paper: Filter by paper/live (optional).

        Returns:
            DataFrame with all trade fields, indexed by datetime.
        """
        sql = "SELECT * FROM trades WHERE 1=1"
        params = []

        if strategy:
            sql += " AND strategy = ?"
            params.append(strategy)

        if is_paper is not None:
            sql += " AND is_paper = ?"
            params.append(int(is_paper))

        sql += " ORDER BY timestamp ASC"

        with self._get_conn() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        return df

    # -------------------------------------------------------------------------
    # Portfolio snapshot methods
    # -------------------------------------------------------------------------

    def insert_snapshot(self, strategy: str, timestamp_ms: int, equity: float,
                        cash: float, positions_value: float,
                        drawdown_pct: float) -> None:
        """Record a portfolio equity snapshot."""
        dt_str = datetime.fromtimestamp(
            timestamp_ms / 1000, tz=timezone.utc
        ).isoformat()

        sql = """
            INSERT INTO portfolio_snapshots
              (strategy, timestamp, datetime, equity, cash,
               positions_value, drawdown_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(sql, (
                strategy, timestamp_ms, dt_str,
                equity, cash, positions_value, drawdown_pct
            ))

    def get_snapshots(self, strategy: str) -> pd.DataFrame:
        """Retrieve portfolio snapshots as a DataFrame indexed by datetime."""
        sql = """
            SELECT * FROM portfolio_snapshots
            WHERE strategy = ?
            ORDER BY timestamp ASC
        """
        with self._get_conn() as conn:
            df = pd.read_sql_query(sql, conn, params=[strategy])

        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("datetime")

        return df


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _date_str_to_ts(date_str: str) -> int:
    """Convert "YYYY-MM-DD" to millisecond timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
