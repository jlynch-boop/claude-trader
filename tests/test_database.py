"""
tests/test_database.py — Tests for src/database.py

Run with: pytest tests/test_database.py -v
"""

import os
import tempfile
import pytest

from src.database import Database


@pytest.fixture
def db():
    """Create a temporary in-memory database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db = Database(db_path=path)
    yield db
    os.unlink(path)


# ---------------------------------------------------------------------------
# OHLCV tests
# ---------------------------------------------------------------------------

SAMPLE_CANDLES = [
    [1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.5],
    [1704070800000, 42300.0, 42800.0, 42100.0, 42600.0, 95.2],
    [1704074400000, 42600.0, 43000.0, 42400.0, 42900.0, 110.1],
]


def test_insert_ohlcv_returns_count(db):
    inserted = db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    assert inserted == 3


def test_insert_ohlcv_deduplicates(db):
    db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    # Insert the same candles again — should be ignored
    inserted = db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    assert inserted == 0


def test_insert_ohlcv_partial_dedup(db):
    db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES[:2])
    # Insert 2 old + 1 new
    inserted = db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    assert inserted == 1


def test_get_ohlcv_returns_dataframe(db):
    db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    df = db.get_ohlcv("BTC/USDT", "1h")
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_get_ohlcv_date_filter(db):
    db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    # All candles are on 2024-01-01, so filter to that date
    df = db.get_ohlcv("BTC/USDT", "1h",
                      start_date="2024-01-01", end_date="2024-01-01")
    assert len(df) == 3


def test_get_ohlcv_empty_when_no_data(db):
    df = db.get_ohlcv("BTC/USDT", "1h")
    assert df.empty


def test_get_ohlcv_values_are_correct(db):
    db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    df = db.get_ohlcv("BTC/USDT", "1h")
    assert df.iloc[0]["open"] == 42000.0
    assert df.iloc[0]["close"] == 42300.0
    assert df.iloc[2]["high"] == 43000.0


def test_get_latest_timestamp_none_when_empty(db):
    result = db.get_latest_timestamp("binance", "BTC/USDT", "1h")
    assert result is None


def test_get_latest_timestamp_returns_max(db):
    db.insert_ohlcv("binance", "BTC/USDT", "1h", SAMPLE_CANDLES)
    latest = db.get_latest_timestamp("binance", "BTC/USDT", "1h")
    assert latest == 1704074400000  # Last candle's timestamp


def test_insert_empty_ohlcv_returns_zero(db):
    inserted = db.insert_ohlcv("binance", "BTC/USDT", "1h", [])
    assert inserted == 0


# ---------------------------------------------------------------------------
# Trade tests
# ---------------------------------------------------------------------------

def test_insert_trade_returns_id(db):
    row_id = db.insert_trade(
        strategy="bollinger_rsi",
        pair="BTC/USDT",
        side="buy",
        price=42000.0,
        amount=0.01,
        fee=0.42,
        timestamp_ms=1704067200000,
        pnl=None,
        is_paper=True,
    )
    assert row_id == 1


def test_get_trades_returns_dataframe(db):
    db.insert_trade("bollinger_rsi", "BTC/USDT", "buy",
                    42000.0, 0.01, 0.42, 1704067200000, None, True)
    db.insert_trade("bollinger_rsi", "BTC/USDT", "sell",
                    43000.0, 0.01, 0.43, 1704074400000, 57.15, True)
    trades = db.get_trades(strategy="bollinger_rsi")
    assert len(trades) == 2
    assert trades.iloc[0]["side"] == "buy"
    assert trades.iloc[1]["pnl"] == pytest.approx(57.15)


def test_get_trades_filter_by_strategy(db):
    db.insert_trade("bollinger_rsi", "BTC/USDT", "buy",
                    42000.0, 0.01, 0.42, 1704067200000)
    db.insert_trade("momentum", "ETH/USDT", "buy",
                    2200.0, 0.1, 0.22, 1704070800000)
    trades = db.get_trades(strategy="momentum")
    assert len(trades) == 1
    assert trades.iloc[0]["pair"] == "ETH/USDT"


# ---------------------------------------------------------------------------
# Portfolio snapshot tests
# ---------------------------------------------------------------------------

def test_insert_and_get_snapshot(db):
    db.insert_snapshot(
        strategy="bollinger_rsi",
        timestamp_ms=1704067200000,
        equity=1000.0,
        cash=800.0,
        positions_value=200.0,
        drawdown_pct=0.0,
    )
    df = db.get_snapshots("bollinger_rsi")
    assert len(df) == 1
    assert df.iloc[0]["equity"] == 1000.0
    assert df.iloc[0]["drawdown_pct"] == 0.0


def test_get_snapshots_empty(db):
    df = db.get_snapshots("nonexistent_strategy")
    assert df.empty
