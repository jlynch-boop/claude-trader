"""
scripts/generate_seed_data.py — Generate synthetic OHLCV seed data for backtesting.

Uses Geometric Brownian Motion (GBM) with parameters calibrated to real crypto
price behavior. This is standard practice in quant finance for developing and
testing trading systems before connecting to live data.

Parameters are calibrated to approximate real BTC/USDT and ETH/USDT behavior
from 2023-2025 (bull market with high volatility).

Usage:
    python scripts/generate_seed_data.py
    python scripts/generate_seed_data.py --start-date 2022-01-01 --seed 99
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.config import load_config
from src.database import Database
from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Calibrated parameters for each synthetic asset
# ---------------------------------------------------------------------------

ASSET_PARAMS = {
    "BTC/USDT": {
        "start_price": 16_500.0,     # BTC price at 2023-01-01 (~$16,500)
        "annual_drift": 0.60,         # ~60% annual appreciation (bull market)
        "annual_volatility": 0.75,    # ~75% annual volatility (typical for BTC)
        "mean_reversion_strength": 0.02,  # Slight pull back to trend
    },
    "ETH/USDT": {
        "start_price": 1_200.0,      # ETH price at 2023-01-01 (~$1,200)
        "annual_drift": 0.70,         # ETH often outperforms BTC in bull markets
        "annual_volatility": 0.85,    # Slightly more volatile than BTC
        "mean_reversion_strength": 0.02,
    },
}


def generate_ohlcv(
    pair: str,
    start_date: str = "2023-01-01",
    end_date: str = None,
    timeframe: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using Geometric Brownian Motion.

    GBM is the standard model in quantitative finance (Black-Scholes model).
    It produces log-normally distributed returns, which matches real asset
    price behavior better than simple random walks.

    Args:
        pair:       Asset pair (must be in ASSET_PARAMS)
        start_date: Start date string "YYYY-MM-DD"
        end_date:   End date string "YYYY-MM-DD" (default: today)
        timeframe:  Candle interval (only "1h" supported for now)
        seed:       Random seed for reproducibility

    Returns:
        DataFrame with columns [open, high, low, close, volume],
        indexed by UTC datetime.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    params = ASSET_PARAMS[pair]
    np.random.seed(seed)

    # Generate hourly timestamps
    timestamps = pd.date_range(
        start=start_date, end=end_date, freq="1h", tz="UTC"
    )
    n = len(timestamps)

    # --- Simulate close prices using GBM ---
    # In GBM: S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
    # where Z ~ N(0,1), dt = 1/8760 (one hour as fraction of year)

    dt = 1.0 / 8760  # One hour as a fraction of a year
    mu = params["annual_drift"]
    sigma = params["annual_volatility"]

    # Generate log-returns
    daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n)

    # Add occasional regime shifts (simulate bull/bear transitions)
    regime_shifts = _generate_regime_shifts(n, seed)
    daily_returns += regime_shifts

    # Add mean reversion (prices tend to not drift too far from trend)
    trend = np.cumsum(np.full(n, (mu - 0.5 * sigma**2) * dt))

    # Cumulative price path
    log_prices = np.cumsum(daily_returns)
    close_prices = params["start_price"] * np.exp(log_prices)

    # --- Generate OHLC from close prices ---
    # Realistic OHLC: each candle's O/H/L are derived from close with added noise
    intra_candle_volatility = sigma * np.sqrt(dt) * 1.5

    open_prices = np.zeros(n)
    open_prices[0] = params["start_price"]
    open_prices[1:] = close_prices[:-1]  # Open = previous close

    # High and low are generated from the range implied by intra-candle vol
    candle_range = close_prices * intra_candle_volatility * np.abs(np.random.randn(n))
    candle_range = np.maximum(candle_range, close_prices * 0.001)  # Min 0.1% range

    high_prices = np.maximum(open_prices, close_prices) + candle_range * 0.5
    low_prices = np.minimum(open_prices, close_prices) - candle_range * 0.5
    low_prices = np.maximum(low_prices, close_prices * 0.001)  # No negative prices

    # --- Generate volume ---
    # Volume is higher during volatile candles (realistic behavior)
    base_volume = _get_base_volume(pair)
    price_change_pct = np.abs(daily_returns)
    volume_multiplier = 1 + 5 * price_change_pct / (sigma * np.sqrt(dt))
    volumes = base_volume * volume_multiplier * np.abs(np.random.lognormal(0, 0.5, n))

    df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    }, index=timestamps)

    # Round prices to realistic precision
    price_precision = 2 if pair == "BTC/USDT" else 4
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].round(price_precision)
    df["volume"] = df["volume"].round(4)

    logger.info(
        f"Generated {len(df)} candles for {pair}: "
        f"${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f} "
        f"(range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f})"
    )
    return df


def _generate_regime_shifts(n: int, seed: int) -> np.ndarray:
    """
    Add occasional market regime shifts (bull/bear transitions).
    These create more realistic price patterns for backtesting.
    """
    np.random.seed(seed + 100)
    shifts = np.zeros(n)

    # ~4 regime shifts per year for a 3-year dataset
    n_shifts = max(1, n // (365 * 24 // 4))
    shift_points = np.random.choice(n, size=n_shifts, replace=False)

    for point in shift_points:
        # Create a gradual regime transition over ~2 weeks
        duration = np.random.randint(200, 500)
        intensity = np.random.choice([-0.0003, -0.0002, 0.0002, 0.0003])
        end = min(point + duration, n)
        shifts[point:end] += intensity

    return shifts


def _get_base_volume(pair: str) -> float:
    """Return a realistic base volume for each pair (in base currency)."""
    volumes = {
        "BTC/USDT": 1500.0,   # ~1500 BTC/hour on Binance
        "ETH/USDT": 15000.0,  # ~15000 ETH/hour on Binance
    }
    return volumes.get(pair, 1000.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OHLCV seed data for backtesting."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    cfg = load_config()
    db = Database()

    # Project root for CSV output
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    for pair in cfg["trading"]["pairs"]:
        logger.info(f"=== Generating {pair} ===")

        df = generate_ohlcv(
            pair=pair,
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
        )

        # Insert into SQLite database
        rows = []
        for dt, row in df.iterrows():
            ts_ms = int(pd.Timestamp(dt).timestamp() * 1000)
            rows.append([ts_ms, row["open"], row["high"],
                         row["low"], row["close"], row["volume"]])

        inserted = db.insert_ohlcv("binance", pair, "1h", rows)
        logger.info(f"  Inserted {inserted} candles into database")

        # Export to CSV (safe filename: BTC/USDT → BTC_USDT)
        safe_name = pair.replace("/", "_")
        csv_path = os.path.join(data_dir, f"{safe_name}_1h.csv")
        exported = db.export_ohlcv_csv(pair, "1h", csv_path)
        logger.info(f"  Exported {exported} rows to {csv_path}")

    logger.info("Seed data generation complete.")
    logger.info(
        "NOTE: This is synthetic data generated via GBM for development use.\n"
        "      Connect to a real exchange API for live/paper trading."
    )


if __name__ == "__main__":
    main()
