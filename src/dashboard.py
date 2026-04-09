"""
dashboard.py — Performance monitoring for paper and live trading.

Shows current portfolio state, recent trades, and cumulative metrics.
Two output modes:
  - Terminal mode (default): prints a text table that refreshes every N seconds
  - Chart mode (--chart): plots equity curve + trade markers + monthly returns

Usage:
    from src.dashboard import Dashboard
    db = Database()
    dash = Dashboard(db, strategy_name="bollinger_rsi")
    dash.print_report()      # single snapshot
    dash.run(refresh_s=30)   # live refresh loop
    dash.plot()              # matplotlib charts
"""

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

from src.database import Database
from src.utils import get_logger

logger = get_logger(__name__)


class Dashboard:
    """
    Reads trade and snapshot data from SQLite and presents it.

    Args:
        db:            Database instance.
        strategy_name: Filter to this strategy's trades (None = all strategies).
    """

    def __init__(self, db: Database, strategy_name: Optional[str] = None):
        self.db            = db
        self.strategy_name = strategy_name

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self, refresh_s: int = 30) -> None:
        """
        Live refresh loop. Prints a new report every `refresh_s` seconds.
        Press Ctrl+C to stop.
        """
        logger.info(f"Dashboard started (refresh every {refresh_s}s). Ctrl+C to quit.")
        try:
            while True:
                self.print_report()
                time.sleep(refresh_s)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")

    def print_report(self) -> None:
        """Print a full performance report to the terminal."""
        trades    = self._load_trades()
        snapshots = self._load_snapshots()

        print("\n" + "=" * 64)
        title = f"PAPER TRADING DASHBOARD"
        if self.strategy_name:
            title += f" — {self.strategy_name}"
        print(f"  {title}")
        print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 64)

        if snapshots.empty:
            print("  No data yet. Start the paper trader to populate this dashboard.")
            print("=" * 64 + "\n")
            return

        self._print_portfolio_summary(snapshots)
        self._print_trade_metrics(trades)
        self._print_recent_trades(trades)

        print("=" * 64 + "\n")

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve, drawdown, and monthly returns.

        Args:
            save_path: If provided, saves the chart here. Otherwise displays it.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot. Run: pip install matplotlib")
            return

        snapshots = self._load_snapshots()
        trades    = self._load_trades()

        if snapshots.empty:
            logger.warning("No snapshot data to plot.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

        self._plot_equity_curve(axes[0], snapshots, trades)
        self._plot_drawdown(axes[1], snapshots)
        self._plot_monthly_returns(axes[2], snapshots)

        title = "Paper Trading Performance"
        if self.strategy_name:
            title += f" — {self.strategy_name}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Chart saved to {save_path}")
            print(f"Chart saved to {save_path}")
        else:
            plt.show()

        plt.close()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_trades(self) -> pd.DataFrame:
        """Load all paper trades from the database."""
        sql = "SELECT * FROM trades WHERE is_paper = 1"
        params = []
        if self.strategy_name:
            sql += " AND strategy = ?"
            params.append(self.strategy_name)
        sql += " ORDER BY timestamp ASC"

        with self.db._get_conn() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def _load_snapshots(self) -> pd.DataFrame:
        """Load portfolio snapshots from the database."""
        sql = "SELECT * FROM portfolio_snapshots"
        params = []
        if self.strategy_name:
            sql += " WHERE strategy = ?"
            params.append(self.strategy_name)
        sql += " ORDER BY timestamp ASC"

        with self.db._get_conn() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        return df

    # -------------------------------------------------------------------------
    # Terminal report sections
    # -------------------------------------------------------------------------

    def _print_portfolio_summary(self, snapshots: pd.DataFrame) -> None:
        """Print current portfolio state."""
        latest  = snapshots.iloc[-1]
        equity  = latest["equity"]
        cash    = latest["cash"]
        pos_val = latest["positions_value"]
        dd      = latest["drawdown_pct"]

        # Compute return vs first snapshot
        starting = snapshots.iloc[0]["equity"]
        total_return = (equity / starting - 1) * 100

        # Duration
        first_dt = snapshots.index[0]
        last_dt  = snapshots.index[-1]
        duration = last_dt - first_dt
        days     = duration.total_seconds() / 86400

        print(f"\n  PORTFOLIO")
        print(f"  {'Equity':<22} ${equity:>12,.2f}")
        print(f"  {'Cash':<22} ${cash:>12,.2f}")
        print(f"  {'Positions value':<22} ${pos_val:>12,.2f}")
        print(f"  {'Total return':<22} {total_return:>+11.2f}%")
        print(f"  {'Current drawdown':<22} {dd:>11.1%}")
        print(f"  {'Running for':<22} {days:>11.1f} days")

    def _print_trade_metrics(self, trades: pd.DataFrame) -> None:
        """Print aggregate trade statistics."""
        if trades.empty:
            print("\n  METRICS  (no trades yet)")
            return

        closed_sells = trades[
            (trades["side"] == "sell") & trades["pnl"].notna()
        ]

        if closed_sells.empty:
            print("\n  METRICS  (no closed trades yet)")
            return

        n_trades   = len(closed_sells)
        winners    = closed_sells[closed_sells["pnl"] > 0]
        losers     = closed_sells[closed_sells["pnl"] < 0]
        win_rate   = len(winners) / n_trades * 100
        avg_win    = winners["pnl"].mean() if len(winners) > 0 else 0
        avg_loss   = losers["pnl"].mean()  if len(losers)  > 0 else 0
        total_pnl  = closed_sells["pnl"].sum()

        gross_win  = winners["pnl"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        print(f"\n  METRICS")
        print(f"  {'Closed trades':<22} {n_trades:>13}")
        print(f"  {'Win rate':<22} {win_rate:>12.1f}%")
        print(f"  {'Profit factor':<22} {pf:>13.3f}")
        print(f"  {'Total P&L':<22} ${total_pnl:>+12.2f}")
        print(f"  {'Avg winner':<22} ${avg_win:>+12.2f}")
        print(f"  {'Avg loser':<22} ${avg_loss:>+12.2f}")

    def _print_recent_trades(self, trades: pd.DataFrame, n: int = 10) -> None:
        """Print the most recent N trades."""
        if trades.empty:
            return

        recent = trades.tail(n * 2)  # grab extra to ensure we have enough sells

        print(f"\n  RECENT TRADES (last {n})")
        print(f"  {'Date':>10}  {'Side':>4}  {'Pair':>10}  "
              f"{'Price':>10}  {'P&L':>10}")
        print("  " + "-" * 52)

        shown = 0
        for _, row in recent.iloc[::-1].iterrows():
            if shown >= n:
                break
            pnl_str = f"${row['pnl']:>+8.2f}" if row["pnl"] is not None else "       open"
            dt_str  = row["datetime"].strftime("%Y-%m-%d") if "datetime" in row.index else "—"
            side_colored = row["side"].upper()
            print(
                f"  {dt_str:>10}  {side_colored:>4}  {row['pair']:>10}  "
                f"${row['price']:>9,.2f}  {pnl_str}"
            )
            shown += 1

    # -------------------------------------------------------------------------
    # Chart sections
    # -------------------------------------------------------------------------

    def _plot_equity_curve(self, ax, snapshots: pd.DataFrame,
                           trades: pd.DataFrame) -> None:
        """Plot equity curve with buy/sell markers."""
        ax.plot(snapshots.index, snapshots["equity"],
                color="#2196F3", linewidth=1.5, label="Equity")
        ax.axhline(y=snapshots["equity"].iloc[0], color="#9E9E9E",
                   linestyle="--", linewidth=1, alpha=0.7, label="Start")

        # Mark trades
        if not trades.empty:
            buys  = trades[trades["side"] == "buy"]
            sells = trades[(trades["side"] == "sell") & trades["pnl"].notna()]

            # Find equity at trade times
            for _, t in buys.iterrows():
                if t["datetime"] in snapshots.index:
                    val = snapshots.loc[t["datetime"], "equity"]
                    ax.scatter(t["datetime"], val, color="#4CAF50",
                               s=40, zorder=5, marker="^")

            for _, t in sells.iterrows():
                color = "#4CAF50" if (t["pnl"] or 0) > 0 else "#F44336"
                if t["datetime"] in snapshots.index:
                    val = snapshots.loc[t["datetime"], "equity"]
                    ax.scatter(t["datetime"], val, color=color,
                               s=40, zorder=5, marker="v")

        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("Equity Curve", fontsize=11)

    def _plot_drawdown(self, ax, snapshots: pd.DataFrame) -> None:
        """Plot drawdown over time."""
        equity      = snapshots["equity"]
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max * 100

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color="#F44336", alpha=0.4)
        ax.axhline(y=0, color="#9E9E9E", linewidth=0.5)
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        ax.set_title("Drawdown", fontsize=11)

    def _plot_monthly_returns(self, ax, snapshots: pd.DataFrame) -> None:
        """Plot monthly return bar chart."""
        equity = snapshots["equity"]

        # Resample to monthly
        monthly_end = equity.resample("ME").last()
        monthly_start = equity.resample("ME").first()
        monthly_returns = (monthly_end / monthly_start - 1) * 100

        if monthly_returns.empty:
            ax.text(0.5, 0.5, "Not enough data for monthly returns",
                    ha="center", va="center", transform=ax.transAxes)
            return

        colors = ["#4CAF50" if r > 0 else "#F44336"
                  for r in monthly_returns.values]
        ax.bar(range(len(monthly_returns)), monthly_returns.values,
               color=colors, alpha=0.8, width=0.7)
        ax.axhline(y=0, color="#9E9E9E", linewidth=0.5)
        ax.set_xticks(range(len(monthly_returns)))
        ax.set_xticklabels(
            [d.strftime("%b\n%Y") for d in monthly_returns.index],
            fontsize=8
        )
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("Monthly Returns", fontsize=11)
