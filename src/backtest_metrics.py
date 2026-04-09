"""
backtest_metrics.py — Performance metrics for backtest results.

Takes a BacktestResult and computes everything you need to evaluate
whether a strategy has a real edge:
  - Sharpe ratio (risk-adjusted return)
  - Max drawdown (worst loss from peak)
  - Win rate (% of trades that were profitable)
  - Profit factor (gross wins / gross losses)
  - Total return
  - Calmar ratio (return / max drawdown)

Gate thresholds for advancing to paper trading:
  - Sharpe > 1.0
  - Max drawdown < 20%
  - Profit factor > 1.3
  - At least 30 trades

Usage:
    from src.backtest_metrics import BacktestMetrics

    metrics = BacktestMetrics(result)
    metrics.print_report()
    metrics.plot_equity_curve()
    summary = metrics.summary()
"""

import math
import numpy as np
import pandas as pd

from src.backtest_engine import BacktestResult
from src.utils import get_logger

logger = get_logger(__name__)

# Gate thresholds — a strategy must pass all to advance to paper trading
GATE_MIN_TRADES    = 30
GATE_MIN_SHARPE    = 1.0
GATE_MAX_DRAWDOWN  = 0.20   # 20%
GATE_MIN_PF        = 1.3    # Profit factor


class BacktestMetrics:
    """Computes and presents performance metrics from a BacktestResult."""

    # Hourly data: annualize by multiplying by sqrt(8760 hours/year)
    ANNUALIZATION_FACTOR = math.sqrt(8760)

    def __init__(self, result: BacktestResult):
        self.result  = result
        self.trades  = result.trades
        self.equity  = result.equity_curve
        self._cache: dict = {}

    # -------------------------------------------------------------------------
    # Core metrics
    # -------------------------------------------------------------------------

    def total_return(self) -> float:
        """Total return as a decimal. 0.15 = 15% gain."""
        if self.equity.empty:
            return 0.0
        return (self.equity.iloc[-1] / self.result.starting_capital) - 1.0

    def annualized_return(self) -> float:
        """
        Compound annual growth rate (CAGR).
        Accounts for the actual duration of the backtest.
        """
        if self.equity.empty or len(self.equity) < 2:
            return 0.0
        n_years = len(self.equity) / 8760  # hours → years
        if n_years <= 0:
            return 0.0
        total = self.total_return()
        return (1 + total) ** (1 / n_years) - 1

    def sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Annualized Sharpe ratio.

        Sharpe = (mean_hourly_return - risk_free_hourly) / std_hourly_return
                 * sqrt(8760)

        A Sharpe > 1.0 means the strategy earns more than 1 unit of return
        per unit of risk. Industry standard: > 1.0 is acceptable, > 2.0 is good.

        Args:
            risk_free_rate: Annual risk-free rate (default 4% = US T-bill rate)
        """
        if self.equity.empty or len(self.equity) < 2:
            return 0.0

        hourly_returns = self.equity.pct_change().dropna()
        if len(hourly_returns) == 0 or hourly_returns.std() == 0:
            return 0.0

        rf_hourly = risk_free_rate / 8760
        excess = hourly_returns - rf_hourly
        return float(excess.mean() / hourly_returns.std() * self.ANNUALIZATION_FACTOR)

    def max_drawdown(self) -> float:
        """
        Maximum peak-to-trough decline in equity, as a decimal.
        0.15 = the worst drop from any peak was 15%.

        This is the most important risk metric. A large max drawdown
        means the strategy can lose a lot before recovering.
        """
        if self.equity.empty:
            return 0.0
        rolling_max = self.equity.cummax()
        drawdown = (self.equity - rolling_max) / rolling_max
        return float(abs(drawdown.min()))

    def win_rate(self) -> float:
        """Fraction of trades that were profitable (pnl > 0)."""
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.pnl > 0)
        return winners / len(self.trades)

    def profit_factor(self) -> float:
        """
        Gross profit / gross loss.

        > 1.0 means the strategy makes more than it loses overall.
        > 1.3 is our gate threshold.
        > 2.0 is excellent.

        Returns float('inf') if there are no losing trades.
        """
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss   = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def calmar_ratio(self) -> float:
        """
        Annualized return / max drawdown.

        Higher is better. Shows how much return you get per unit of drawdown risk.
        > 1.0 means you earn more than your max drawdown per year.
        """
        dd = self.max_drawdown()
        if dd == 0:
            return float("inf")
        return self.annualized_return() / dd

    def avg_trade_pnl(self) -> float:
        """Average P&L per trade in quote currency."""
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    def avg_winner(self) -> float:
        """Average P&L of winning trades."""
        winners = [t.pnl for t in self.trades if t.pnl > 0]
        return sum(winners) / len(winners) if winners else 0.0

    def avg_loser(self) -> float:
        """Average P&L of losing trades (negative number)."""
        losers = [t.pnl for t in self.trades if t.pnl < 0]
        return sum(losers) / len(losers) if losers else 0.0

    def expectancy(self) -> float:
        """
        Expected profit per trade = win_rate * avg_winner + loss_rate * avg_loser.
        A positive expectancy means the strategy has a mathematical edge.
        """
        wr = self.win_rate()
        return wr * self.avg_winner() + (1 - wr) * self.avg_loser()

    def passes_gate(self) -> bool:
        """
        Returns True if this strategy meets all thresholds for paper trading.

        Thresholds:
          - At least 30 trades (statistical significance)
          - Sharpe ratio > 1.0
          - Max drawdown < 20%
          - Profit factor > 1.3
        """
        return (
            len(self.trades) >= GATE_MIN_TRADES
            and self.sharpe_ratio() >= GATE_MIN_SHARPE
            and self.max_drawdown() <= GATE_MAX_DRAWDOWN
            and self.profit_factor() >= GATE_MIN_PF
        )

    def summary(self) -> dict:
        """Return all metrics as a flat dictionary."""
        return {
            "strategy":          self.result.strategy_name,
            "pair":              self.result.pair,
            "start":             str(self.result.start_date.date()),
            "end":               str(self.result.end_date.date()),
            "starting_capital":  self.result.starting_capital,
            "final_equity":      round(self.equity.iloc[-1], 2) if not self.equity.empty else 0,
            "total_return_pct":  round(self.total_return() * 100, 2),
            "annualized_return": round(self.annualized_return() * 100, 2),
            "sharpe_ratio":      round(self.sharpe_ratio(), 3),
            "max_drawdown_pct":  round(self.max_drawdown() * 100, 2),
            "calmar_ratio":      round(self.calmar_ratio(), 3),
            "n_trades":          len(self.trades),
            "win_rate_pct":      round(self.win_rate() * 100, 2),
            "profit_factor":     round(self.profit_factor(), 3),
            "avg_trade_pnl":     round(self.avg_trade_pnl(), 4),
            "avg_winner":        round(self.avg_winner(), 4),
            "avg_loser":         round(self.avg_loser(), 4),
            "expectancy":        round(self.expectancy(), 4),
            "passes_gate":       self.passes_gate(),
        }

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted performance report to the terminal."""
        s = self.summary()

        print("\n" + "=" * 56)
        print(f"  BACKTEST REPORT: {s['strategy']} — {s['pair']}")
        print("=" * 56)
        print(f"  Period:          {s['start']} → {s['end']}")
        print(f"  Starting capital ${s['starting_capital']:>10,.2f}")
        print(f"  Final equity     ${s['final_equity']:>10,.2f}")
        print("-" * 56)
        print(f"  Total return     {s['total_return_pct']:>+10.2f}%")
        print(f"  Ann. return      {s['annualized_return']:>+10.2f}%")
        print(f"  Sharpe ratio     {s['sharpe_ratio']:>10.3f}  {'✓' if s['sharpe_ratio'] >= GATE_MIN_SHARPE else '✗'} (need > {GATE_MIN_SHARPE})")
        print(f"  Max drawdown    {s['max_drawdown_pct']:>10.2f}%  {'✓' if s['max_drawdown_pct'] <= GATE_MAX_DRAWDOWN*100 else '✗'} (need < {GATE_MAX_DRAWDOWN*100:.0f}%)")
        print(f"  Calmar ratio     {s['calmar_ratio']:>10.3f}")
        print("-" * 56)
        print(f"  Trades           {s['n_trades']:>10}    {'✓' if s['n_trades'] >= GATE_MIN_TRADES else '✗'} (need >= {GATE_MIN_TRADES})")
        print(f"  Win rate         {s['win_rate_pct']:>10.1f}%")
        print(f"  Profit factor    {s['profit_factor']:>10.3f}  {'✓' if s['profit_factor'] >= GATE_MIN_PF else '✗'} (need > {GATE_MIN_PF})")
        print(f"  Avg trade P&L   ${s['avg_trade_pnl']:>10.4f}")
        print(f"  Avg winner      ${s['avg_winner']:>10.4f}")
        print(f"  Avg loser       ${s['avg_loser']:>10.4f}")
        print(f"  Expectancy      ${s['expectancy']:>10.4f}")
        print("=" * 56)
        gate = "PASSES — ready for paper trading" if s["passes_gate"] else "FAILS  — do not advance"
        print(f"  Gate check:      {gate}")
        print("=" * 56 + "\n")

    def plot_equity_curve(self, save_path: str = None) -> None:
        """
        Plot the equity curve with drawdown overlay.

        Args:
            save_path: If provided, saves the chart to this path.
                       Otherwise displays interactively.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot equity curve")
            return

        if self.equity.empty:
            logger.warning("No equity data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)

        # Top: equity curve
        ax1.plot(self.equity.index, self.equity.values,
                 color="#2196F3", linewidth=1.5, label="Equity")
        ax1.axhline(y=self.result.starting_capital, color="#9E9E9E",
                    linestyle="--", linewidth=1, alpha=0.7, label="Starting capital")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(
            f"{self.result.strategy_name} — {self.result.pair}\n"
            f"Return: {self.total_return():+.1%}  |  "
            f"Sharpe: {self.sharpe_ratio():.2f}  |  "
            f"Max DD: {self.max_drawdown():.1%}  |  "
            f"Trades: {len(self.trades)}"
        )
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Mark trade entries and exits on the equity curve
        for trade in self.trades:
            color = "#4CAF50" if trade.pnl > 0 else "#F44336"
            if trade.exit_time in self.equity.index:
                val = self.equity.get(trade.exit_time)
                if val is not None:
                    ax1.scatter(trade.exit_time, val,
                                color=color, s=15, alpha=0.6, zorder=5)

        # Bottom: drawdown
        rolling_max = self.equity.cummax()
        drawdown = (self.equity - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color="#F44336", alpha=0.4, label="Drawdown")
        ax2.axhline(y=-self.max_drawdown() * 100, color="#F44336",
                    linestyle="--", linewidth=1, alpha=0.8,
                    label=f"Max DD: {self.max_drawdown():.1%}")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()

        plt.close()
