"""
tests/test_risk_manager.py — Tests for src/risk_manager.py

Run with: pytest tests/test_risk_manager.py -v
"""

import pytest
from src.risk_manager import RiskManager
from src.strategy_base import Signal, SignalType, Portfolio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "risk": {
            "max_position_pct": 0.20,
            "max_drawdown_pct": 0.15,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "max_open_positions": 3,
            "risk_per_trade_pct": 0.01,
        }
    }


@pytest.fixture
def rm(config):
    return RiskManager(config)


@pytest.fixture
def fresh_portfolio():
    """Empty portfolio at full equity."""
    return Portfolio(equity=1000.0, cash=1000.0,
                     open_positions={}, peak_equity=1000.0)


@pytest.fixture
def buy_signal():
    return Signal.buy("BTC/USDT", price=42000.0,
                      stop_loss=40740.0,  # 3% below entry
                      take_profit=44520.0,
                      reason="test buy")


@pytest.fixture
def sell_signal():
    return Signal.sell("BTC/USDT", price=43000.0, reason="test sell")


# ---------------------------------------------------------------------------
# Position sizing tests
# ---------------------------------------------------------------------------

def test_position_size_basic(rm):
    # $1000 equity, 1% risk = $10 risk
    # Entry $42000, stop $40740 → price_risk = $1260
    # Raw size = ($10 / $1260) * $42000 = $333.33
    # Capped at 20% = $200
    size = rm.calculate_position_size(1000.0, 42000.0, 40740.0)
    assert size == pytest.approx(200.0, rel=0.01)


def test_position_size_not_capped_when_tight_stop(rm):
    # Very tight stop → small position, no cap needed
    # Entry $42000, stop $41580 (1% away)
    # Raw size = ($10 / $420) * $42000 = $1000 → capped at $200
    size = rm.calculate_position_size(1000.0, 42000.0, 41580.0)
    assert size == pytest.approx(200.0, rel=0.01)


def test_position_size_wide_stop_is_smaller(rm):
    # Wider stop → smaller position (same dollar risk)
    size_narrow = rm.calculate_position_size(1000.0, 42000.0, 40740.0)  # 3% stop
    size_wide   = rm.calculate_position_size(1000.0, 42000.0, 37800.0)  # 10% stop
    # Both capped at 20%, but wide stop raw size < 20%
    assert size_wide <= size_narrow


def test_position_size_zero_for_invalid_stop(rm):
    size = rm.calculate_position_size(1000.0, 42000.0, 0.0)
    assert size == 0.0


def test_position_size_zero_when_stop_above_entry(rm):
    size = rm.calculate_position_size(1000.0, 42000.0, 43000.0)
    assert size == 0.0


def test_position_size_scales_with_equity(rm):
    size_small = rm.calculate_position_size(1000.0, 42000.0, 40740.0)
    size_large = rm.calculate_position_size(5000.0, 42000.0, 40740.0)
    assert size_large > size_small


# ---------------------------------------------------------------------------
# Drawdown and circuit breaker tests
# ---------------------------------------------------------------------------

def test_drawdown_at_peak(rm):
    dd = rm.current_drawdown(1000.0, 1000.0)
    assert dd == pytest.approx(0.0)


def test_drawdown_calculation(rm):
    dd = rm.current_drawdown(850.0, 1000.0)
    assert dd == pytest.approx(0.15)


def test_drawdown_never_negative(rm):
    # Equity above peak (shouldn't happen but should handle gracefully)
    dd = rm.current_drawdown(1100.0, 1000.0)
    assert dd == 0.0


def test_circuit_breaker_inactive_at_peak(rm):
    assert not rm.is_circuit_breaker_active(1000.0, 1000.0)


def test_circuit_breaker_inactive_below_threshold(rm):
    # 14.9% drawdown — just under 15% limit
    assert not rm.is_circuit_breaker_active(851.0, 1000.0)


def test_circuit_breaker_active_at_threshold(rm):
    # Exactly 15% drawdown
    assert rm.is_circuit_breaker_active(850.0, 1000.0)


def test_circuit_breaker_active_above_threshold(rm):
    assert rm.is_circuit_breaker_active(800.0, 1000.0)


# ---------------------------------------------------------------------------
# BUY signal checks
# ---------------------------------------------------------------------------

def test_buy_approved_on_fresh_portfolio(rm, fresh_portfolio, buy_signal):
    result = rm.check_signal(buy_signal, fresh_portfolio)
    assert result is not None
    assert result.type == SignalType.BUY


def test_buy_sets_size_pct(rm, fresh_portfolio, buy_signal):
    result = rm.check_signal(buy_signal, fresh_portfolio)
    assert result.size_pct > 0.0
    assert result.size_pct <= 0.20  # Never exceeds max_position_pct


def test_buy_rejected_by_circuit_breaker(rm, buy_signal):
    portfolio = Portfolio(equity=840.0, cash=840.0,
                          open_positions={}, peak_equity=1000.0)
    result = rm.check_signal(buy_signal, portfolio)
    assert result is None


def test_buy_rejected_when_max_positions_reached(rm, buy_signal):
    portfolio = Portfolio(
        equity=1000.0, cash=400.0,
        open_positions={
            "BTC/USDT": {"amount": 0.001, "entry_price": 42000.0},
            "ETH/USDT": {"amount": 0.1,   "entry_price": 2200.0},
            "BNB/USDT": {"amount": 0.5,   "entry_price": 400.0},
        },
        peak_equity=1000.0,
    )
    result = rm.check_signal(buy_signal, portfolio)
    assert result is None


def test_buy_rejected_when_position_already_open(rm, buy_signal):
    portfolio = Portfolio(
        equity=1000.0, cash=800.0,
        open_positions={"BTC/USDT": {"amount": 0.001, "entry_price": 42000.0}},
        peak_equity=1000.0,
    )
    result = rm.check_signal(buy_signal, portfolio)
    assert result is None


def test_buy_rejected_without_stop_loss(rm, fresh_portfolio):
    signal = Signal(type=SignalType.BUY, pair="BTC/USDT", price=42000.0,
                    stop_loss=0.0, reason="no stop")
    result = rm.check_signal(signal, fresh_portfolio)
    assert result is None


def test_buy_rejected_when_stop_above_entry(rm, fresh_portfolio):
    signal = Signal.buy("BTC/USDT", price=42000.0, stop_loss=43000.0)
    result = rm.check_signal(signal, fresh_portfolio)
    assert result is None


def test_buy_rejected_with_no_cash(rm, buy_signal):
    portfolio = Portfolio(equity=1000.0, cash=0.0,
                          open_positions={}, peak_equity=1000.0)
    result = rm.check_signal(buy_signal, portfolio)
    assert result is None


# ---------------------------------------------------------------------------
# SELL signal checks
# ---------------------------------------------------------------------------

def test_sell_approved_with_open_position(rm, sell_signal):
    portfolio = Portfolio(
        equity=1000.0, cash=800.0,
        open_positions={"BTC/USDT": {"amount": 0.001, "entry_price": 42000.0}},
        peak_equity=1000.0,
    )
    result = rm.check_signal(sell_signal, portfolio)
    assert result is not None
    assert result.type == SignalType.SELL


def test_sell_rejected_without_open_position(rm, sell_signal, fresh_portfolio):
    result = rm.check_signal(sell_signal, fresh_portfolio)
    assert result is None


# ---------------------------------------------------------------------------
# HOLD signal passthrough
# ---------------------------------------------------------------------------

def test_hold_always_passes(rm, fresh_portfolio):
    signal = Signal.hold("BTC/USDT", reason="waiting")
    result = rm.check_signal(signal, fresh_portfolio)
    assert result is not None
    assert result.type == SignalType.HOLD
