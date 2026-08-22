"""Tests for profitability/realism upgrades: frictions, exits, correlation,
adaptive risk, and the event-driven backtest engine.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import pytest

from src.paper.account import default_account, get_position
from src.paper.broker_sim import apply_slippage, execute_entry, execute_exit
from src.paper.correlation import correlation_scale, factor_group
from src.paper.exits import open_profit_r, plan_exit, ratchet_stop, runner_target
from src.paper.journal import adaptive_risk_pct, rolling_expectancy


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


# --------------------------------------------------------------------------- #
# Frictions
# --------------------------------------------------------------------------- #
def test_apply_slippage_direction():
    # LONG entry pays up, LONG exit receives less
    assert apply_slippage(100.0, direction="LONG", is_entry=True, slippage_bps=10) > 100.0
    assert apply_slippage(100.0, direction="LONG", is_entry=False, slippage_bps=10) < 100.0
    # SHORT entry sells lower, SHORT cover pays up
    assert apply_slippage(100.0, direction="SHORT", is_entry=True, slippage_bps=10) < 100.0
    assert apply_slippage(100.0, direction="SHORT", is_entry=False, slippage_bps=10) > 100.0
    assert apply_slippage(100.0, direction="LONG", is_entry=True, slippage_bps=0) == 100.0


def test_frictions_reduce_round_trip_pnl(data_root):
    acct = default_account()
    acct["params"]["slippage_bps"] = 10.0
    acct["params"]["commission_per_trade"] = 1.0
    execute_entry(
        acct, symbol="NVDA", direction="LONG", price=100.0, shares=10,
        stop=95.0, target=110.0, trading_date="2026-07-08",
        signal={"source": "swing", "horizon": "Swing"}, book="swing",
    )
    # Flat round trip at same quote must LOSE money to frictions.
    exit_trade = execute_exit(acct, price=100.0, trading_date="2026-07-08", book="swing")
    assert exit_trade["pnl"] < 0
    assert acct["realized_pnl"] < 0


# --------------------------------------------------------------------------- #
# Partial exits / scale-outs
# --------------------------------------------------------------------------- #
def test_partial_scale_out_keeps_runner(data_root):
    acct = default_account()
    acct["params"]["slippage_bps"] = 0.0
    execute_entry(
        acct, symbol="NVDA", direction="LONG", price=100.0, shares=10,
        stop=95.0, target=110.0, trading_date="2026-07-08",
        signal={"source": "swing", "horizon": "Swing"}, book="swing",
    )
    t = execute_exit(acct, price=110.0, trading_date="2026-07-08", book="swing", shares=5)
    assert t["action"] == "SCALE_OUT"
    assert t["remaining_shares"] == 5
    pos = get_position(acct, "swing")
    assert pos is not None and pos["shares"] == 5
    assert pos["scaled_out"] is True
    assert t["pnl"] == pytest.approx((110.0 - 100.0) * 5)


# --------------------------------------------------------------------------- #
# Exit management logic (pure)
# --------------------------------------------------------------------------- #
def test_ratchet_moves_to_breakeven_then_trails():
    pos = {
        "direction": "LONG", "avg_entry": 100.0, "initial_stop": 90.0,
        "risk_per_share": 10.0, "stop": 90.0, "high_water": 100.0, "low_water": 100.0,
    }
    params = {"breakeven_trigger_r": 1.0, "breakeven_buffer_r": 0.0,
              "trail_trigger_r": 1.5, "trail_distance_r": 1.0}
    # +1R → breakeven
    pos["high_water"] = 110.0
    new_stop, note = ratchet_stop(pos, params)
    assert new_stop == pytest.approx(100.0)
    assert note is not None
    # +2R → trail to high_water - 1R = 120 - 10 = 110
    pos["stop"] = 100.0
    pos["high_water"] = 120.0
    new_stop, note = ratchet_stop(pos, params)
    assert new_stop == pytest.approx(110.0)


def test_ratchet_never_loosens():
    pos = {
        "direction": "LONG", "avg_entry": 100.0, "initial_stop": 90.0,
        "risk_per_share": 10.0, "stop": 110.0, "high_water": 112.0, "low_water": 100.0,
    }
    params = {"breakeven_trigger_r": 1.0, "trail_trigger_r": 1.5, "trail_distance_r": 5.0}
    new_stop, _ = ratchet_stop(pos, params)
    assert new_stop >= 110.0  # would-be trail (112-50) never lowers the stop


def test_plan_exit_scales_at_target():
    pos = {
        "direction": "LONG", "avg_entry": 100.0, "initial_stop": 95.0,
        "risk_per_share": 5.0, "stop": 95.0, "target1": 105.0, "target2": 115.0,
        "shares": 10, "scaled_out": False, "high_water": 100.0, "low_water": 100.0,
    }
    params = {"exit_management": True, "scale_out_enabled": True, "scale_out_pct": 0.5,
              "breakeven_buffer_r": 0.05}
    plan = plan_exit(pos, 105.0, params)
    assert plan["action"] == "scale_out"
    assert plan["shares"] == 5
    assert plan["new_stop"] >= 100.0  # runner protected at/above breakeven


def test_open_profit_r_and_runner_target():
    pos = {"direction": "LONG", "avg_entry": 100.0, "risk_per_share": 5.0,
           "target2": None, "stop": 95.0}
    assert open_profit_r(pos, 110.0) == pytest.approx(2.0)
    rt = runner_target(pos, {"runner_target_r": 3.0})
    assert rt == pytest.approx(115.0)


# --------------------------------------------------------------------------- #
# Correlation guard
# --------------------------------------------------------------------------- #
def test_correlation_scale_same_group_same_dir():
    scale, note = correlation_scale(
        new_symbol="AMD", new_direction="LONG",
        open_symbol="NVDA", open_direction="LONG",
    )
    assert scale == 0.5 and note
    # same symbol same dir → blocked
    scale2, _ = correlation_scale(
        new_symbol="NVDA", new_direction="LONG",
        open_symbol="NVDA", open_direction="LONG",
    )
    assert scale2 == 0.0
    # opposite direction = hedge → no scaling
    scale3, _ = correlation_scale(
        new_symbol="AMD", new_direction="SHORT",
        open_symbol="NVDA", open_direction="LONG",
    )
    assert scale3 == 1.0
    assert factor_group("NVDA") == "semis_ai"


def test_allocation_applies_correlation_guard(data_root):
    from src.paper.allocation import allocate_for_entry

    acct = default_account()
    # Simulate an open swing NVDA long.
    execute_entry(
        acct, symbol="NVDA", direction="LONG", price=100.0, shares=10,
        stop=95.0, target=115.0, trading_date="2026-07-08",
        signal={"source": "swing", "horizon": "Swing"}, book="swing",
    )
    alloc = allocate_for_entry(
        acct, book="intraday",
        signal={"symbol": "AMD", "direction": "LONG", "entry_price": 100.0,
                "stop_price": 95.0, "target_price": 110.0, "win_prob": 65,
                "expected_return_pct": 5.0, "horizon": "Intraday",
                "entry_status": {"status": "READY"}},
        quote=100.0, entry_status={"status": "READY"},
    )
    assert alloc["corr_scale"] == 0.5
    assert alloc["corr_note"]


# --------------------------------------------------------------------------- #
# Adaptive risk from expectancy
# --------------------------------------------------------------------------- #
def test_rolling_expectancy_and_adaptive_risk():
    pnls = [50, -20, 40, -10, 60, -15, 30, -5, 45, -12]
    stats = rolling_expectancy(pnls)
    assert stats["n"] == 10
    assert stats["profit_factor"] > 1.0
    assert stats["expectancy"] > 0
    new_risk, note = adaptive_risk_pct(1.5, stats, cap=3.0, base=1.5)
    assert new_risk >= 1.5 and note  # positive edge scales up toward monthly ambition

    losing = [-30] * 10
    lose_stats = rolling_expectancy(losing)
    new_risk2, note2 = adaptive_risk_pct(1.5, lose_stats, cap=3.0, base=1.5)
    assert new_risk2 < 1.5 and note2  # broken edge scales down

    # Cold start (too few trades) stays near base without whipsaw.
    thin = rolling_expectancy([100, -50])
    nr, _ = adaptive_risk_pct(1.5, thin, cap=3.0, base=1.5)
    assert 0.35 <= nr <= 3.0

    # With 5+ losing trades, risk collapses toward the floor (smoothed step).
    mid = rolling_expectancy([-40] * 5)
    nr2, note3 = adaptive_risk_pct(1.5, mid, cap=3.0, base=1.5)
    assert nr2 < 1.5 and note3
    assert nr2 <= 1.0  # half-step toward 0.35 from 1.5

    # Proven edge can push risk toward the 3% cap (10%/mo ambition math).
    strong = rolling_expectancy([80, -20, 90, -15, 70, -25, 100, -10, 60, -18])
    nr3, note4 = adaptive_risk_pct(1.5, strong, cap=3.0, base=1.5)
    assert nr3 > 1.5 and note4
    assert nr3 <= 3.0


def test_plan_eod_soft_keeps_winners():
    from src.paper.exits import plan_eod_exit

    pos = {
        "direction": "LONG", "avg_entry": 100.0, "initial_stop": 95.0,
        "risk_per_share": 5.0, "stop": 95.0, "shares": 10,
        "scaled_out": False, "high_water": 108.0, "low_water": 99.0,
    }
    params = {
        "eod_exit_mode": "soft",
        "force_exit_intraday_at_close": True,
        "eod_force_close_if_pnl_r_below": 0.25,
        "eod_allow_runner_overnight": True,
        "eod_scale_out_winners": True,
        "scale_out_enabled": True,
        "scale_out_pct": 0.5,
        "breakeven_buffer_r": 0.05,
        "breakeven_trigger_r": 1.0,
        "trail_trigger_r": 1.5,
        "trail_distance_r": 1.0,
    }
    # +1.6R winner → scale-out + overnight runner (not full cut)
    plan = plan_eod_exit(pos, 108.0, params)
    assert plan["action"] == "scale_out"
    assert plan["promote_overnight"] is True

    # Flat / loser → force close
    plan2 = plan_eod_exit(pos, 100.5, params)  # +0.1R < 0.25
    assert plan2["action"] == "exit"

    # Already scaled runner in profit → hold overnight
    pos["scaled_out"] = True
    pos["shares"] = 5
    plan3 = plan_eod_exit(pos, 108.0, params)
    assert plan3["action"] == "hold"
    assert plan3["promote_overnight"] is True


def test_expected_r_gate_unit():
    from src.paper.allocation import MIN_EXPECTED_R, expected_r

    # p=0.6, R=2 → 0.6*2 - 0.4 = 0.8
    assert expected_r(win_prob=60, rr=2.0) == pytest.approx(0.8)
    # p=0.5, R=1.5 → 0.75 - 0.5 = 0.25
    assert expected_r(win_prob=50, rr=1.5) == pytest.approx(0.25)
    assert expected_r(win_prob=50, rr=1.5) >= MIN_EXPECTED_R
    # Weak: p=0.5, R=1.2 → 0.6 - 0.5 = 0.1 < floor
    assert expected_r(win_prob=50, rr=1.2) < MIN_EXPECTED_R


def test_load_account_migrates_exit_params(data_root):
    from src.paper.account import account_path, load_account, save_account

    acct = load_account()
    # Simulate a legacy account missing new keys / old risk.
    acct["params"] = {
        "risk_pct": 1.0,
        "force_exit_intraday_at_close": True,
        "cash_reserve_min_pct": 20.0,
        "cash_reserve_max_pct": 40.0,
        "params_schema_version": 0,
    }
    save_account(acct)
    loaded = load_account()
    p = loaded["params"]
    assert p["exit_management"] is True
    assert p["scale_out_enabled"] is True
    assert p["eod_exit_mode"] == "soft"
    assert p["risk_pct"] == 1.5
    assert p["cash_reserve_min_pct"] == 10.0
    assert p["adaptive_risk_cap"] == 3.0
    assert p["params_schema_version"] >= 3
    assert p["max_daily_loss_pct"] == 2.0
    assert p["max_loss_per_trade_r"] == 1.0
    assert p["breakeven_trigger_r"] == 0.5
    assert account_path().exists()


# --------------------------------------------------------------------------- #
# Event-driven backtest
# --------------------------------------------------------------------------- #
def _rising_bars(symbol: str, n: int = 60, base: float = 100.0, slope: float = 1.0):
    from datetime import date, timedelta

    from src.paper.backtest import Bar

    bars = []
    prev_close = base
    d = date(2026, 1, 2)
    for i in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        open_ = prev_close + 0.2
        close = base + slope * (i + 1)
        high = max(open_, close) + 0.6
        low = min(open_, close) - 0.6
        bars.append(Bar(
            date=d.isoformat(),
            open=round(open_, 2), high=round(high, 2),
            low=round(low, 2), close=round(close, 2), volume=1e6,
        ))
        prev_close = close
        d += timedelta(days=1)
    return bars


def test_backtest_runs_on_synthetic_uptrend(data_root):
    from src.paper.backtest import run_paper_backtest

    bars = {
        "NVDA": _rising_bars("NVDA", slope=1.2),
        "AMD": _rising_bars("AMD", slope=0.8, base=80.0),
        "QQQ": _rising_bars("QQQ", slope=0.5, base=400.0),
    }
    result = run_paper_backtest(
        symbols=["NVDA", "AMD"], bars=bars, benchmark_symbol="QQQ",
        walk_forward_folds=2,
    )
    assert result.get("error") is None
    assert result["n_days"] > 20
    assert result["n_entries"] >= 1
    assert result["n_trades"] >= 1
    # Metric contract present
    for key in ("agent_cumulative_pct", "agent_sharpe", "agent_max_drawdown_pct",
                "benchmark_cumulative_pct", "profit_factor", "expectancy",
                "excess_return_pct", "walk_forward"):
        assert key in result
    # In a clean uptrend with frictions, the engine should be net positive.
    assert result["final_equity"] > result["starting_cash"]


def test_backtest_frictions_make_it_more_conservative(data_root):
    from src.paper.backtest import run_paper_backtest

    bars = {
        "NVDA": _rising_bars("NVDA", slope=1.2),
        "AMD": _rising_bars("AMD", slope=0.8, base=80.0),
        "QQQ": _rising_bars("QQQ", slope=0.5, base=400.0),
    }
    # Force EOD flatten so path differences don't swamp the friction effect.
    common = {"eod_exit_mode": "force", "force_exit_intraday_at_close": True}
    frictionless = run_paper_backtest(
        symbols=["NVDA", "AMD"], bars=bars, benchmark_symbol="QQQ",
        params_override={**common, "slippage_bps": 0.0}, walk_forward_folds=2,
    )
    with_frictions = run_paper_backtest(
        symbols=["NVDA", "AMD"], bars=bars, benchmark_symbol="QQQ",
        params_override={**common, "slippage_bps": 20.0}, walk_forward_folds=2,
    )
    assert with_frictions["final_equity"] <= frictionless["final_equity"]
