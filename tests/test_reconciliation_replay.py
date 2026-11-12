"""Tests for the reconciliation replay (src/jobs/reconciliation_replay.py).

Covers the replay's own logic: tick-time quotes, bar truncation, leg
construction / matching, and a synthetic two-day decide_and_act replay
through the real paper rules (no network: fabricated bars + files).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.jobs.reconciliation_replay import (
    ReplayContext,
    attribute_cause,
    build_legs,
    match_legs,
    seed_account,
    simulate_day,
)
from src.paper.account import default_account
from src.utils.trading_calendar import ET


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _bars_ctx(symbol="TEST", date_str="2026-11-02", base=100.0, touch=None):
    """Fabricated 1m bars 09:30-15:59 at ``base``; ``touch`` = (hh:mm, price)
    sets one bar's high (LONG target touch) / low for the test."""
    minute = []
    for h in range(9, 16):
        for m in range(60):
            if h == 9 and m < 30:
                continue
            if h == 15 and m >= 59:
                continue
            ts = f"{date_str}T{h:02d}:{m:02d}:00-05:00"
            o = h_ = lo = c = base
            if touch and f"{h:02d}:{m:02d}" == touch[0]:
                h_ = max(base, touch[1])
                lo = min(base, touch[1])
                c = touch[1]
            minute.append([ts, o, h_, lo, c])
    daily = {date_str: {"open": base, "high": base, "low": base, "close": base}}
    if touch:
        daily[date_str]["high"] = max(base, touch[1])
        daily[date_str]["low"] = min(base, touch[1])
        daily[date_str]["close"] = touch[1]
    bars = {"symbols": {symbol: {"minute": {date_str: minute}, "daily": daily}}}
    return ReplayContext(bars, breaker_path="/tmp/replay-test-breaker.json")


def _tick(ctx, date_str, slot):
    hh, mm = (int(x) for x in slot.split(":"))
    d = datetime.fromisoformat(date_str)
    ctx.set_tick(ET.localize(datetime(d.year, d.month, d.day, hh, mm)))


# ---------------------------------------------------------------------------
# Quote + bar truncation (no lookahead)
# ---------------------------------------------------------------------------

def test_quote_uses_open_of_bar_starting_at_tick():
    ctx = _bars_ctx()
    _tick(ctx, "2026-11-02", "10:00")
    px, src = ctx.quote("TEST", "2026-11-02")
    assert px == 100.0
    assert src == "replay_bar_open"


def test_quote_falls_back_to_last_close_before_tick():
    ctx = _bars_ctx()
    _tick(ctx, "2026-11-02", "16:00")
    px, src = ctx.quote("TEST", "2026-11-02")
    assert px == 100.0
    assert src == "replay_bar_close"


def test_bars_before_excludes_tick_bar():
    ctx = _bars_ctx()
    _tick(ctx, "2026-11-02", "10:00")
    bars = ctx.bars_before("TEST", "2026-11-02")
    assert bars, "expected bars before 10:00"
    assert all(b["ts"][11:16] < "10:00" for b in bars)
    assert bars[-1]["ts"][11:16] == "09:59"


def test_quote_bar_prev_close_walks_back():
    bars = {"symbols": {"TEST": {
        "minute": {"2026-11-02": [["2026-11-02T09:30:00-05:00", 100, 100, 100, 100]]},
        "daily": {"2026-10-30": {"open": 98, "high": 99, "low": 97, "close": 98.5},
                   "2026-11-02": {"open": 100, "high": 100, "low": 100, "close": 100}},
    }}}
    ctx = ReplayContext(bars, breaker_path="/tmp/replay-test-breaker.json")
    bar = ctx.quote_bar("TEST", "2026-11-02")
    assert bar["open"] == 100
    assert bar["prev_close"] == 98.5


# ---------------------------------------------------------------------------
# Leg construction
# ---------------------------------------------------------------------------

def _trade(action, symbol, direction, shares, price, tdate, pnl=None, r=None, book="intraday"):
    return {"action": action, "symbol": symbol, "direction": direction,
            "shares": shares, "price": price, "trading_date": tdate,
            "pnl": pnl, "r_multiple": r, "book": book, "reason": ""}


def test_build_legs_scale_out_splits_one_entry():
    trades = [
        _trade("ENTRY", "TQQQ", "LONG", 45, 78.93, "2026-09-22"),
        _trade("SCALE_OUT", "TQQQ", "LONG", 27, 81.0, "2026-09-22", pnl=55.92),
        _trade("EXIT", "TQQQ", "LONG", 18, 80.41, "2026-09-23", pnl=26.57),
    ]
    legs, open_legs = build_legs(trades)
    assert len(legs) == 2
    assert legs[0]["shares"] == 27 and legs[0]["pnl"] == 55.92
    assert legs[1]["shares"] == 18 and legs[1]["pnl"] == 26.57
    assert legs[0]["entry_price"] == 78.93
    assert open_legs == []


def test_build_legs_orphan_exit_and_open_leftover():
    trades = [
        _trade("ENTRY", "AMD", "LONG", 2, 557.5, "2026-11-12", book="swing"),
        _trade("EXIT", "MSFT", "SHORT", 2, 512.21, "2026-09-25", pnl=-37.16, book="swing"),
    ]
    legs, open_legs = build_legs(trades)
    assert len(legs) == 1 and legs[0]["orphan_exit"] is True
    assert len(open_legs) == 1 and open_legs[0]["symbol"] == "AMD"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _leg(symbol, direction, entry_date, entry_price, exit_price, pnl, shares=2, r=None):
    return {"symbol": symbol, "direction": direction, "book": "swing",
            "entry_date": entry_date, "entry_price": entry_price,
            "entry_shares": shares, "exit_date": "2026-11-02",
            "exit_price": exit_price, "shares": shares, "pnl": pnl,
            "r_multiple": r, "exit_reason": "", "exit_action": "EXIT",
            "orphan_exit": False}


def test_match_legs_statuses():
    recorded = [
        _leg("TSLA", "LONG", "2026-10-21", 385.54, 413.06, 137.6, shares=5),
        _leg("MU", "LONG", "2026-10-26", 889.77, 923.13, 66.72),
        _leg("MSFT", "SHORT", "2026-09-21", 493.63, 512.21, -37.16),
    ]
    replayed = [
        _leg("TSLA", "LONG", "2026-10-21", 385.60, 410.11, 123.5, shares=5),
        _leg("NVDA", "SHORT", "2026-11-09", 176.14, 153.51, 135.78, shares=6),
    ]
    rows = match_legs(recorded, replayed, "2026-10-20", "2026-11-12")
    by_status = {}
    for row in rows:
        by_status.setdefault(row["status"], []).append(row)
    # TSLA: same entry, exit at target vs recorded tick quote -> MISMATCH
    assert len(by_status["MISMATCH"]) == 1
    assert by_status["MISMATCH"][0]["recorded"]["symbol"] == "TSLA"
    # MU: no replay leg -> MISSING_IN_REPLAY
    assert len(by_status["MISSING_IN_REPLAY"]) == 1
    # MSFT: entry before window -> OUT_OF_WINDOW
    assert len(by_status["OUT_OF_WINDOW"]) == 1
    # NVDA: replay-only -> EXTRA_IN_REPLAY
    assert len(by_status["EXTRA_IN_REPLAY"]) == 1


def test_match_legs_match_within_tolerance():
    recorded = [_leg("MU", "LONG", "2026-10-26", 889.77, 923.13, 66.72)]
    replayed = [_leg("MU", "LONG", "2026-10-26", 890.10, 922.50, 65.50)]
    rows = match_legs(recorded, replayed, "2026-10-20", "2026-11-12")
    assert rows[0]["status"] == "MATCH"
    assert rows[0]["pnl_delta"] == pytest.approx(-1.22, abs=0.01)


def test_attribute_cause_target_fill():
    row = {"status": "MISMATCH",
           "recorded": {"symbol": "MU", "direction": "LONG", "shares": 2,
                        "exit_price": 923.13, "pnl": 66.72},
           "replayed": {"symbol": "MU", "direction": "LONG", "shares": 2,
                        "exit_price": 915.12, "pnl": 50.0,
                        "exit_reason": "短线止盈触发 @ 915.12 (target=915.12)"},
           "pnl_delta": -16.72, "exit_price_delta_pct": 0.87, "r_delta": None}
    cause = attribute_cause(row, {})
    assert cause is not None and "目标价" in cause


# ---------------------------------------------------------------------------
# Synthetic end-to-end replay day (real decide_and_act, fabricated data)
# ---------------------------------------------------------------------------

def _write_morning(data_root, date_str, symbol="TEST", direction="LONG",
                   entry=100.0, stop=95.0, target=110.0, win_prob=70.0,
                   horizon="Intraday", generated_at=None):
    import os

    d = data_root / "reports" / date_str
    d.mkdir(parents=True, exist_ok=True)
    plan = {"symbol": symbol, "direction": direction, "entry_price": entry,
            "stop_price": stop, "target_price": target, "win_prob": win_prob,
            "horizon": horizon, "entry_zone": {"low": 99.0, "high": 101.0,
                                               "mid": 100.0},
            "expected_return_pct": 10.0, "risk_reward": 2.0}
    morning = {"trading_date": date_str,
               "generated_at": generated_at or f"{date_str}T08:00:00-05:00",
               "best_opportunity": plan,
               "best_trades": {"primary": plan, "top_trades": [plan]},
               "swing_trade": None,
               "trade_candidates": []}
    (d / "morning.json").write_text(json.dumps(morning), encoding="utf-8")
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{date_str}.json").write_text(json.dumps({"news": {"errors": []}}),
                                              encoding="utf-8")


def _write_morning_empty(data_root, date_str):
    """Morning file with no candidates (so day 2 cannot re-enter)."""
    d = data_root / "reports" / date_str
    d.mkdir(parents=True, exist_ok=True)
    morning = {"trading_date": date_str, "best_opportunity": None,
               "best_trades": {"primary": None, "top_trades": []},
               "swing_trade": None, "trade_candidates": []}
    (d / "morning.json").write_text(json.dumps(morning), encoding="utf-8")
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{date_str}.json").write_text(json.dumps({"news": {"errors": []}}),
                                              encoding="utf-8")


def _seed(recorded_curve):
    recorded = {"params": {
        "risk_pct": 1.5, "prefer_swing_if_no_intraday": True, "dual_books": True,
        "miss_threshold_pct": 1.0, "force_exit_intraday_at_close": True,
        "max_price_deviation_pct": 30.0, "cash_reserve_min_pct": 5.0,
        "cash_reserve_max_pct": 15.0, "smart_allocation": True,
        "max_stop_slippage_pct": 0.5, "max_trade_loss_equity_pct": 2.0,
        "min_stop_dist_pct": 0.8, "slippage_reserve_pct": 0.2,
        "stop_cooldown_enabled": True, "max_position_pct": 25.0},
        "equity_curve": recorded_curve}
    return seed_account(recorded, "2026-11-02")


def test_late_morning_file_is_invisible(data_root, tmp_path):
    """A morning.json written after the close must not produce entries."""
    curve = [{"ts": "2026-10-30T16:00:00-05:00", "equity": 9482.89,
              "cash": 9482.89, "label": "x"}]
    _write_morning(data_root, "2026-11-02",
                   generated_at="2026-11-02T19:57:36-05:00")
    acct = _seed(curve)
    ctx = _bars_ctx()
    ctx.breaker_path = tmp_path / "breaker.json"
    day = simulate_day(acct, "2026-11-02", ctx)
    assert acct["trades"] == []
    assert day["errors"] == []
    assert all(s["action"] in ("SKIP", "HOLD", "WAIT") for s in day["slots"])


def test_simulate_day_enters_and_exits_at_target(data_root, tmp_path):
    curve = [{"ts": "2026-10-30T16:00:00-05:00", "equity": 9482.89,
              "cash": 9482.89, "label": "x"}]
    _write_morning(data_root, "2026-11-02")
    # Day 2 has no candidates: the position must exit once at the target
    # and must NOT re-enter (a re-entry would be scanned against the
    # day's earlier bars, including the touch bar itself).
    _write_morning_empty(data_root, "2026-11-03")
    acct = _seed(curve)
    breaker = tmp_path / "breaker.json"
    ctx = _bars_ctx()
    ctx.breaker_path = breaker
    day1 = simulate_day(acct, "2026-11-02", ctx)
    entries = [t for t in acct["trades"] if t.get("action") == "ENTRY"]
    assert len(entries) == 1
    assert entries[0]["symbol"] == "TEST"
    assert entries[0]["price"] == pytest.approx(100.0, abs=0.01)
    assert entries[0]["shares"] > 0
    assert day1["errors"] == []

    # Day 2: price touches the 110 target in the 10:00 bar -> exit at target.
    ctx2 = _bars_ctx(date_str="2026-11-03", touch=("10:00", 110.0))
    ctx2.breaker_path = breaker
    day2 = simulate_day(acct, "2026-11-03", ctx2)
    exits = [t for t in acct["trades"] if t.get("action") == "EXIT"]
    assert len(exits) == 1
    assert exits[0]["price"] == pytest.approx(110.0, abs=0.01)
    assert exits[0]["r_multiple"] == pytest.approx(2.0, abs=0.01)
    assert day2["errors"] == []
