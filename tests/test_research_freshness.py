"""Tests for the research-freshness gate (2026-11-12).

A morning / step-3 research payload may drive NEW entries only when it was
generated on the trading date and at/before 11:00 ET. A "morning" plan
generated at 15:52 (2026-10-21 TSLA, entered 15:53) must not open positions.
Missing/unparseable generated_at fails open. Existing positions are always
still managed — the gate blocks entries only.
"""

from __future__ import annotations

import json

import pytest

from src.paper.account import (
    default_account,
    get_position,
)
from src.paper.broker_sim import execute_entry
from src.paper.execution_agent import decide_and_act
from src.paper.signals import (
    load_candidate_signals,
    pick_intraday_signal,
    pick_swing_signal,
    research_is_fresh,
)

DATE = "2026-11-12"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _slot(symbol, *, price=100.0, direction="LONG", win_prob=70, er=2.0):
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": price,
        "entry_zone": {"low": price - 1, "mid": price, "high": price + 1},
        "stop_price": price - 2,
        "target_price": price + 4,
        "win_prob": win_prob,
        "expected_return_pct": er,
    }


def _write_morning(root, date_str, generated_at):
    p = root / "reports" / date_str / "morning.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_trades": {
            "primary": _slot("AAA"),
            "top_trades": [_slot("CCC", price=50.0, win_prob=50, er=1.0)],
        },
        "swing_trade": _slot("AAA"),
    }
    if generated_at is not None:
        payload["generated_at"] = generated_at
    p.write_text(json.dumps(payload), encoding="utf-8")


def _write_step3(root, date_str, generated_at):
    p = root / "reports" / date_str / "step3.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_trade_update": {
            "primary": _slot("BBB", price=200.0, win_prob=80, er=3.0),
        },
    }
    if generated_at is not None:
        payload["generated_at"] = generated_at
    p.write_text(json.dumps(payload), encoding="utf-8")


def _prices():
    return {"AAA": 100.0, "BBB": 200.0, "CCC": 50.0}


# ---------------------------------------------------------------------------
# research_is_fresh unit checks
# ---------------------------------------------------------------------------


def test_fresh_morning_timestamps():
    assert research_is_fresh({"generated_at": "2026-11-12T08:00:26-05:00"}, DATE)
    assert research_is_fresh({"generated_at": "2026-11-12T11:00:00-05:00"}, DATE)


def test_late_morning_timestamps():
    # The real 2026-10-21 case: generated 15:52, entered 15:53.
    assert not research_is_fresh(
        {"generated_at": "2026-10-21T15:52:21.136502-04:00"}, "2026-10-21"
    )
    assert not research_is_fresh({"generated_at": "2026-11-12T11:00:01-05:00"}, DATE)
    assert not research_is_fresh({"generated_at": "2026-11-12T19:57:36-05:00"}, DATE)


def test_wrong_date_is_stale():
    assert not research_is_fresh({"generated_at": "2026-11-11T20:00:00-05:00"}, DATE)


def test_missing_or_bad_generated_at_fails_open():
    assert research_is_fresh({}, DATE)
    assert research_is_fresh({"generated_at": None}, DATE)
    assert research_is_fresh({"generated_at": "nonsense"}, DATE)
    # Naive (no timezone) cannot be judged -> fail open.
    assert research_is_fresh({"generated_at": "2026-11-12T08:00:00"}, DATE)


# ---------------------------------------------------------------------------
# Slot suppression via load_candidate_signals + pickers
# ---------------------------------------------------------------------------


def test_fresh_morning_enters(data_root):
    _write_morning(data_root, DATE, "2026-11-12T08:00:26-05:00")
    ctx = load_candidate_signals(DATE)
    assert ctx["morning_stale"] is False
    assert ctx["morning_primary"]["symbol"] == "AAA"
    assert ctx["swing"]["symbol"] == "AAA"
    assert [t["symbol"] for t in ctx["top_trades"]] == ["CCC"]

    picked = pick_intraday_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert picked["action"] == "enter"
    assert picked["signal"]["symbol"] == "AAA"

    swing = pick_swing_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert swing["action"] == "enter"
    assert swing["signal"]["symbol"] == "AAA"


def test_late_morning_suppresses_all_slots(data_root):
    _write_morning(data_root, DATE, "2026-11-12T15:52:21-05:00")
    ctx = load_candidate_signals(DATE)
    assert ctx["morning_stale"] is True
    assert ctx["morning_primary"] is None
    assert ctx["top_trades"] == []
    assert ctx["swing"] is None

    picked = pick_intraday_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert picked["action"] == "skip"
    assert "过期" in (picked["reason"] or "")

    swing = pick_swing_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert swing["action"] == "skip"
    assert "过期" in (swing["reason"] or "")


def test_morning_from_previous_day_is_stale(data_root):
    _write_morning(data_root, DATE, "2026-11-11T20:00:00-05:00")
    ctx = load_candidate_signals(DATE)
    assert ctx["morning_stale"] is True
    assert ctx["swing"] is None


def test_missing_generated_at_fails_open(data_root):
    _write_morning(data_root, DATE, None)
    ctx = load_candidate_signals(DATE)
    assert ctx["morning_stale"] is False
    picked = pick_swing_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert picked["action"] == "enter"


def test_late_step3_suppresses_session_primary_only(data_root):
    _write_morning(data_root, DATE, "2026-11-12T08:00:26-05:00")
    _write_step3(data_root, DATE, "2026-11-12T15:53:31-05:00")
    ctx = load_candidate_signals(DATE)
    assert ctx["morning_stale"] is False
    assert ctx["step3_stale"] is True
    assert ctx["session_primary"] is None
    # The fresh morning plan still drives entries.
    picked = pick_intraday_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert picked["action"] == "enter"
    assert picked["signal"]["symbol"] == "AAA"


def test_fresh_step3_session_primary_usable(data_root):
    _write_morning(data_root, DATE, "2026-11-12T08:00:26-05:00")
    _write_step3(data_root, DATE, "2026-11-12T10:00:00-05:00")
    ctx = load_candidate_signals(DATE)
    assert ctx["step3_stale"] is False
    assert ctx["session_primary"]["symbol"] == "BBB"
    picked = pick_intraday_signal(
        DATE, current_price_by_symbol=_prices(), session_phase="open"
    )
    assert picked["action"] == "enter"
    assert picked["signal"]["symbol"] == "BBB"


# ---------------------------------------------------------------------------
# The gate blocks entries only — position management keeps running.
# ---------------------------------------------------------------------------


def _open_leg(acct, symbol, *, book, price, shares, stop, target, trading_date):
    return execute_entry(
        acct,
        symbol=symbol,
        direction="LONG",
        price=price,
        shares=shares,
        stop=stop,
        target=target,
        reason="test",
        trading_date=trading_date,
        signal={"source": "test", "horizon": "Swing" if book == "swing" else "Intraday",
                "entry_price": price, "stop_price": stop, "target_price": target,
                "entry_zone": {"low": price - 1, "mid": price, "high": price + 1}},
        book=book,
    )


def test_stale_morning_does_not_block_stop_exit(data_root, monkeypatch):
    """On a stale-research day, an open leg's stop still fires and no new
    leg opens from the suppressed plan."""
    _write_morning(data_root, DATE, "2026-11-12T15:52:21-05:00")
    acct = default_account()
    _open_leg(acct, "AAA", book="intraday", price=100.0, shares=5,
              stop=98.0, target=104.0, trading_date="2026-11-12")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote",
        lambda *a, **k: (97.0, "test"),
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.session_in_close_window", lambda d: False
    )
    decision = decide_and_act(
        acct, DATE, session_phase="open", bars_by_symbol={},
    )
    rows = [
        b for b in (decision.get("books") or [])
        if b.get("book") == "intraday" and b.get("action") == "EXIT"
    ]
    assert rows, "stop exit must still fire on a stale-research day"
    assert "止损" in (rows[0]["reason"] or "")
    assert get_position(acct, "intraday") is None
    assert not any(
        (b.get("trade") or {}).get("action") == "ENTRY"
        for b in (decision.get("books") or [])
    )
