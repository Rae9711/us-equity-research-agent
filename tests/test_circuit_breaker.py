"""Tests for the day/week circuit breaker (item 4, approved 2026-11-12).

Rules under test:
- Day down >= 1.0% vs prior close -> block new entries until next day.
- Week down >= 2.0% vs last close before Monday -> block until next week.
- A trip latches even if equity recovers the same day/week.
- Existing positions are never gated by the module (it only reports
  status; the wiring gates _try_entry calls only).
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.paper.circuit_breaker import (
    breaker_decision_row,
    check_circuit_breaker,
    entries_allowed,
    evaluate,
    reference_equities,
    _default_state,
)


def _acct(equity, curve, starting_cash=10000.0):
    return {
        "starting_cash": starting_cash,
        "cash": equity,
        "equity": equity,
        "equity_curve": [
            {"ts": ts, "equity": eq, "cash": eq, "label": "tick:HOLD"}
            for ts, eq in curve
        ],
        "positions": {"intraday": [], "swing": []},
    }


def _curve(pairs):
    # pairs: (date_str, equity) -> one point at 16:00 ET that day
    return [(f"{d}T16:00:00-05:00", eq) for d, eq in pairs]


def test_day_trip_at_exactly_minus_1pct(tmp_path):
    acct = _acct(9900.0, _curve([("2026-11-11", 10000.0)]))
    status, events = check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert status["blocked"] is True
    assert status["tripped_day"] is True
    assert status["tripped_week"] is False
    assert status["day_pct"] == -1.0
    assert len(events) == 1 and events[0]["type"] == "trip" and events[0]["scope"] == "day"
    assert entries_allowed(status) is False


def test_day_no_trip_at_minus_0_9pct(tmp_path):
    acct = _acct(9910.0, _curve([("2026-11-11", 10000.0)]))
    status, events = check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert status["blocked"] is False
    assert status["day_pct"] == -0.9
    assert events == []


def test_week_trip_at_exactly_minus_2pct(tmp_path):
    # Day move is only -0.51% (day_ref 9850 -> 9800), week move is -2.0%.
    acct = _acct(
        9800.0,
        _curve([("2026-11-06", 10000.0), ("2026-11-11", 9850.0)]),
    )
    status, events = check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert status["blocked"] is True
    assert status["tripped_day"] is False
    assert status["tripped_week"] is True
    assert status["week_pct"] == -2.0
    assert any(e["type"] == "trip" and e["scope"] == "week" for e in events)


def test_week_no_trip_at_minus_1_9pct(tmp_path):
    acct = _acct(
        9810.0,
        _curve([("2026-11-06", 10000.0), ("2026-11-11", 9850.0)]),
    )
    status, events = check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert status["blocked"] is False
    assert status["week_pct"] == -1.9
    assert events == []


def test_day_trip_latches_despite_recovery_and_resets_next_day(tmp_path):
    path = tmp_path / "cb.json"
    acct = _acct(9850.0, _curve([("2026-11-11", 10000.0)]))
    status, _ = check_circuit_breaker(acct, "2026-11-12", state_path=path)
    assert status["blocked"] is True

    # Equity recovers above the threshold the same day: still blocked.
    acct["equity"] = 10050.0
    status, events = check_circuit_breaker(acct, "2026-11-12", state_path=path)
    assert status["blocked"] is True
    assert status["tripped_day"] is True
    assert events == []  # no duplicate trip event

    # Next trading day: reset event, unblocked (day_ref is now 2026-11-12 close).
    acct2 = _acct(
        10050.0,
        _curve([("2026-11-11", 10000.0), ("2026-11-12", 10050.0)]),
    )
    status, events = check_circuit_breaker(acct2, "2026-11-13", state_path=path)
    assert status["blocked"] is False
    assert status["tripped_day"] is False
    assert any(e["type"] == "reset" and e["scope"] == "day" for e in events)


def test_week_trip_latches_and_resets_next_week(tmp_path):
    path = tmp_path / "cb.json"
    # Week of 2026-11-09 (Mon) to 2026-11-13 (Fri). Prior Friday close 10000.
    acct = _acct(9750.0, _curve([("2026-11-06", 10000.0), ("2026-11-11", 9900.0)]))
    status, _ = check_circuit_breaker(acct, "2026-11-12", state_path=path)
    assert status["tripped_week"] is True

    # Friday same week, equity recovered: still blocked.
    acct["equity"] = 10000.0
    status, _ = check_circuit_breaker(acct, "2026-11-13", state_path=path)
    assert status["blocked"] is True

    # Next Monday: week resets (day also resets; refs recomputed).
    acct2 = _acct(
        10000.0,
        _curve([("2026-11-06", 10000.0), ("2026-11-13", 10000.0)]),
    )
    status, events = check_circuit_breaker(acct2, "2026-11-16", state_path=path)
    assert status["blocked"] is False
    assert status["tripped_week"] is False
    assert any(e["type"] == "reset" and e["scope"] == "week" for e in events)


def test_both_day_and_week_trip_together(tmp_path):
    acct = _acct(9700.0, _curve([("2026-11-06", 10000.0)]))
    status, events = check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert status["blocked"] is True
    assert status["tripped_day"] is True
    assert status["tripped_week"] is True
    assert len([e for e in events if e["type"] == "trip"]) == 2
    assert status["reason"] is not None and "当日" in status["reason"] and "当周" in status["reason"]


def test_no_history_falls_back_to_starting_cash(tmp_path):
    acct = _acct(9900.0, [])
    day_ref, week_ref = reference_equities(acct, "2026-11-12")
    assert day_ref == 10000.0 and week_ref == 10000.0
    status, _ = check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert status["blocked"] is True  # -1.0% vs starting cash


def test_state_survives_reload_and_caps_events(tmp_path):
    path = tmp_path / "cb.json"
    acct = _acct(9850.0, _curve([("2026-11-11", 10000.0)]))
    check_circuit_breaker(acct, "2026-11-12", state_path=path)
    # Fresh load of the same file: latch persists.
    status, _ = check_circuit_breaker(acct, "2026-11-12", state_path=path)
    assert status["blocked"] is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["day"]["tripped"] is True
    assert saved["day"]["date"] == "2026-11-12"
    assert len(saved["events"]) <= 50


def test_decision_rows_for_trip_and_reset():
    trip = {
        "type": "trip", "scope": "day", "trading_date": "2026-11-12",
        "ref_equity": 10000.0, "equity": 9850.0, "pct": -1.5, "ts": "2026-11-12T10:00:00-05:00",
    }
    row = breaker_decision_row(trip, "2026-11-12")
    assert row["action"] == "CIRCUIT_BREAKER_TRIP"
    assert row["trading_date"] == "2026-11-12"
    assert "熔断触发" in row["reason"]
    assert "照常管理" in row["reason"]

    reset = dict(trip, type="reset", scope="week", pct=-2.0)
    row2 = breaker_decision_row(reset, "2026-11-16")
    assert row2["action"] == "CIRCUIT_BREAKER_RESET"
    assert "复位" in row2["reason"]


def test_evaluate_does_not_mutate_account(tmp_path):
    acct = _acct(9850.0, _curve([("2026-11-11", 10000.0)]))
    before = json.dumps(acct, sort_keys=True)
    check_circuit_breaker(acct, "2026-11-12", state_path=tmp_path / "cb.json")
    assert json.dumps(acct, sort_keys=True) == before
    # Positions are untouched by the module: no exit gating exists here.
    assert acct["positions"] == {"intraday": [], "swing": []}


def test_reference_equities_use_last_point_before_each_boundary():
    acct = _acct(
        10100.0,
        _curve(
            [
                ("2026-11-06", 10000.0),
                ("2026-11-09", 10100.0),
                ("2026-11-10", 10200.0),
                ("2026-11-11", 9900.0),
                ("2026-11-12", 10100.0),  # today: must NOT be a reference
            ]
        ),
    )
    day_ref, week_ref = reference_equities(acct, "2026-11-12")
    assert day_ref == 9900.0  # last point before today
    assert week_ref == 10000.0  # last point before Monday 2026-11-09
