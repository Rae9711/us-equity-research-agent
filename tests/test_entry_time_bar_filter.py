"""Tests for D1: _scan_bars_for_exit must not exit on pre-entry bars.

A leg that exits and re-enters the same day used to be exited instantly
on a stale pre-entry touch, because the scan ran over the whole day's
1-minute bars with no filter on the position's entry time
(``opened_at``). Only bars at/after the entry time may now trigger an
exit; bars before it are skipped for touch purposes (their closes still
feed the previous-close chain used for gap detection).
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from src.paper.account import default_account, get_position
from src.paper.broker_sim import execute_entry
from src.paper.execution_agent import (
    _scan_bars_for_exit,
    decide_and_act,
)
from src.utils.trading_calendar import ET


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _bar(ts, open_, high, low, close):
    return {"ts": ts, "open": open_, "high": high, "low": low, "close": close}


def _open_leg(acct, symbol, *, book, price, shares, stop, target,
              trading_date, direction="LONG"):
    return execute_entry(
        acct,
        symbol=symbol,
        direction=direction,
        price=price,
        shares=shares,
        stop=stop,
        target=target,
        reason="test",
        trading_date=trading_date,
        signal={
            "source": "test",
            "horizon": "Swing" if book == "swing" else "Intraday",
        },
        book=book,
    )


def _fake_quote(price):
    return lambda *a, **k: (price, "test")


def _exit_row(decision, book):
    rows = [
        b for b in (decision.get("books") or [])
        if b.get("book") == book and b.get("action") == "EXIT"
    ]
    return rows[0] if rows else None


def test_reentered_leg_does_not_exit_on_pre_entry_touch(data_root, monkeypatch):
    """Synthetic repro of D1: exit, re-enter the same day, and the stale
    09:32 pre-entry stop touch must NOT exit the new leg."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(338.0)
    )
    bars = [
        _bar("2026-11-09T09:31:00-05:00", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09T09:32:00-05:00", 340.0, 340.0, 334.0, 336.0),
        _bar("2026-11-09T10:30:00-05:00", 336.0, 337.0, 336.0, 336.5),
    ]
    # First leg exits on the 09:32 stop touch.
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    assert _exit_row(decision, "intraday") is not None
    assert get_position(acct, "intraday") is None

    # Re-enter the same symbol later the same day (10:00 ET). The 09:32
    # touch predates this leg and must not exit it.
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    get_position(acct, "intraday")["opened_at"] = "2026-11-09T10:00:00-05:00"
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None


def test_post_entry_touch_still_exits(data_root, monkeypatch):
    """A genuine touch after the entry time still exits, at the trigger."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    get_position(acct, "intraday")["opened_at"] = "2026-11-09T10:00:00-05:00"
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(338.0)
    )
    bars = [
        # Pre-entry touch: ignored.
        _bar("2026-11-09T09:31:00-05:00", 340.0, 341.0, 334.0, 336.0),
        # Post-entry touch: exits.
        _bar("2026-11-09T10:30:00-05:00", 336.0, 337.0, 334.0, 335.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert row["trade"]["price"] == pytest.approx(334.0)
    assert row["trade"]["bar_ts"] == "2026-11-09T10:30:00-05:00"
    assert get_position(acct, "intraday") is None


def test_missing_opened_at_falls_back_to_full_scan(
    data_root, monkeypatch, caplog
):
    """No entry timestamp: fall back to scanning the full day (old
    behavior) and log it, instead of silently dropping touches."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    get_position(acct, "intraday")["opened_at"] = None
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(338.0)
    )
    bars = [
        _bar("2026-11-09T09:31:00-05:00", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09T09:32:00-05:00", 340.0, 340.0, 334.0, 336.0),
    ]
    with caplog.at_level(logging.WARNING, logger="src.paper.execution_agent"):
        decision = decide_and_act(
            acct, "2026-11-09", session_phase="open",
            bars_by_symbol={"ARM": bars},
        )
    assert _exit_row(decision, "intraday") is not None
    assert "no opened_at" in caplog.text


def test_scan_bars_for_exit_entry_ts_unit():
    """Direct unit check of the entry-time filter on the scan function."""
    entry = ET.localize(datetime(2026, 11, 9, 10, 0))
    pre_only = [
        _bar("2026-11-09T09:31:00-05:00", 340.0, 341.0, 334.0, 336.0),
    ]
    assert (
        _scan_bars_for_exit("LONG", 335.0, 354.0, pre_only, None,
                            entry_ts=entry)
        is None
    )
    with_post = pre_only + [
        _bar("2026-11-09T10:30:00-05:00", 336.0, 337.0, 334.0, 335.0),
    ]
    hit = _scan_bars_for_exit("LONG", 335.0, 354.0, with_post, None,
                              entry_ts=entry)
    assert hit is not None
    assert hit[0] == "stop"
    assert hit[1]["ts"] == "2026-11-09T10:30:00-05:00"
    # Without an entry time the same bars scan in full (old behavior).
    hit_full = _scan_bars_for_exit("LONG", 335.0, 354.0, with_post, None)
    assert hit_full is not None
    assert hit_full[1]["ts"] == "2026-11-09T09:31:00-05:00"
