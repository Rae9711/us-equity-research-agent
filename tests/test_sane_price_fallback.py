"""Tests for D3: sane-price guard fallback to the last 1-minute bar close.

When the guard rejects the current (live) quote, position management
used to refuse the stop/target/close exit entirely and keep holding.
It now falls back to the last 1-minute bar close already fetched for
that symbol, provided THAT price also passes the sane check against the
same anchors; when the fallback is also insane or unavailable, the old
refuse-and-hold behavior is kept.
"""

from __future__ import annotations

import logging

import pytest

from src.paper.account import default_account, get_position
from src.paper.broker_sim import execute_entry
from src.paper.execution_agent import decide_and_act


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
            # Mirror production _try_entry: the broker's sane check anchors
            # on the planned entry, so the test signal must carry it.
            "entry_price": price,
            "stop_price": stop,
            "target_price": target,
            "entry_zone": {"low": price - 1, "mid": price, "high": price + 1},
        },
        book=book,
    )


def _fake_quote(price):
    return lambda *a, **k: (price, "test")


def _book_rows(decision, book):
    return [b for b in (decision.get("books") or []) if b.get("book") == book]


def _exit_row(decision, book):
    rows = [b for b in _book_rows(decision, book) if b.get("action") == "EXIT"]
    return rows[0] if rows else None


def test_rejected_quote_falls_back_to_sane_bar_close(
    data_root, monkeypatch, caplog
):
    """Rejected live quote ($100 vs ~$341 anchors) + sane last bar close
    ($334.50, below the $335 stop) -> exit proceeds on the bar close."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(100.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09 09:32", 340.0, 341.0, 336.0, 334.5),
    ]
    with caplog.at_level(logging.INFO, logger="src.paper.execution_agent"):
        decision = decide_and_act(
            acct, "2026-11-09", session_phase="open",
            bars_by_symbol={"ARM": bars},
        )
    row = _exit_row(decision, "intraday")
    assert row is not None
    assert row["trade"]["price"] == pytest.approx(334.5)
    assert row["quote"] == pytest.approx(334.5)
    assert row["quote_source"] == "minute_bar_close_fallback"
    assert row.get("quote_fallback") is True
    assert get_position(acct, "intraday") is None
    assert "last 1m bar close $334.5" in caplog.text


def test_both_insane_still_refuses_and_holds(data_root, monkeypatch):
    """Rejected live quote + insane last bar close -> still refuses the
    exit and holds, as before."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(100.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09 09:32", 340.0, 341.0, 336.0, 50.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    refused = [
        b for b in _book_rows(decision, "intraday") if b.get("quote_rejected")
    ]
    assert refused, "expected a refuse-and-hold row"
    assert refused[0]["action"] == "HOLD"
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None


def test_rejected_quote_without_bars_still_refuses(data_root, monkeypatch):
    """Rejected live quote with no bars available (no fallback) -> still
    refuses the exit and holds."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(100.0)
    )
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open", bars_by_symbol={},
    )
    refused = [
        b for b in _book_rows(decision, "intraday") if b.get("quote_rejected")
    ]
    assert refused, "expected a refuse-and-hold row"
    assert refused[0]["action"] == "HOLD"
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None


def test_sane_quote_uses_live_quote_no_fallback(data_root, monkeypatch):
    """Normal path unchanged: a sane live quote is used as-is, with no
    fallback flag and no fallback quote source."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(338.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09 09:32", 340.0, 341.0, 338.0, 340.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None
    hold = [
        b for b in _book_rows(decision, "intraday")
        if b.get("action") == "HOLD"
    ][0]
    assert hold["quote"] == pytest.approx(338.0)
    assert hold["quote_source"] == "test"
    assert "quote_fallback" not in hold


# ---------------------------------------------------------------------------
# 10% band + quote/bar agreement (2026-11-12): a genuine gap is accepted,
# two wild readings that disagree are still refused.
# ---------------------------------------------------------------------------

def test_band_is_10_percent_by_default():
    """The default band tightened 30% -> 10%: a quote 13% from every anchor
    is now insane (it was sane under 30%)."""
    from src.paper.price_guard import (
        DEFAULT_MAX_DEVIATION_PCT,
        is_sane_fill_price,
        prices_agree,
    )

    assert DEFAULT_MAX_DEVIATION_PCT == 10.0
    anchors = [341.0, 335.0, 354.0]
    ok, _ = is_sane_fill_price(360.0, anchors=anchors)
    assert ok is True  # 1.7% from the target anchor
    ok, reason = is_sane_fill_price(400.0, anchors=anchors)
    assert ok is False and "10%" in (reason or "")
    assert prices_agree(300.0, 301.0) is True
    assert prices_agree(300.0, 400.0) is False


def test_gapped_quote_agreeing_with_bars_is_accepted(data_root, monkeypatch):
    """Live quote $300 is 12% below every plan anchor (rejected at 10%),
    but the last 1m bar close $301 agrees with it: a real gap, not a bad
    tick. Management proceeds and the gap-through-stop exit fires."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(300.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09 09:32", 300.0, 301.0, 299.0, 301.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    row = _exit_row(decision, "intraday")
    assert row is not None, "gap-through-stop exit must fire"
    assert row.get("quote_rejected") is not True
    assert row.get("quote_agrees_with_bars") is True
    assert get_position(acct, "intraday") is None


def test_gapped_quote_disagreeing_with_bars_still_refuses(data_root, monkeypatch):
    """Live quote $300 (12% below anchors) and last bar close $400 (13%
    above the target anchor) disagree with each other: two wild readings,
    no independent confirmation — still refuse-and-hold."""
    acct = default_account()
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote(300.0)
    )
    bars = [
        _bar("2026-11-09 09:31", 340.0, 341.0, 339.0, 340.0),
        _bar("2026-11-09 09:32", 400.0, 401.0, 399.0, 400.0),
    ]
    decision = decide_and_act(
        acct, "2026-11-09", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    refused = [
        b for b in _book_rows(decision, "intraday") if b.get("quote_rejected")
    ]
    assert refused, "expected a refuse-and-hold row"
    assert refused[0]["action"] == "HOLD"
    assert _exit_row(decision, "intraday") is None
    assert get_position(acct, "intraday") is not None
