"""Wiring tests for the day/week circuit breaker (item 4).

These tests only pass once docs/circuit-breaker-wiring.patch is applied:
they exercise decide_and_act / run_paper_tick with the breaker wired in.

- A tripped breaker blocks ALL new entries in both books (and the legacy
  single-book path) but never blocks exits.
- Trip/reset events land in the account decision record.
- The tick summary exposes the breaker status (daily-report payload hook).
"""

from __future__ import annotations

import pytest

from src.paper.account import (
    default_account,
    get_position,
    save_account,
)
from src.paper.execution_agent import decide_and_act
from src.paper.tick import run_paper_tick


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


@pytest.fixture
def open_gate(monkeypatch):
    gate = {
        "blocked_symbols": [],
        "global_block": False,
        "broad_failed": False,
        "whole_collection_failed": False,
        "source": "test",
        "reasons": [],
    }
    monkeypatch.setattr(
        "src.paper.execution_agent.news_gate_for_date",
        lambda *a, **k: gate,
    )
    return gate


def _picked(symbol, *, price=341.0, stop=335.0, target=354.0,
            direction="LONG", book="intraday", win_prob=60):
    return {
        "action": "enter",
        "signal": {
            "symbol": symbol,
            "direction": direction,
            "entry_price": price,
            "entry_zone": {"low": price - 1, "mid": price, "high": price + 1},
            "stop_price": stop,
            "target_price": target,
            "horizon": "Swing" if book == "swing" else "Intraday",
            "source": "morning_primary",
            "win_prob": win_prob,
        },
        "quote": price,
        "quote_source": "test",
        "entry_status": {"status": "READY"},
        "reason": "test",
    }


def _tripped_account(*, equity=9850.0, prior_close=10000.0):
    acct = default_account()
    acct["cash"] = equity
    acct["equity"] = equity
    acct["realized_pnl"] = 0.0
    acct["equity_curve"] = [
        {"ts": "2026-11-11T16:00:00-05:00", "equity": prior_close,
         "cash": prior_close, "label": "tick:HOLD"},
    ]
    return acct


def _open_leg(acct, symbol, *, book, price, shares, stop, target,
              trading_date, direction="LONG"):
    from src.paper.broker_sim import execute_entry

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


def _bar(ts, open_, high, low, close):
    return {"ts": ts, "open": open_, "high": high, "low": low, "close": close}


def test_breaker_blocks_new_entries_both_books(data_root, open_gate, monkeypatch):
    acct = _tripped_account()
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_intraday_signal",
        lambda *a, **k: _picked("ARM", book="intraday"),
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_swing_signal",
        lambda *a, **k: _picked("AMD", price=557.0, stop=493.4,
                                target=643.27, book="swing"),
    )
    decision = decide_and_act(acct, "2026-11-12", session_phase="open")
    assert decision["circuit_breaker"]["blocked"] is True
    assert decision["circuit_breaker"]["tripped_day"] is True
    assert get_position(acct, "intraday") is None
    assert get_position(acct, "swing") is None
    skips = [b for b in (decision.get("books") or [])
             if b.get("book") == "circuit_breaker"]
    assert skips and skips[0]["action"] == "SKIP"
    assert "熔断" in (skips[0]["reason"] or "")
    trips = [d for d in acct.get("decisions", [])
             if d.get("action") == "CIRCUIT_BREAKER_TRIP"]
    assert trips, "trip event must be recorded in the decision record"


def test_breaker_does_not_block_when_flat(data_root, open_gate, monkeypatch):
    acct = _tripped_account(equity=9990.0)  # -0.10% vs prior close
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_intraday_signal",
        lambda *a, **k: _picked("ARM", book="intraday"),
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_swing_signal",
        lambda *a, **k: {"action": "skip", "reason": "test", "signal": None},
    )
    decision = decide_and_act(acct, "2026-11-12", session_phase="open")
    assert decision["circuit_breaker"]["blocked"] is False
    assert get_position(acct, "intraday") is not None
    assert get_position(acct, "intraday")["symbol"] == "ARM"


def test_breaker_does_not_block_stop_exit(data_root, open_gate, monkeypatch):
    acct = _tripped_account(equity=9850.0)
    _open_leg(acct, "ARM", book="intraday", price=341.0, shares=5,
              stop=335.0, target=354.0, trading_date="2026-11-09")
    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote",
        lambda *a, **k: (338.0, "test"),
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_intraday_signal",
        lambda *a, **k: _picked("ARM", book="intraday"),
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_swing_signal",
        lambda *a, **k: {"action": "skip", "reason": "test", "signal": None},
    )
    bars = [_bar("2026-11-12 09:31", 340.0, 341.0, 339.0, 340.0),
            _bar("2026-11-12 09:32", 340.0, 340.0, 334.0, 336.0)]
    decision = decide_and_act(
        acct, "2026-11-12", session_phase="open",
        bars_by_symbol={"ARM": bars},
    )
    exits = [b for b in (decision.get("books") or [])
             if b.get("book") == "intraday" and b.get("action") == "EXIT"]
    assert exits, "stop exit must still execute while the breaker is tripped"
    assert get_position(acct, "intraday") is None
    assert decision["circuit_breaker"]["blocked"] is True


def test_breaker_blocks_single_book_path(data_root, open_gate, monkeypatch):
    acct = _tripped_account()
    acct["params"]["dual_books"] = False
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_signal",
        lambda *a, **k: _picked("ARM", book="intraday"),
    )
    decision = decide_and_act(acct, "2026-11-12", session_phase="open")
    assert decision["circuit_breaker"]["blocked"] is True
    assert decision["action"] == "SKIP"
    assert "熔断" in (decision["reason"] or "")
    assert get_position(acct, "intraday") is None


def test_tick_summary_exposes_breaker_status(data_root, open_gate, monkeypatch):
    acct = _tripped_account()
    save_account(acct)
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_intraday_signal",
        lambda *a, **k: _picked("ARM", book="intraday"),
    )
    monkeypatch.setattr(
        "src.paper.execution_agent.pick_swing_signal",
        lambda *a, **k: {"action": "skip", "reason": "test", "signal": None},
    )
    summary = run_paper_tick("2026-11-12", session_phase="open")
    assert summary["circuit_breaker"]["blocked"] is True
    assert summary["circuit_breaker"]["tripped_day"] is True
