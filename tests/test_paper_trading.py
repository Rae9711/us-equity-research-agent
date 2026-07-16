"""Tests for paper (simulated) trading fills and exits.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json

import pytest

from src.paper.account import default_account, get_position, load_account, save_account
from src.paper.broker_sim import (
    InsufficientCashError,
    can_afford,
    execute_entry,
    execute_exit,
    size_shares,
)
from src.paper.execution_agent import decide_and_act
from src.paper.tick import run_paper_tick


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _write_morning(root, trading_date: str, primary: dict, swing: dict | None = None) -> None:
    day = root / "reports" / trading_date
    day.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_trades": {"primary": primary},
        "best_opportunity": primary,
        "advisory": True,
    }
    if swing:
        payload["swing_trade"] = swing
        payload["best_trades"]["swing"] = swing
    (day / "morning.json").write_text(json.dumps(payload), encoding="utf-8")


def test_size_shares_and_insufficient_cash(data_root):
    acct = default_account()
    acct["cash"] = 50.0
    acct["equity"] = 50.0
    save_account(acct)

    shares = size_shares(
        cash=50.0,
        equity=50.0,
        entry_price=340.0,
        stop_price=335.0,
        direction="LONG",
    )
    assert shares == 0

    shares2, err = can_afford(
        acct, price=340.0, stop=335.0, direction="LONG"
    )
    assert shares2 == 0
    assert err is not None

    with pytest.raises(InsufficientCashError):
        execute_entry(
            acct,
            symbol="ARM",
            direction="LONG",
            price=340.0,
            shares=1,
            stop=335.0,
            target=354.0,
            reason="test",
            trading_date="2026-07-08",
        )


def test_entry_fill_and_target_exit(data_root):
    acct = default_account()
    trade = execute_entry(
        acct,
        symbol="ARM",
        direction="LONG",
        price=341.0,
        shares=10,
        stop=335.0,
        target=354.0,
        reason="READY 入场",
        trading_date="2026-07-08",
        signal={"source": "morning_primary", "horizon": "Intraday"},
    )
    assert trade["action"] == "ENTRY"
    assert trade["book"] == "intraday"
    assert acct["position"]["symbol"] == "ARM"
    assert get_position(acct, "intraday")["symbol"] == "ARM"
    assert acct["cash"] == pytest.approx(10000.0 - 3410.0)
    assert len(acct["trades"]) == 1

    exit_trade = execute_exit(
        acct,
        price=354.0,
        reason="止盈",
        trading_date="2026-07-08",
    )
    assert exit_trade["action"] == "EXIT"
    assert exit_trade["pnl"] == pytest.approx(130.0)
    assert acct["position"] is None
    assert get_position(acct, "intraday") is None
    assert acct["realized_pnl"] == pytest.approx(130.0)
    assert acct["cash"] == pytest.approx(10000.0 + 130.0)


def test_stop_exit(data_root):
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="LONG",
        price=341.0,
        shares=5,
        stop=335.0,
        target=354.0,
        reason="entry",
        trading_date="2026-07-08",
    )
    out = execute_exit(acct, price=334.0, reason="止损", trading_date="2026-07-08")
    assert out["pnl"] == pytest.approx((334.0 - 341.0) * 5)


def test_decide_entry_when_ready(data_root, monkeypatch):
    primary = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_price": 341.0,
        "entry_zone": {"low": 340.0, "mid": 341.0, "high": 342.0},
        "stop_price": 335.0,
        "target_price": 354.0,
        "expected_return_pct": 3.8,
        "horizon": "Intraday",
    }
    _write_morning(data_root, "2026-07-08", primary)

    acct = default_account()
    decision = decide_and_act(
        acct,
        "2026-07-08",
        session_phase="open",
        force_price=341.0,
    )
    assert decision["action"] == "ENTRY"
    assert get_position(acct, "intraday")["symbol"] == "ARM"
    assert get_position(acct, "intraday")["shares"] >= 1
    assert acct["position"]["symbol"] == "ARM"


def test_decide_skip_when_missed(data_root):
    primary = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_price": 341.0,
        "entry_zone": {"low": 340.0, "mid": 341.0, "high": 342.0},
        "stop_price": 335.0,
        "target_price": 354.0,
        "horizon": "Intraday",
    }
    _write_morning(data_root, "2026-07-08", primary)
    # No swing → skip when gapped above
    acct = default_account()
    decision = decide_and_act(
        acct,
        "2026-07-08",
        session_phase="open",
        force_price=360.0,
    )
    assert decision["action"] == "SKIP"
    assert acct["position"] is None
    assert get_position(acct, "intraday") is None


def test_decide_exit_on_stop(data_root):
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="LONG",
        price=341.0,
        shares=5,
        stop=335.0,
        target=354.0,
        reason="entry",
        trading_date="2026-07-08",
        signal={"source": "morning_primary", "horizon": "Intraday"},
    )
    decision = decide_and_act(
        acct,
        "2026-07-08",
        session_phase="open",
        force_price=334.0,
    )
    assert decision["action"] == "EXIT"
    assert "止损" in (decision["reason"] or "")
    assert acct["position"] is None


def test_dual_books_eod_flattens_intraday_only(data_root, monkeypatch):
    acct = default_account()
    execute_entry(
        acct,
        symbol="ARM",
        direction="LONG",
        price=341.0,
        shares=5,
        stop=335.0,
        target=354.0,
        trading_date="2026-07-08",
        signal={"source": "morning_primary", "horizon": "Intraday"},
        book="intraday",
    )
    execute_entry(
        acct,
        symbol="NVDA",
        direction="LONG",
        price=120.0,
        shares=10,
        stop=110.0,
        target=140.0,
        trading_date="2026-07-08",
        signal={"source": "swing", "horizon": "Swing"},
        book="swing",
    )
    assert get_position(acct, "intraday")["symbol"] == "ARM"
    assert get_position(acct, "swing")["symbol"] == "NVDA"

    quotes = {"ARM": 341.0, "NVDA": 122.0}

    def _fake_quote(symbol, trading_date, **kwargs):
        return quotes.get(str(symbol).upper()), "test"

    monkeypatch.setattr(
        "src.paper.execution_agent.resolve_quote", _fake_quote
    )

    decision = decide_and_act(
        acct,
        "2026-07-08",
        session_phase="closed",
    )
    assert get_position(acct, "intraday") is None
    assert get_position(acct, "swing")["symbol"] == "NVDA"
    assert any(
        b.get("book") == "intraday" and b.get("action") == "EXIT"
        for b in (decision.get("books") or [])
    )


def test_dual_books_enter_both_when_ready(data_root):
    primary = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_price": 100.0,
        "entry_zone": {"low": 99.0, "mid": 100.0, "high": 101.0},
        "stop_price": 95.0,
        "target_price": 110.0,
        "horizon": "Intraday",
    }
    swing = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_price": 100.0,
        "entry_zone": {"low": 99.0, "mid": 100.0, "high": 101.0},
        "stop_price": 90.0,
        "target_price": 130.0,
        "horizon": "Swing",
    }
    _write_morning(data_root, "2026-07-08", primary, swing=swing)
    acct = default_account()
    decision = decide_and_act(
        acct,
        "2026-07-08",
        session_phase="open",
        force_price=100.0,
    )
    assert get_position(acct, "intraday") is not None
    assert get_position(acct, "swing") is not None
    assert decision["action"] == "ENTRY"
    books = {b["book"]: b["action"] for b in decision.get("books") or []}
    assert books.get("intraday") == "ENTRY"
    assert books.get("swing") == "ENTRY"


def test_run_paper_tick_persists(data_root):
    primary = {
        "symbol": "ARM",
        "direction": "LONG",
        "entry_price": 341.0,
        "entry_zone": {"low": 340.0, "mid": 341.0, "high": 342.0},
        "stop_price": 335.0,
        "target_price": 354.0,
        "horizon": "Intraday",
    }
    _write_morning(data_root, "2026-07-08", primary)
    result = run_paper_tick(
        "2026-07-08",
        session_phase="open",
        force_price=341.0,
        allow_non_trading_day=True,
    )
    assert result["ok"] is True
    assert result["action"] == "ENTRY"
    loaded = load_account()
    assert loaded["position"]["symbol"] == "ARM"
    assert get_position(loaded, "intraday")["symbol"] == "ARM"
    assert loaded["decisions"]
    assert (data_root / "paper" / "account.json").exists()
