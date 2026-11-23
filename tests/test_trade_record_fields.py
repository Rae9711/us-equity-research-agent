"""Regression: post-call annotations on a trade must persist to the ledger.

2026-11-19: the AMD intraday stop exit (R=-1.18, fill hit the 0.5%
slippage cap) reached the ledger WITHOUT slippage_capped /
gap_through_stop / stop_trigger / fill_price — only the reason text said
the cap was hit. Root cause: ``append_trade`` stored a copy of the trade
dict while ``execute_entry``/``execute_exit`` returned the pre-copy dict,
so every annotation the management/entry paths added afterwards (stop
fields, target fields, allocation, quote_outage_exit) was lost from the
persisted record.

These tests assert on ``account["trades"]`` — the persisted rows — not
on the returned dict alone (the older tests only checked the returned
dict, which is why the bug slipped through).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import pytest

from src.paper.account import default_account, get_position
from src.paper.broker_sim import execute_entry, execute_exit
from src.paper.execution_agent import (
    _force_outage_exit_if_due,
    decide_and_act,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _raw_bar(symbol, trading_date, *, open, high, low, close):
    return {
        "stocks": {
            "quotes": {
                symbol: {
                    "open": open,
                    "high": high,
                    "low": low,
                    "close": close,
                    "date": trading_date,
                }
            }
        }
    }


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
        book=book,
    )


# ---------------------------------------------------------------------------
# Mechanism: the returned trade IS the stored ledger row
# ---------------------------------------------------------------------------


def test_execute_entry_returns_stored_row(data_root):
    acct = default_account()
    trade = _open_leg(
        acct, "AMD", book="intraday", price=340.0, shares=5,
        stop=335.0, target=360.0, trading_date="2026-08-24",
    )
    # Simulate the entry path annotating allocation after the call.
    trade["allocation"] = {"cash_reserve_pct": 11.0, "risk_pct": 1.39}
    trade["position_pct"] = 17.0

    assert trade is acct["trades"][-1]
    row = acct["trades"][-1]
    assert row["allocation"] == {"cash_reserve_pct": 11.0, "risk_pct": 1.39}
    assert row["position_pct"] == 17.0


def test_execute_exit_returns_stored_row(data_root):
    acct = default_account()
    _open_leg(
        acct, "ARM", book="intraday", price=341.0, shares=5,
        stop=335.0, target=354.0, trading_date="2026-08-24",
    )
    trade = execute_exit(
        acct, price=345.0, reason="manual", trading_date="2026-08-24",
        book="intraday",
    )
    # Simulate the stop branch annotating fill fields after the call.
    trade["stop_trigger"] = 335.0
    trade["fill_price"] = 345.0
    trade["slippage_capped"] = True
    trade["gap_through_stop"] = False

    assert trade is acct["trades"][-1]
    row = acct["trades"][-1]
    assert row["stop_trigger"] == 335.0
    assert row["fill_price"] == 345.0
    assert row["slippage_capped"] is True
    assert row["gap_through_stop"] is False


# ---------------------------------------------------------------------------
# End to end: bar-scan stop (the 2026-11-19 AMD case, capped fill)
# ---------------------------------------------------------------------------


def test_bar_scan_stop_fields_persist_to_ledger(data_root):
    """AMD 2026-11-19 replay: entry 549.65, stop 534.56, 1m bar low 531.0
    -> fill capped at stop - 0.5% = 531.8872, R=-1.18. The ledger row must
    carry the fill evidence, not just the reason text."""
    acct = default_account()
    _open_leg(
        acct, "AMD", book="intraday", price=549.65, shares=3,
        stop=534.56, target=580.0, trading_date="2026-08-24",
    )
    raw = _raw_bar(
        "AMD", "2026-08-24", open=548.0, high=550.0, low=531.0, close=532.0
    )
    bars = [{
        "ts": "2026-08-24T10:15:00-05:00",
        "open": 548.0, "high": 550.0, "low": 531.0, "close": 532.0,
    }]
    decision = decide_and_act(
        acct, "2026-08-24", session_phase="open", raw=raw,
        bars_by_symbol={"AMD": bars},
    )

    assert decision["action"] == "EXIT"
    assert "成交价触滑点上限 0.5%" in (decision["reason"] or "")
    assert decision["trade"] is acct["trades"][-1]
    row = acct["trades"][-1]
    assert row["action"] == "EXIT"
    assert row["stop_trigger"] == pytest.approx(534.56)
    assert row["fill_price"] == pytest.approx(531.8872, abs=0.01)
    assert row["slippage_capped"] is True
    assert row["gap_through_stop"] is False
    assert row["bar_ts"] == "2026-08-24T10:15:00-05:00"
    assert row["r_multiple"] == pytest.approx(-1.18, abs=0.01)
    assert get_position(acct, "intraday") is None


def test_bar_scan_target_fields_persist_to_ledger(data_root):
    acct = default_account()
    _open_leg(
        acct, "AVGO", book="intraday", price=340.0, shares=5,
        stop=330.0, target=350.0, trading_date="2026-08-24",
    )
    raw = _raw_bar(
        "AVGO", "2026-08-24", open=345.0, high=352.0, low=344.0, close=351.0
    )
    bars = [{
        "ts": "2026-08-24T14:27:00-05:00",
        "open": 345.0, "high": 352.0, "low": 344.0, "close": 351.0,
    }]
    decision = decide_and_act(
        acct, "2026-08-24", session_phase="open", raw=raw,
        bars_by_symbol={"AVGO": bars},
    )

    assert decision["action"] == "EXIT"
    assert decision["trade"] is acct["trades"][-1]
    row = acct["trades"][-1]
    assert row["fill_price"] == pytest.approx(350.0)
    assert row["gap_through_target"] is False
    assert row["bar_ts"] == "2026-08-24T14:27:00-05:00"
    assert row["r_multiple"] == pytest.approx(1.0)


def test_single_price_stop_fields_persist_to_ledger(data_root):
    """No 1-minute bars (bars_by_symbol={}) -> the single-price stop branch
    must still record its fill evidence on the ledger row."""
    acct = default_account()
    _open_leg(
        acct, "MU", book="intraday", price=340.0, shares=5,
        stop=335.0, target=360.0, trading_date="2026-08-24",
    )
    raw = _raw_bar(
        "MU", "2026-08-24", open=339.0, high=340.0, low=333.0, close=334.0
    )
    decision = decide_and_act(
        acct, "2026-08-24", session_phase="open", raw=raw,
        bars_by_symbol={},
    )

    assert decision["action"] == "EXIT"
    assert decision["trade"] is acct["trades"][-1]
    row = acct["trades"][-1]
    assert row["stop_trigger"] == pytest.approx(335.0)
    assert row["fill_price"] == pytest.approx(334.0)
    assert row["slippage_capped"] is False
    assert row["gap_through_stop"] is False


def test_quote_outage_exit_flag_persists_to_ledger(data_root):
    acct = default_account()
    _open_leg(
        acct, "NVDA", book="intraday", price=340.0, shares=5,
        stop=335.0, target=360.0, trading_date="2026-08-24",
    )
    pos = get_position(acct, "intraday")
    pos["last_price"] = 340.0
    base: dict = {}
    forced = _force_outage_exit_if_due(
        acct, "2026-08-24", "intraday", pos, base, 3, {},
    )

    assert forced is True
    assert base["trade"] is acct["trades"][-1]
    assert acct["trades"][-1]["quote_outage_exit"] is True
    assert get_position(acct, "intraday") is None
