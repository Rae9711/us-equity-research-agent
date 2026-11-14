"""Tests for the Alpaca paper mirror (reconciliation job).

All Alpaca HTTP is mocked (httpx.MockTransport). No real request is ever
sent and no real key is needed. Quote lookups are stubbed.
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from src.broker.alpaca_client import AlpacaPaperClient
from src.jobs.alpaca_mirror_sync import run as mirror_run
from src.paper.alpaca_mirror import (
    fresh_mirror_state,
    leg_key,
    load_mirror_state,
    plan_mirror_actions,
)
from src.utils.trading_calendar import ET


def _amd_leg(**over):
    leg = {
        "book": "swing",
        "symbol": "AMD",
        "direction": "LONG",
        "shares": 2,
        "stop": 493.4,
        "target": 643.27,
        "entry": 557.495,
        "horizon": "Swing",
        "opened_date": "2026-11-12",
    }
    leg.update(over)
    return leg


def _plan(legs=None, positions=None, orders=None, state=None, **kw):
    kw.setdefault("breaker_tripped_today", False)
    kw.setdefault("quotes", {"AMD": 530.64})
    return plan_mirror_actions(
        legs if legs is not None else [_amd_leg()],
        positions if positions is not None else [],
        orders if orders is not None else [],
        state if state is not None else fresh_mirror_state(),
        **kw,
    )


def test_plan_places_fresh_bracket_for_unmirrored_leg():
    plan = _plan()
    assert len(plan["actions"]) == 1
    action = plan["actions"][0]
    assert action["kind"] == "place_bracket"
    assert action["leg_key"] == "swing:AMD:2026-11-12"
    signal = action["signal"]
    assert signal["symbol"] == "AMD"
    assert signal["direction"] == "LONG"
    assert signal["shares"] == 2
    assert signal["stop"] == 493.4
    assert signal["target"] == 643.27
    assert signal["entry"] == 557.495
    assert signal["horizon"] == "Swing"
    assert signal["client_order_id"] == "tos-mirror-amd-swing-2026-11-12"
    assert plan["divergences"] == []


def test_plan_skips_when_day_breaker_tripped():
    plan = _plan(breaker_tripped_today=True)
    assert plan["actions"] == []
    assert plan["skipped"][0]["reason"] == "breaker_tripped"


def test_plan_skips_when_quote_beyond_stop_or_target():
    plan = _plan(quotes={"AMD": 490.0})
    assert plan["actions"] == []
    assert plan["skipped"][0]["reason"] == "beyond_stop"
    plan = _plan(quotes={"AMD": 650.0})
    assert plan["actions"] == []
    assert plan["skipped"][0]["reason"] == "beyond_target"


def test_plan_skips_short_when_quote_beyond_stop():
    leg = _amd_leg(direction="SHORT", stop=600.0, target=500.0, entry=557.495)
    plan = _plan(legs=[leg], quotes={"AMD": 610.0})
    assert plan["actions"] == []
    assert plan["skipped"][0]["reason"] == "beyond_stop"


def test_plan_skips_when_no_quote():
    plan = _plan(quotes={"AMD": None})
    assert plan["actions"] == []
    assert plan["skipped"][0]["reason"] == "no_quote"


def test_plan_skips_already_mirrored_leg():
    state = fresh_mirror_state()
    state["legs"][leg_key("swing", "AMD", "2026-11-12")] = {"status": "placed", "order_id": "o1"}
    pos = [{"symbol": "AMD", "direction": "LONG", "qty": 2.0, "avg_entry_price": 557.5, "unrealized_pl": -50.0}]
    plan = _plan(positions=pos, state=state)
    assert plan["actions"] == []
    assert plan["skipped"][0]["reason"] == "mirrored"
    assert plan["divergences"] == []


def test_plan_flags_alpaca_flat_while_local_open():
    state = fresh_mirror_state()
    state["legs"][leg_key("swing", "AMD", "2026-11-12")] = {"status": "placed", "order_id": "o1"}
    plan = _plan(state=state)  # Alpaca flat, local leg still open
    assert plan["actions"] == []
    assert plan["divergences"][0]["type"] == "alpaca_flat_local_open"
    assert plan["mark_exited"] == ["swing:AMD:2026-11-12"]
    assert plan["skipped"][0]["reason"] == "alpaca_exited"


def test_plan_never_replaces_an_exited_record():
    state = fresh_mirror_state()
    state["legs"][leg_key("swing", "AMD", "2026-11-12")] = {"status": "alpaca_exited"}
    plan = _plan(state=state)
    assert plan["actions"] == []
    assert plan["divergences"][0]["type"] == "alpaca_flat_local_open"
    assert plan["mark_exited"] == []


def test_plan_closes_alpaca_position_with_no_local_leg():
    pos = [{"symbol": "NVDA", "direction": "SHORT", "qty": 6.0, "avg_entry_price": 176.0, "unrealized_pl": 0.0}]
    orders = [{"id": "o-nvda", "symbol": "NVDA", "direction": "SHORT", "type": "limit", "qty": 6.0, "status": "open", "order_class": "bracket"}]
    plan = _plan(legs=[], positions=pos, orders=orders)
    kinds = [(a["kind"], a.get("symbol")) for a in plan["actions"]]
    assert ("cancel_order", "NVDA") in kinds
    assert ("close_position", "NVDA") in kinds
    # Cancel comes before close in execution order (job sorts by kind).
    assert kinds.index(("cancel_order", "NVDA")) < kinds.index(("close_position", "NVDA"))


def test_plan_cancels_stray_order_with_no_position():
    orders = [{"id": "o-stray", "symbol": "MU", "direction": "LONG", "type": "market", "qty": 3.0, "status": "open", "order_class": "bracket"}]
    plan = _plan(legs=[], orders=orders)
    assert plan["actions"] == [
        {"kind": "cancel_order", "order_id": "o-stray", "symbol": "MU", "reason": "no_local_leg"}
    ]


def test_plan_flags_qty_mismatch_without_placing():
    state = fresh_mirror_state()
    state["legs"][leg_key("swing", "AMD", "2026-11-12")] = {"status": "placed", "order_id": "o1"}
    pos = [{"symbol": "AMD", "direction": "LONG", "qty": 5.0, "avg_entry_price": 557.5, "unrealized_pl": 0.0}]
    plan = _plan(positions=pos, state=state)
    assert plan["actions"] == []
    assert plan["divergences"][0]["type"] == "qty_mismatch"


def test_plan_flags_entry_basis_difference():
    state = fresh_mirror_state()
    state["legs"][leg_key("swing", "AMD", "2026-11-12")] = {"status": "placed", "order_id": "o1"}
    # Alpaca filled ~525 (Monday market) vs local entry 557.495: 5.8% apart.
    pos = [{"symbol": "AMD", "direction": "LONG", "qty": 2.0, "avg_entry_price": 525.0, "unrealized_pl": 0.0}]
    plan = _plan(positions=pos, state=state)
    assert plan["actions"] == []
    assert plan["divergences"][0]["type"] == "entry_basis_differs"


def test_plan_adopts_matching_alpaca_position():
    pos = [{"symbol": "AMD", "direction": "LONG", "qty": 2.0, "avg_entry_price": 557.5, "unrealized_pl": 0.0}]
    plan = _plan(positions=pos)
    assert plan["actions"] == []
    assert len(plan["adopt"]) == 1
    assert plan["adopt"][0]["record"]["status"] == "adopted"
    assert plan["skipped"][0]["reason"] == "adopted"


def test_plan_adopts_matching_resting_bracket():
    orders = [{"id": "o-amd", "symbol": "AMD", "direction": "LONG", "type": "market", "qty": 2.0, "status": "accepted", "order_class": "bracket", "client_order_id": "tos-x"}]
    plan = _plan(orders=orders)
    assert plan["actions"] == []
    assert plan["adopt"][0]["record"]["status"] == "placed"
    assert plan["adopt"][0]["record"]["order_id"] == "o-amd"


def test_plan_diverges_on_mismatched_alpaca_position_instead_of_doubling():
    pos = [{"symbol": "AMD", "direction": "SHORT", "qty": 2.0, "avg_entry_price": 557.5, "unrealized_pl": 0.0}]
    plan = _plan(positions=pos)
    assert plan["actions"] == []
    assert plan["divergences"][0]["type"] == "alpaca_position_mismatch"


# ---------------------------------------------------------------------------
# Job-level tests (mock transport, temp state dir)
# ---------------------------------------------------------------------------


def _write_account(tmp_path, legs):
    paper = tmp_path / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "account.json").write_text(
        json.dumps({"equity": 10131.13, "cash": 9069.85, "positions": {"swing": legs}}),
        encoding="utf-8",
    )
    return tmp_path


def _mock_client(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/v2/positions":
            return httpx.Response(200, json=[])
        if request.method == "GET" and request.url.path == "/v2/orders":
            return httpx.Response(200, json=[])
        if request.method == "GET" and request.url.path == "/v2/account":
            return httpx.Response(
                200,
                json={"equity": "10000", "cash": "10000", "buying_power": "40000", "status": "ACTIVE"},
            )
        if request.method == "POST" and request.url.path == "/v2/orders":
            body = json.loads(request.content)
            calls.append(("POST_BODY", body))
            return httpx.Response(
                200,
                json={"id": "order-1", "client_order_id": body["client_order_id"], "status": "accepted", "legs": []},
            )
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    return AlpacaPaperClient(key_id="PK_TEST", secret_key="secret", transport=transport)


def _quote_stub(symbol, trading_date, fallback):
    return 530.64, "stub"


def test_job_dry_run_plans_without_placing(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_account(tmp_path, [_amd_leg()])
    calls: list = []
    client = _mock_client(calls)
    saturday = datetime(2026, 11, 14, 10, 0, tzinfo=ET)  # market closed
    summary, code = mirror_run(
        tmp_path, dry_run=True, now=saturday, quote_fn=_quote_stub, client=client
    )
    assert code == 0
    assert summary["market_open"] is False
    assert len(summary["actions"]) == 1
    assert summary["actions"][0]["kind"] == "place_bracket"
    assert summary["deferred"][0]["would_execute"] is True
    assert summary["executed"] == []
    assert not any(c[0] == "POST" for c in calls)
    assert not (tmp_path / "paper" / "alpaca_mirror.json").exists()


def test_job_live_market_open_places_bracket(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_account(tmp_path, [_amd_leg()])
    calls: list = []
    client = _mock_client(calls)
    monday = datetime(2026, 11, 16, 10, 0, tzinfo=ET)  # market open
    summary, code = mirror_run(
        tmp_path, dry_run=False, now=monday, quote_fn=_quote_stub, client=client
    )
    assert code == 0
    assert summary["market_open"] is True
    assert len(summary["executed"]) == 1
    assert summary["executed"][0]["kind"] == "place_bracket"
    assert summary["executed"][0]["order_id"] == "order-1"
    post_bodies = [c[1] for c in calls if c[0] == "POST_BODY"]
    assert len(post_bodies) == 1
    assert post_bodies[0]["order_class"] == "bracket"
    assert post_bodies[0]["qty"] == 2
    assert post_bodies[0]["take_profit"]["limit_price"] == 643.27
    assert post_bodies[0]["stop_loss"]["stop_price"] == 493.4
    assert post_bodies[0]["time_in_force"] == "gtc"
    state = load_mirror_state(tmp_path)
    rec = state["legs"]["swing:AMD:2026-11-12"]
    assert rec["status"] == "placed"
    assert rec["order_id"] == "order-1"


def test_job_live_market_closed_defers_without_placing(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_account(tmp_path, [_amd_leg()])
    calls: list = []
    client = _mock_client(calls)
    saturday = datetime(2026, 11, 14, 10, 0, tzinfo=ET)
    summary, code = mirror_run(
        tmp_path, dry_run=False, now=saturday, quote_fn=_quote_stub, client=client
    )
    assert code == 0
    assert summary["market_open"] is False
    assert summary["deferred"][0]["reason"] == "market_closed"
    assert not any(c[0] == "POST" for c in calls)
