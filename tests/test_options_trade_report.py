"""Options paper path, circuit breakers, and trade report cards."""

from __future__ import annotations

import pytest

from src.paper.account import default_account, get_position
from src.paper.allocation import daily_loss_breach, allocate_for_entry
from src.paper.broker_sim import execute_entry, execute_exit
from src.paper.exits import plan_exit
from src.paper.options_sim import (
    OptionChainUnavailable,
    can_afford_option,
    execute_option_entry,
    execute_option_exit,
    plan_option_exit,
    resolve_option_entry_quote,
    size_contracts,
)
from src.paper.trade_report import build_method_snapshot, build_trade_cards


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def test_size_contracts_and_afford():
    assert size_contracts(equity=10_000, cash=10_000, premium=2.0, risk_pct=1.0) == 0
    # risk $100 / ($2*100) = 0 contracts if risk_pct 1%; raise risk
    n = size_contracts(equity=10_000, cash=10_000, premium=1.0, risk_pct=2.0)
    assert n == 2  # $200 risk / $100 per contract
    acct = default_account()
    c, err = can_afford_option(acct, premium=1.5, risk_pct=3.0)
    assert c >= 1
    assert err is None


def test_option_entry_exit_forced_quote(data_root):
    acct = default_account()
    sig = {
        "symbol": "QQQ",
        "direction": "LONG",
        "instrument": "QQQ 0DTE Call",
        "horizon": "0DTE",
        "source": "test",
        "stop_price": 480.0,
        "target_price": 500.0,
        "win_prob": 62,
        "risk_reward": 2.0,
        "forced_option_quote": {
            "strike": 490.0,
            "premium": 2.50,
            "expiration": "2026-08-06",
            "zero_dte": True,
        },
    }
    q = resolve_option_entry_quote(sig, underlying_px=490.0, trading_date="2026-08-06")
    assert q["premium"] == 2.5
    assert q["strike"] == 490.0

    trade = execute_option_entry(
        acct,
        quote=q,
        contracts=2,
        signal=sig,
        reason="test option entry",
        trading_date="2026-08-06",
        book="intraday",
        underlying_stop=480.0,
        underlying_target=500.0,
    )
    assert trade["action"] == "ENTRY"
    assert trade["asset_class"] == "option"
    assert trade["method_snapshot"]["asset_class"] == "option"
    assert "narrative_zh" in trade
    pos = get_position(acct, "intraday")
    assert pos["asset_class"] == "option"
    assert pos["contracts"] == 2
    # Premium paid 2.5 * 100 * 2 = 500
    assert acct["cash"] == 9500.0

    exit_t = execute_option_exit(
        acct, premium=3.0, reason="test tp", trading_date="2026-08-06", book="intraday"
    )
    assert exit_t["action"] == "EXIT"
    assert exit_t["pnl"] == 100.0  # (3-2.5)*2*100
    assert exit_t["method_snapshot"]
    assert get_position(acct, "intraday") is None


def test_option_skip_without_chain(data_root):
    sig = {
        "symbol": "QQQ",
        "direction": "LONG",
        "instrument": "QQQ 0DTE Call",
        "horizon": "0DTE",
        # no forced quote → resolve will try network; force failure via empty forced incomplete
        "forced_option_quote": {"strike": None, "premium": None},
    }
    try:
        resolve_option_entry_quote(sig, underlying_px=490.0, trading_date="2026-08-06")
        raised = False
    except OptionChainUnavailable:
        raised = True
    assert raised is True


def test_option_premium_stop_plan():
    pos = {
        "asset_class": "option",
        "option_right": "call",
        "avg_entry": 2.0,
        "stop": 1.0,
        "target": 4.0,
        "shares": 1,
        "contracts": 1,
        "zero_dte": True,
        "instrument": "QQQ 0DTE Call",
    }
    plan = plan_option_exit(pos, 0.9, 480.0, {"option_premium_stop_pct": 50.0}, phase="open")
    assert plan["action"] == "exit"
    assert "止损" in plan["reason"] or "亏损" in plan["reason"]

    plan_eod = plan_option_exit(pos, 2.1, 490.0, {}, phase="closed")
    assert plan_eod["action"] == "exit"
    assert "0DTE" in plan_eod["reason"]


def test_max_loss_per_trade_r_exit():
    pos = {
        "direction": "LONG",
        "avg_entry": 100.0,
        "initial_stop": 98.0,
        "stop": 98.0,
        "risk_per_share": 2.0,
        "shares": 10,
        "target": 106.0,
        "high_water": 100.0,
        "low_water": 100.0,
    }
    plan = plan_exit(pos, 97.5, {"exit_management": True, "max_loss_per_trade_r": 1.0})
    # (97.5-100)/2 = -1.25R ≤ -1R
    assert plan["action"] == "exit"
    assert "熔断" in plan["reason"]


def test_daily_loss_breach_blocks_allocate(data_root):
    acct = default_account()
    acct["trades"] = [
        {
            "action": "EXIT",
            "trading_date": "2026-08-06",
            "pnl": -250.0,
            "symbol": "NVDA",
        }
    ]
    hit, reason = daily_loss_breach(acct, "2026-08-06")
    assert hit is True
    assert "日亏熔断" in reason

    alloc = allocate_for_entry(
        acct,
        book="intraday",
        signal={
            "symbol": "ARM",
            "direction": "LONG",
            "entry_price": 100.0,
            "stop_price": 97.0,
            "target_price": 110.0,
            "win_prob": 66,
            "expected_return_pct": 5.0,
            "horizon": "Intraday",
            "entry_status": {"status": "READY"},
            "risk_reward": 3.0,
        },
        quote=100.0,
        entry_status={"status": "READY"},
        trading_date="2026-08-06",
    )
    assert alloc["hold_cash"] is True
    assert "日亏熔断" in alloc["reason_zh"]


def test_equity_trade_report_fields(data_root):
    acct = default_account()
    trade = execute_entry(
        acct,
        symbol="NVDA",
        direction="LONG",
        price=100.0,
        shares=10,
        stop=97.0,
        target=110.0,
        signal={
            "symbol": "NVDA",
            "instrument": "Stock",
            "horizon": "Intraday",
            "source": "test",
            "win_prob": 60,
            "risk_reward": 2.0,
            "entry_status": "READY",
        },
        reason="test entry",
        trading_date="2026-08-06",
        book="intraday",
    )
    assert trade["method_snapshot"]["what"]
    assert trade["narrative_zh"]
    exit_t = execute_exit(
        acct, price=105.0, reason="tp", trading_date="2026-08-06", book="intraday"
    )
    assert exit_t["pnl"] is not None
    assert exit_t["narrative_zh"]
    cards = build_trade_cards(acct, trading_date="2026-08-06")
    assert any(c.get("action") == "EXIT" for c in cards)
    assert any(c.get("method_snapshot") for c in cards)


def test_try_entry_option_via_decide(data_root):
    """Instrument with Call + forced quote routes to option fill, not shares."""
    from src.paper.execution_agent import decide_and_act

    primary = {
        "symbol": "QQQ",
        "direction": "LONG",
        "instrument": "QQQ 0DTE Call",
        "entry_price": 490.0,
        "entry_zone": {"low": 488.0, "mid": 490.0, "high": 492.0},
        "stop_price": 485.0,
        "target_price": 500.0,
        "win_prob": 62,
        "expected_return_pct": 4.0,
        "horizon": "0DTE",
        "forced_option_quote": {
            "strike": 490.0,
            "premium": 1.25,
            "expiration": "2026-08-06",
            "zero_dte": True,
        },
    }
    day = data_root / "reports" / "2026-08-06"
    day.mkdir(parents=True, exist_ok=True)
    import json

    (day / "morning.json").write_text(
        json.dumps(
            {
                "best_trades": {"primary": primary, "top_trades": [primary]},
                "best_opportunity": primary,
            }
        ),
        encoding="utf-8",
    )
    acct = default_account()
    decision = decide_and_act(
        acct, "2026-08-06", session_phase="open", force_price=490.0
    )
    assert decision["action"] == "ENTRY"
    pos = get_position(acct, "intraday")
    assert pos is not None
    assert pos["asset_class"] == "option"
    assert pos.get("strike") == 490.0
    # Paid premium not share notional
    assert acct["cash"] < 10_000
    assert float(pos["avg_entry"]) == 1.25


def test_build_method_snapshot_option():
    snap = build_method_snapshot(
        signal={"symbol": "QQQ", "instrument": "QQQ 0DTE Call", "win_prob": 60},
        trade={
            "asset_class": "option",
            "instrument": "QQQ 0DTE Call",
            "option_right": "call",
            "strike": 490,
            "contracts": 2,
            "symbol": "QQQ",
            "direction": "LONG",
        },
        allocation={"risk_pct": 1.5, "cash_reserve_pct": 15, "expected_r": 0.4},
    )
    assert snap["asset_class"] == "option"
    assert "行权价" in snap["what"]
