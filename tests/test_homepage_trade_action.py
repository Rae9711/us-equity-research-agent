"""Homepage trade badge: P16 Wait / No Trade closes stocks too → No Trade."""

from __future__ import annotations

from src.web.homepage import _has_stock_setup, _resolve_p16_gate, _trade_action


def _morning_p16_no_trade_with_mu() -> dict:
    return {
        "total_score": -3,
        "bias": "Bearish Bias",
        "index_trade": "NO TRADE",
        "parts": {
            "P16": {
                "judgment": "计划：No Trade",
                "one_liner": "指数观望",
                "body_md": "",
            },
            "P13": {"judgment": "Edge：1/4 YES", "edges": {"stock_edge": {"edge": "YES"}}},
        },
        "best_opportunity": {
            "direction": "SHORT",
            "symbol": "MU",
            "p16_gate": "No Trade",
        },
        "best_trades": {
            "index_trade": "NO TRADE",
            "p16_gate": "No Trade",
            "primary": {
                "symbol": "MU",
                "direction": "SHORT",
                "trade_action": "Small",
                "entry_price": 913.0,
            },
        },
    }


def test_resolve_p16_gate_from_best_opportunity():
    assert _resolve_p16_gate(_morning_p16_no_trade_with_mu()) == "No Trade"


def test_has_stock_setup_from_primary():
    assert _has_stock_setup(_morning_p16_no_trade_with_mu()) is True


def test_p16_no_trade_with_stock_setup_is_no_trade():
    morning = _morning_p16_no_trade_with_mu()
    assert _trade_action(morning, None) == "No Trade"


def test_p16_no_trade_with_step4_stock_is_no_trade():
    morning = _morning_p16_no_trade_with_mu()
    step4 = {
        "should_trade": True,
        "stock_trade": "SHORT MU",
        "index_trade": "NO TRADE",
    }
    assert _trade_action(morning, step4) == "No Trade"


def test_p16_trade_with_stock_remains_trade():
    morning = _morning_p16_no_trade_with_mu()
    morning["best_opportunity"]["p16_gate"] = "Trade"
    morning["best_trades"]["p16_gate"] = "Trade"
    morning["parts"]["P16"]["judgment"] = "计划：Trade"
    assert _trade_action(morning, None) == "Trade"


def test_p16_wait_with_stock_is_no_trade():
    morning = _morning_p16_no_trade_with_mu()
    morning["best_opportunity"]["p16_gate"] = "Wait"
    morning["best_trades"]["p16_gate"] = "Wait"
    morning["parts"]["P16"]["judgment"] = "计划：Wait"
    assert _trade_action(morning, None) == "No Trade"


def test_no_setup_p16_no_trade_is_no_trade():
    morning = _morning_p16_no_trade_with_mu()
    morning["best_trades"]["primary"] = None
    morning["best_opportunity"] = {"direction": "NO TRADE", "p16_gate": "No Trade"}
    assert _trade_action(morning, None) == "No Trade"
