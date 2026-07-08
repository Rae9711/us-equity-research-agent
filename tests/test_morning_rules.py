"""Tests for morning rule engine — driver taxonomy and options logic."""

from __future__ import annotations

from src.research.rules import (
    DRIVER_TYPES,
    _build_p16,
    _build_p9_options,
    _daily_driver_type_and_driver,
    compute_rule_parts,
)
from src.utils.driver_match import driver_match_level


def _minimal_raw(trading_date: str = "2026-03-18") -> dict:
    return {
        "trading_date": trading_date,
        "prior_trading_day": "2026-03-17",
        "market": {
            "quotes": {
                "SPY": {"close": 500, "prev_close": 499, "change_pct": 0.2},
                "QQQ": {"close": 420, "prev_close": 418, "high": 422, "low": 415, "change_pct": 0.3},
                "DIA": {"close": 380, "prev_close": 379, "change_pct": 0.1},
                "DX-Y.NYB": {"close": 104, "change_pct": 0.1},
                "^VIX": {"close": 16, "change_pct": -1},
            }
        },
        "sector": {
            "quotes": {
                "SMH": {"close": 220, "prev_close": 225, "change_pct": -2.5},
                "XLK": {"change_pct": -0.5},
            }
        },
        "stocks": {"quotes": {"NVDA": {"close": 120, "change_pct": -2.0}}},
        "macro": {
            "series": {"DGS10": {"value": "4.3"}, "DGS2": {"value": "4.1"}},
            "economic_calendar": [
                {"release_name": "FOMC Minutes", "date": trading_date},
            ],
        },
        "options": {"avg_implied_volatility": 0.3},
        "news": {
            "polygon": [{"title": "Iran tensions escalate", "description": "conflict"}],
            "rss": [],
        },
    }


def test_no_catalyst_day_not_macro():
    dtype, driver = _daily_driver_type_and_driver(
        catalysts_today=[],
        chip_selloff=False,
        smh_pct=0.8,
        strongest_sector="SMH",
        qqq_pct=0.2,
    )
    assert dtype == "Momentum"
    assert driver == "AI Momentum"
    assert dtype != "Macro"
    assert dtype in DRIVER_TYPES


def test_nfp_day_is_macro():
    dtype, driver = _daily_driver_type_and_driver(
        catalysts_today=[{"name": "NFP"}],
        chip_selloff=False,
        smh_pct=-1.0,
        strongest_sector="XLF",
        qqq_pct=0.1,
    )
    assert dtype == "Macro"
    assert driver == "NFP"


def test_p9_buy_no_implies_zero_dte_no():
    p9 = _build_p9_options(
        catalyst_today=True,
        pre_holiday=False,
        iv_level="Low",
        total=2,
        options={},
    )
    assert p9["buy_options"] == "No"
    assert p9["zero_dte"] == "No"


def test_vague_macro_driver_mismatch():
    assert driver_match_level("Macro", "AI Momentum", morning_driver_type="Macro") == "错"


def test_p16_has_long_short_wait_conditions():
    p16 = _build_p16(
        qqq_q={"high": 422, "low": 415},
        prior_raw={"market": {"quotes": {"QQQ": {"high": 422, "low": 415, "close": 418}}}},
        total=1,
        bias="Neutral",
        catalyst_today=True,
        chip_selloff=False,
        p16_gate_hint="Wait",
        macro_calendar={
            "catalysts": [{"name": "FOMC Minutes", "category": "scheduled"}],
            "headline": "FOMC Minutes",
        },
        driver_tree={"primary": {"type": "Fed", "label": "FOMC Minutes"}},
    )
    plan = p16["trade_plan"]
    assert len(plan["conditions_long"]) >= 1
    assert len(plan["conditions_short"]) >= 1
    assert len(plan["conditions_wait"]) >= 1
    assert any("FOMC" in c for c in plan["conditions_wait"])


def test_compute_rule_parts_includes_macro_calendar_and_driver_tree():
    from unittest.mock import patch

    with patch("src.research.rules._load_prior_raw", return_value={}):
        bundle = compute_rule_parts(_minimal_raw())
    assert bundle.get("macro_calendar")
    assert bundle["macro_calendar"]["has_material_catalyst"] is True
    assert bundle.get("driver_tree")
    assert bundle["driver_tree"]["primary"] is not None
    assert bundle.get("trade_plan")
