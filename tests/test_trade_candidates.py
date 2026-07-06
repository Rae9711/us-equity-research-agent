"""Tests for P18 Trade Candidates / Best Opportunity engine."""

from __future__ import annotations

from src.research.trade_candidates import (
    CANDIDATE_SYMBOLS,
    _instrument,
    _p16_gate,
    compute_trade_decision,
)


def _bullish_raw() -> dict:
    return {
        "trading_date": "2026-07-06",
        "market": {
            "quotes": {
                "QQQ": {"close": 520.0, "prev_close": 518.0, "high": 521.0, "low": 517.0, "change_pct": 0.4},
                "SPY": {"close": 600.0, "prev_close": 599.0, "change_pct": 0.2},
                "TQQQ": {"close": 80.0, "prev_close": 79.0, "change_pct": 1.2},
                "^VIX": {"close": 14.0, "prev_close": 15.0, "change_pct": -6.0},
            }
        },
        "sector": {
            "quotes": {
                "SMH": {"close": 280.0, "prev_close": 275.0, "high": 282.0, "low": 274.0, "change_pct": 1.8},
            }
        },
        "stocks": {
            "quotes": {
                "NVDA": {"close": 140.0, "prev_close": 138.0, "high": 141.0, "low": 137.0, "change_pct": 1.4},
                "TSLA": {"close": 250.0, "prev_close": 248.0, "change_pct": 0.8},
            }
        },
        "options": {"avg_implied_volatility": 0.20},
        "macro": {"economic_calendar": []},
    }


def test_candidate_symbols_order():
    assert CANDIDATE_SYMBOLS == ["NVDA", "QQQ", "SMH", "TSLA", "SPY", "TQQQ"]


def test_zero_dte_instrument():
    p9 = {"buy_options": "Yes", "zero_dte": "Yes", "buy_call": "Yes", "buy_put": "No"}
    assert _instrument("QQQ", "LONG", p9) == "QQQ 0DTE Call"


def test_stock_when_no_options():
    p9 = {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"}
    assert _instrument("NVDA", "LONG", p9) == "Stock"
    assert _instrument("QQQ", "LONG", p9) == "ETF"


def test_p16_no_trade_gate():
    parts = {"P16": {"judgment": "计划：不交易", "one_liner": "", "body_md": ""}}
    assert _p16_gate(parts, total=3) == "No Trade"


def test_bullish_momentum_picks_best():
    raw = _bullish_raw()
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
    }
    parts = {
        "P9": {
            "buy_options": "Yes",
            "zero_dte": "Yes",
            "buy_call": "Yes",
            "buy_put": "No",
        },
        "P11": {"scores": {"VIX": 1, "Bond": 0}},
        "P13": {"judgment": "Edge：YES"},
        "P16": {"judgment": "计划 Trade", "one_liner": "买 QQQ call"},
    }
    result = compute_trade_decision(
        raw,
        rule_bundle=rule_bundle,
        parts=parts,
        regime_label="AI Expansion",
        regime_confidence=0.75,
    )
    best = result["best_opportunity"]
    assert best["direction"] == "LONG"
    assert best["symbol"] in CANDIDATE_SYMBOLS
    assert "0DTE Call" in best["instrument"] or best["instrument"] in ("ETF", "Stock")
    assert best["confidence"] >= 55
    assert len(result["trade_candidates"]) == 6
    assert result["trade_candidates"][0]["rank"] == 1


def test_no_edge_returns_no_trade():
    raw = _bullish_raw()
    raw["options"]["avg_implied_volatility"] = 0.50
    rule_bundle = {"bias": "Neutral", "total": 0, "driver_type": "No Catalyst", "daily_driver": "No dominant catalyst"}
    parts = {
        "P9": {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"},
        "P11": {"scores": {"VIX": 0, "Bond": 0}},
        "P16": {"judgment": "Wait", "one_liner": "观望"},
    }
    result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)
    assert result["best_opportunity"]["direction"] == "NO TRADE"
