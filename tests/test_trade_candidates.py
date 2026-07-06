"""Tests for P18 Trade Candidates / Best Opportunity engine (v2)."""

from __future__ import annotations

from unittest.mock import patch

from src.research.edges import compute_edges, format_p13_from_edges
from src.research.trade_candidates import (
    CANDIDATE_SYMBOLS,
    FINAL_SCORE_THRESHOLD,
    _instrument,
    _p16_gate,
    _score_candidate_v2,
    compute_trade_decision,
)


def _bullish_raw() -> dict:
    return {
        "trading_date": "2026-07-06",
        "market": {
            "quotes": {
                "QQQ": {
                    "close": 520.0,
                    "prev_close": 518.0,
                    "high": 521.0,
                    "low": 517.0,
                    "change_pct": 0.4,
                    "session_type": "premarket",
                },
                "SPY": {"close": 600.0, "prev_close": 599.0, "change_pct": 0.2},
                "TQQQ": {"close": 80.0, "prev_close": 79.0, "change_pct": 1.2},
                "^VIX": {"close": 14.0, "prev_close": 15.0, "change_pct": -6.0},
            }
        },
        "sector": {
            "quotes": {
                "SMH": {
                    "close": 280.0,
                    "prev_close": 275.0,
                    "high": 282.0,
                    "low": 274.0,
                    "change_pct": 1.8,
                },
            }
        },
        "stocks": {
            "quotes": {
                "NVDA": {
                    "close": 140.0,
                    "prev_close": 138.0,
                    "high": 141.0,
                    "low": 137.0,
                    "change_pct": 1.4,
                    "session_type": "premarket",
                    "volume": 1000,
                },
                "TSLA": {
                    "close": 255.0,
                    "prev_close": 240.0,
                    "open": 246.0,
                    "change_pct": 2.5,
                    "volume": 5000,
                    "session_type": "premarket",
                    "high": 258,
                    "low": 238,
                },
            }
        },
        "news": {
            "polygon": [
                {"ticker": "TSLA", "title": "TSLA delivery beat"},
                {"ticker": "NVDA", "title": "NVDA AI demand"},
            ],
            "rss": [],
        },
        "options": {"avg_implied_volatility": 0.20},
        "macro": {"economic_calendar": []},
    }


def _mock_obs(symbol: str, raw: dict, prior_raw: dict, trading_day):  # noqa: ARG001
    section = "market" if symbol in ("QQQ", "SPY", "TQQQ") else (
        "sector" if symbol == "SMH" else "stocks"
    )
    q = ((raw.get(section) or {}).get("quotes") or {}).get(symbol) or {}
    close = float(q.get("close") or 100)
    prev = float(q.get("prev_close") or close * 0.99)
    open_px = float(q.get("open") or prev * 1.01)
    chg = q.get("change_pct")
    gap = (open_px - prev) / prev * 100 if prev else 0
    return {
        "ticker": symbol,
        "last": close,
        "close": close,
        "open": open_px,
        "prev_close": prev,
        "change_pct": chg,
        "gap_pct": round(gap, 2),
    }


def test_candidate_symbols_order():
    assert CANDIDATE_SYMBOLS[0] == "TSLA"
    assert "NVDA" in CANDIDATE_SYMBOLS


def test_split_edges_four_fields():
    raw = _bullish_raw()
    edges = compute_edges(
        raw,
        catalysts_today=[],
        qqq_pct=0.4,
        smh_pct=1.8,
        spy_pct=0.2,
        sym_pcts={"TSLA": 2.5, "NVDA": 1.4},
        driver_type="Momentum",
    )
    assert "macro_edge" in edges
    assert "index_edge" in edges
    assert "sector_edge" in edges
    assert "stock_edge" in edges
    assert edges["macro_edge"]["edge"] == "NO"
    assert edges["sector_edge"]["edge"] == "YES"
    p13 = format_p13_from_edges(edges)
    assert "Macro Edge" in p13["body_md"]
    assert "Stock Edge" in p13["body_md"]


def test_zero_dte_instrument():
    p9 = {"buy_options": "Yes", "zero_dte": "Yes", "buy_call": "Yes", "buy_put": "No"}
    assert _instrument("QQQ", "LONG", p9) == "QQQ 0DTE Call"


def test_p16_no_trade_gate():
    parts = {"P16": {"judgment": "计划：不交易", "one_liner": "", "body_md": ""}}
    assert _p16_gate(parts, total=3) == "No Trade"


def test_tsla_ranks_high_with_prior_momentum_and_gap():
    """TSLA +6% prior day can rank #1 when today's gap is tradeable."""
    row = _score_candidate_v2(
        "TSLA",
        obs={"last": 255.0, "prev_close": 240.0, "gap_pct": 2.5},
        prior_day_chg=6.0,
        rs_vs_qqq=2.1,
        gap_pct=2.5,
        news_count=2,
        volume_ok=True,
        vix_chg=-6.0,
        driver_type="Momentum",
        has_news_catalyst=True,
        q={"high": 258, "low": 238},
    )
    assert row["expected_return_pct"] >= 3.0
    assert row["win_prob"] >= 55
    assert row["trade_action"] in ("BUY", "Small")
    assert row["final_score"] > 1.0


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
def test_bullish_momentum_picks_primary(_prior, _obs):
    prior_raw = {
        "stocks": {
            "quotes": {
                "TSLA": {"change_pct": 6.0, "close": 240.0},
            }
        }
    }
    _prior.return_value = prior_raw

    raw = _bullish_raw()
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
        "catalysts_today": [],
        "edges": compute_edges(
            raw,
            catalysts_today=[],
            qqq_pct=0.4,
            smh_pct=1.8,
            spy_pct=0.2,
            sym_pcts={"TSLA": 2.5},
            driver_type="Momentum",
        ),
    }
    parts = {
        "P9": {
            "buy_options": "Yes",
            "zero_dte": "Yes",
            "buy_call": "Yes",
            "buy_put": "No",
        },
        "P11": {"scores": {"VIX": 1, "Bond": 0}},
        "P13": format_p13_from_edges(rule_bundle["edges"]),
        "P16": {"judgment": "计划 Trade", "one_liner": "买 TSLA"},
    }
    result = compute_trade_decision(
        raw,
        rule_bundle=rule_bundle,
        parts=parts,
        regime_label="AI Expansion",
        regime_confidence=0.75,
        edges=rule_bundle["edges"],
    )
    best = result["best_opportunity"]
    primary = result["best_trades"]["primary"]
    ranked = result["trade_candidates"]

    assert ranked[0]["symbol"] == "TSLA"
    assert primary is not None
    assert primary["symbol"] == "TSLA"
    assert best["direction"] == "LONG"
    assert best["confidence"] >= 55
    assert len(ranked) == 6


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
def test_macro_no_index_no_still_trades_on_stock_edge(_prior, _obs):
    _prior.return_value = {
        "stocks": {"quotes": {"TSLA": {"change_pct": 6.0, "close": 240.0}}},
    }
    raw = _bullish_raw()
    edges = compute_edges(
        raw,
        catalysts_today=[],
        qqq_pct=0.1,
        smh_pct=0.2,
        spy_pct=0.1,
        sym_pcts={"TSLA": 2.5, "NVDA": 1.4},
        driver_type="Momentum",
    )
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
        "catalysts_today": [],
        "edges": edges,
    }
    parts = {
        "P9": {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"},
        "P11": {"scores": {"VIX": 1, "Bond": 0}},
        "P13": format_p13_from_edges(edges),
        "P16": {"judgment": "Trade", "one_liner": "个股"},
    }
    result = compute_trade_decision(
        raw,
        rule_bundle=rule_bundle,
        parts=parts,
        edges=edges,
    )
    assert result["best_trades"].get("primary") is not None or result["best_opportunity"]["direction"] == "LONG"


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
def test_p16_no_trade_still_picks_stock_on_stock_edge(_prior, _obs):
    """P16 No Trade gates index only — stock primary still populated when stock_edge YES."""
    _prior.return_value = {
        "stocks": {"quotes": {"TSLA": {"change_pct": 6.0, "close": 240.0}}},
    }
    raw = _bullish_raw()
    edges = compute_edges(
        raw,
        catalysts_today=[],
        qqq_pct=0.1,
        smh_pct=0.2,
        spy_pct=0.1,
        sym_pcts={"TSLA": 2.5, "NVDA": 1.4},
        driver_type="Momentum",
    )
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
        "catalysts_today": [],
        "edges": edges,
    }
    parts = {
        "P9": {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"},
        "P11": {"scores": {"VIX": 1, "Bond": 0}},
        "P13": format_p13_from_edges(edges),
        "P16": {"judgment": "计划：不交易", "one_liner": "指数观望", "body_md": ""},
    }
    result = compute_trade_decision(
        raw,
        rule_bundle=rule_bundle,
        parts=parts,
        edges=edges,
    )
    primary = result["best_trades"]["primary"]
    assert primary is not None
    assert primary["symbol"] == "TSLA"
    assert result["best_trades"]["index_trade"] == "NO TRADE"
    assert result["best_opportunity"]["direction"] == "LONG"
    assert result["best_opportunity"]["symbol"] == "TSLA"
    assert "TSLA" in result["best_opportunity"]["one_liner"]


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
def test_low_scores_threshold_message(_prior, _obs):
    _prior.return_value = {}
    raw = _bullish_raw()
    for section in ("market", "sector", "stocks"):
        for sym, q in ((raw.get(section) or {}).get("quotes") or {}).items():
            q["change_pct"] = 0.05
    raw["news"] = {"polygon": [], "rss": []}
    rule_bundle = {
        "bias": "Neutral",
        "total": 0,
        "driver_type": "No Catalyst",
        "daily_driver": "No dominant catalyst",
        "catalysts_today": [],
    }
    parts = {
        "P9": {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"},
        "P11": {"scores": {"VIX": 0, "Bond": 0}},
        "P16": {"judgment": "Wait", "one_liner": "观望"},
    }
    result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)
    msg = result["best_trades"].get("threshold_message")
    assert msg or result["best_opportunity"]["direction"] in ("NO TRADE", "LONG")
