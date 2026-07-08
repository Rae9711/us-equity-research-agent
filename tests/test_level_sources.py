"""Tests for auditable level sources and why_vs_runner_up."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from src.research.level_sources import (
    LevelAnchors,
    compute_anchors,
    derive_trade_levels,
    format_if_level,
    format_tagged,
    source_label,
)
from src.research.trade_candidates import _why_vs_runner_up, compute_trade_decision


def test_source_labels():
    assert source_label("prev_low") == "昨日低点"
    assert source_label("vwap") == "VWAP"


def test_format_tagged_level():
    assert "255.0" in format_tagged(255.0, "vwap", prefix="Above")
    assert "VWAP" in format_tagged(255.0, "vwap")


def test_format_if_level_p16():
    assert format_if_level(711.0, "prev_low") == "711.0 (昨日低点)"


def test_derive_trade_levels_long():
    anchors = LevelAnchors(
        vwap=254.0,
        orb_high=256.0,
        orb_low=248.0,
        prev_high=258.0,
        prev_low=238.0,
        prior_close=240.0,
        orb_from_minute=True,
    )
    levels = derive_trade_levels(
        "LONG",
        anchors,
        current=255.0,
        expected_high=262.0,
        expected_low=246.0,
        expected_close=260.0,
    )
    assert levels["entry_source"] == "vwap"
    assert levels["stop_source"] == "orb_low"
    assert levels["target_source"] in ("expected_close", "expected_high")
    assert "(" in levels["entry"]


@patch("src.research.level_sources._ohlcv_bars_yfinance", return_value=[])
def test_compute_anchors_fallback_prev_day(_yf):
    raw = {
        "stocks": {
            "quotes": {
                "TSLA": {"close": 255.0, "prev_close": 240.0, "high": 258, "low": 238},
            }
        }
    }
    prior_raw = {
        "stocks": {
            "quotes": {
                "TSLA": {"close": 240.0, "high": 258, "low": 238},
            }
        }
    }
    obs = {"last": 255.0, "prev_close": 240.0}
    anchors = compute_anchors(
        "TSLA", raw, prior_raw, date(2026, 7, 7), section="stocks", q=raw["stocks"]["quotes"]["TSLA"], obs=obs
    )
    assert anchors.prev_low == 238.0
    assert anchors.orb_high == 258.0
    assert anchors.orb_from_minute is False


def test_why_vs_runner_up():
    top = {
        "symbol": "TSLA",
        "relative_strength": -1.2,
        "expected_return_pct": 3.5,
        "win_prob": 68,
        "final_score": 4.2,
        "factor_breakdown": {"beta_proxy": 2.1},
    }
    runner = {
        "symbol": "NVDA",
        "relative_strength": -0.3,
        "expected_return_pct": 2.1,
        "win_prob": 62,
        "final_score": 2.8,
        "factor_breakdown": {"beta_proxy": 1.4},
    }
    text = _why_vs_runner_up(top, runner)
    assert "TSLA" in text
    assert "NVDA" in text
    assert "RS" in text


def _mock_obs(symbol, raw, prior_raw, trading_day):  # noqa: ARG001
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


@patch("src.research.trade_candidates.compute_anchors")
@patch("src.research.trade_candidates.derive_trade_levels")
@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
def test_ranked_candidate_has_auditable_fields(_prior, _obs, _levels, _anchors):
    _prior.return_value = {
        "stocks": {"quotes": {"TSLA": {"change_pct": 6.0, "close": 240.0, "high": 258, "low": 238}}},
    }
    _anchors.return_value = LevelAnchors(prev_high=258, prev_low=238, prior_close=240)
    _levels.return_value = {
        "entry": "Above 255.0 (VWAP)",
        "entry_source": "vwap",
        "stop": "248.0 (ORB低点)",
        "stop_source": "orb_low",
        "target": "260.0 (预期收盘)",
        "target_source": "expected_close",
        "targets": ["260.0 (预期收盘)"],
        "level_anchors": {},
    }

    from tests.test_trade_candidates import _bullish_raw
    from src.research.edges import compute_edges, format_p13_from_edges

    raw = _bullish_raw()
    edges = compute_edges(
        raw,
        catalysts_today=[],
        qqq_pct=0.4,
        smh_pct=1.8,
        spy_pct=0.2,
        sym_pcts={"TSLA": 2.5},
        driver_type="Momentum",
    )
    result = compute_trade_decision(
        raw,
        rule_bundle={
            "bias": "Bullish Bias",
            "total": 3,
            "driver_type": "Momentum",
            "daily_driver": "AI Momentum",
            "catalysts_today": [],
            "edges": edges,
        },
        parts={
            "P9": {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"},
            "P16": {"judgment": "计划 Trade"},
            "P13": format_p13_from_edges(edges),
        },
        edges=edges,
    )
    top = result["trade_candidates"][0]
    assert top["symbol"] == "TSLA"
    assert "factor_breakdown" in top
    assert "edge_type" in top
    assert "score_formula_display" in top
    assert top.get("why_vs_runner_up") != "—"

    primary = result["best_trades"]["primary"]
    assert primary["entry_source"] == "vwap"
    assert primary["stop_source"] == "orb_low"
