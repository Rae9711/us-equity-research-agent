"""Tests for per-symbol research (2026-11-17).

Before: one market direction stamped on all 11 candidates (77 direction
flips synchronized across symbols on the same days). After: each symbol
is scored LONG and SHORT on its own evidence and keeps the better
direction; candidates rank by calibrated expectancy.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.research.context import build_research_context
from src.research.edges import compute_edges, format_p13_from_edges
from src.research.trade_candidates import (
    _calibrated_rank_key,
    compute_trade_decision,
)
from tests.test_trade_candidates import _bullish_raw, _mock_obs


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _weak_mu_raw():
    """Bullish market, but MU itself is weak: down 3.9% today, prior -2.5%."""
    raw = _bullish_raw()
    raw["stocks"]["quotes"]["MU"] = {
        "close": 98.0, "prev_close": 102.0, "open": 101.0, "change_pct": -3.9,
        "volume": 3000, "session_type": "premarket", "high": 101.5, "low": 97.0,
    }
    raw["news"]["polygon"].append({"ticker": "MU", "title": "MU guidance cut"})
    return raw


def _mock_obs_with_mu(symbol, raw, prior_raw, trading_day, **kwargs):
    if symbol == "MU":
        return {
            "ticker": "MU", "last": 98.0, "close": 98.0, "open": 101.0,
            "prev_close": 102.0, "change_pct": -3.9, "gap_pct": -0.98,
        }
    return _mock_obs(symbol, raw, prior_raw, trading_day, **kwargs)


def _run(raw, *, bias="Bullish Bias", total=3, data_root=None):
    rule_bundle = {
        "bias": bias, "total": total, "driver_type": "Momentum",
        "daily_driver": "AI Momentum", "catalysts_today": [],
        "macro_calendar": {"catalysts": [], "has_material_catalyst": False},
        "driver_tree": {"primary": {"type": "Momentum", "label": "AI Momentum"}},
        "edges": compute_edges(raw, catalysts_today=[], qqq_pct=0.4, smh_pct=1.8,
                               spy_pct=0.2, sym_pcts={"TSLA": 2.5, "MU": -3.9},
                               driver_type="Momentum"),
    }
    parts = {
        "P9": {"buy_options": "Yes", "zero_dte": "No", "buy_call": "Yes", "buy_put": "No"},
        "P11": {"scores": {"VIX": 1, "Bond": 0}},
        "P13": format_p13_from_edges(rule_bundle["edges"]),
        "P16": {"judgment": "计划 Trade", "one_liner": "买 TSLA"},
    }
    prior = {"stocks": {"quotes": {
        "TSLA": {"change_pct": 6.0, "close": 240.0},
        "MU": {"change_pct": -2.5, "close": 102.0},
    }}}
    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs_with_mu), \
         patch("src.research.trade_candidates._load_prior_raw", return_value=prior), \
         patch("src.research.trade_candidates._fetch_swing_context",
               return_value={"atr": 3.0, "atr_source": "proxy_test", "week_high": None, "week_low": None, "prior_5d_chg_pct": 3.0}):
        return compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)


def test_per_symbol_directions_differ_same_day(data_root):
    """Same bullish day: strong TSLA is LONG, weak MU is SHORT."""
    result = _run(_weak_mu_raw(), data_root=data_root)
    assert result["market_direction"] == "LONG"
    by_sym = {r["symbol"]: r for r in result["trade_candidates"]}
    tsla, mu = by_sym["TSLA"], by_sym["MU"]
    assert tsla["direction"] == "LONG"
    assert tsla["direction_source"] == "per-symbol"
    assert mu["direction"] == "SHORT"
    assert mu["direction_source"] == "per-symbol"
    assert mu["win_prob"] != tsla["win_prob"]
    # The SHORT side was actually scored, not inherited.
    assert mu["alt_direction"] == "LONG"
    assert mu["alt_direction_final_score"] is not None
    # Slots follow their own row's direction.
    top_dirs = {(t["symbol"], t["direction"]) for t in result["best_trades"]["top_trades"]}
    assert ("MU", "SHORT") in top_dirs or mu["trade_action"] == "Pass"


def test_per_symbol_direction_tiebreak_is_market_direction():
    """A symbol with no directional edge follows the market bias."""
    from src.research.trade_candidates import _pick_per_symbol_direction

    long_row = {"final_score": 5.0, "win_prob": 60.0, "expected_return_pct": 1.0}
    short_row = {"final_score": 5.0, "win_prob": 60.0, "expected_return_pct": 1.0}
    assert _pick_per_symbol_direction(long_row, short_row, market_direction="LONG")["direction"] == "LONG"
    assert _pick_per_symbol_direction(long_row, short_row, market_direction="SHORT")["direction"] == "SHORT"
    better_short = {"final_score": 6.0, "win_prob": 65.0, "expected_return_pct": 1.2}
    out = _pick_per_symbol_direction(long_row, better_short, market_direction="LONG")
    assert out["direction"] == "SHORT"
    assert out["alt_direction_final_score"] == 5.0


def test_ranking_by_calibrated_expectancy(data_root):
    """Known-positive expectancy outranks higher final_score; known
    negative ranks below no-data rows; no map at all == old order."""
    # No map: order is pure final_score.
    rows = [
        {"symbol": "A", "final_score": 9.0, "trade_action": "Small", "calibrated_expectancy_r": None},
        {"symbol": "B", "final_score": 5.0, "trade_action": "Small", "calibrated_expectancy_r": None},
    ]
    assert [r["symbol"] for r in sorted(rows, key=_calibrated_rank_key)] == ["A", "B"]
    # With a map: B's +0.30R beats A's +0.10R despite a lower final_score.
    # Pass rows sink below every tradeable row whatever their expectancy.
    rows = [
        {"symbol": "A", "final_score": 9.0, "trade_action": "Small", "calibrated_expectancy_r": 0.10},
        {"symbol": "B", "final_score": 5.0, "trade_action": "Small", "calibrated_expectancy_r": 0.30},
        {"symbol": "C", "final_score": 7.0, "trade_action": "Small", "calibrated_expectancy_r": None},
        {"symbol": "D", "final_score": 8.0, "trade_action": "Small", "calibrated_expectancy_r": -0.20},
        {"symbol": "E", "final_score": 10.0, "trade_action": "Pass", "calibrated_expectancy_r": 0.50},
    ]
    assert [r["symbol"] for r in sorted(rows, key=_calibrated_rank_key)] == ["B", "A", "C", "D", "E"]


def test_ranking_uses_calibrated_expectancy_end_to_end(data_root):
    """With a map on disk, compute_trade_decision ranks by calibrated
    expectancy, not final_score alone."""
    paper = data_root / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "shadow_summary.json").write_text(json.dumps({
        "recalibration": {
            "<50": {"n": 13, "wins": 3, "calibrated_win_prob": 0.319},
            "50-60": {"n": 3, "wins": 2, "calibrated_win_prob": 0.572},
            "60-70": {"n": 5, "wins": 3, "calibrated_win_prob": 0.627},
            ">70": {"n": 13, "wins": 5, "calibrated_win_prob": 0.530},
        },
        "recalibration_by_horizon": {},
    }), encoding="utf-8")
    result = _run(_weak_mu_raw(), data_root=data_root)
    rows = [r for r in result["trade_candidates"] if r.get("direction_source") == "per-symbol"]
    assert rows, "expected per-symbol rows"
    assert all("calibrated_expectancy_r" in r for r in rows)
    # The ranked list is sorted by the calibrated key: no tier-2 row
    # (known negative) appears above a tier-0/1 row.
    keys = [_calibrated_rank_key(r) for r in result["trade_candidates"]]
    assert keys == sorted(keys)


def test_context_has_per_symbol_block():
    raw = _weak_mu_raw()
    ctx = build_research_context(raw)
    assert "per_symbol" in ctx
    by_sym = {row["symbol"]: row for row in ctx["per_symbol"]}
    assert by_sym["MU"]["change_pct"] == pytest.approx(-3.9)
    assert any("MU guidance cut" in t for t in by_sym["MU"]["headlines"])
    assert by_sym["TSLA"]["change_pct"] == pytest.approx(2.5)
    # Existing keys untouched.
    assert "mag7" in ctx and "headlines" in ctx
