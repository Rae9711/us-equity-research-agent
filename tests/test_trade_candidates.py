"""Tests for P18 Trade Candidates / Best Opportunity engine (v2)."""

from __future__ import annotations

from unittest.mock import patch

from src.research.edges import compute_edges, format_p13_from_edges
from src.research.trade_candidates import (
    CANDIDATE_SYMBOLS,
    FINAL_SCORE_THRESHOLD,
    _apply_price_based_return,
    _enforce_level_invariants,
    _expected_return_from_prices,
    _geo_context_penalty,
    _instrument,
    _p16_gate,
    _return_calculation_string,
    _risk_reward_from_levels,
    _score_candidate_v2,
    _slot_is_actionable,
    _trade_horizon,
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


def _mock_obs(symbol: str, raw: dict, prior_raw: dict, trading_day, **kwargs):  # noqa: ARG001
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


def test_candidate_symbols_includes_semis():
    assert CANDIDATE_SYMBOLS[0] == "TSLA"
    assert "NVDA" in CANDIDATE_SYMBOLS
    assert "AMD" in CANDIDATE_SYMBOLS
    assert "MU" in CANDIDATE_SYMBOLS
    assert "AVGO" in CANDIDATE_SYMBOLS
    assert "META" in CANDIDATE_SYMBOLS
    assert "ARM" in CANDIDATE_SYMBOLS


def test_short_weak_rs_beats_strong_nvda():
    """Bearish sector: weakest semi (MU) should outrank NVDA leader for SHORT."""
    mu = _score_candidate_v2(
        "MU",
        obs={"last": 100.0, "prev_close": 102.0, "gap_pct": -1.0},
        prior_day_chg=-2.5,
        rs_vs_qqq=-1.8,
        rs_vs_smh=-1.2,
        gap_pct=-1.0,
        news_count=1,
        volume_ok=True,
        vix_chg=6.0,
        driver_type="Macro",
        has_news_catalyst=True,
        q={"high": 103, "low": 98},
        direction="SHORT",
    )
    nvda = _score_candidate_v2(
        "NVDA",
        obs={"last": 140.0, "prev_close": 139.0, "gap_pct": 0.5},
        prior_day_chg=1.5,
        rs_vs_qqq=0.8,
        rs_vs_smh=1.0,
        gap_pct=0.5,
        news_count=2,
        volume_ok=True,
        vix_chg=6.0,
        driver_type="AI",
        has_news_catalyst=True,
        q={"high": 142, "low": 138},
        direction="SHORT",
    )
    assert mu["final_score"] > nvda["final_score"]
    assert mu["win_prob"] > nvda["win_prob"]
    assert (mu.get("relative_weakness_score") or 0) > (nvda.get("relative_weakness_score") or 0)


def test_long_still_rewards_positive_rs():
    """LONG direction should still favor relative strength."""
    strong = _score_candidate_v2(
        "NVDA",
        obs={"last": 140.0, "prev_close": 138.0, "gap_pct": 1.0},
        prior_day_chg=3.0,
        rs_vs_qqq=1.2,
        rs_vs_smh=0.8,
        gap_pct=1.0,
        news_count=1,
        volume_ok=True,
        vix_chg=-4.0,
        driver_type="AI",
        has_news_catalyst=True,
        q={"high": 142, "low": 137},
        direction="LONG",
    )
    weak = _score_candidate_v2(
        "MU",
        obs={"last": 100.0, "prev_close": 102.0, "gap_pct": -1.0},
        prior_day_chg=-2.0,
        rs_vs_qqq=-1.5,
        rs_vs_smh=-1.0,
        gap_pct=-1.0,
        news_count=0,
        volume_ok=True,
        vix_chg=-4.0,
        driver_type="AI",
        has_news_catalyst=False,
        q={"high": 103, "low": 98},
        direction="LONG",
    )
    assert strong["final_score"] > weak["final_score"]


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


def test_stock_short_not_labeled_0dte_put():
    """P9 buy_options=No → stock SHORT is Stock, horizon Intraday (not 0DTE Put)."""
    p9 = {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"}
    assert _instrument("MU", "SHORT", p9) == "Stock"
    assert _trade_horizon(direction="SHORT", p9=p9, total=-1, instrument="Stock") == "Intraday"


def test_horizon_0dte_only_when_p9_authorizes():
    p9_yes = {"buy_options": "Yes", "zero_dte": "Yes", "buy_call": "Yes", "buy_put": "No"}
    assert _trade_horizon(
        direction="LONG", p9=p9_yes, total=2, instrument="QQQ 0DTE Call"
    ) == "0DTE"
    p9_swing = {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"}
    assert _trade_horizon(direction="LONG", p9=p9_swing, total=5, instrument="Stock") == "Swing"
    # zero_dte Yes but instrument is Stock (options not used) → not 0DTE
    p9_mismatch = {"buy_options": "No", "zero_dte": "Yes", "buy_call": "No", "buy_put": "No"}
    assert (
        _trade_horizon(direction="SHORT", p9=p9_mismatch, total=-2, instrument="Stock")
        == "Intraday"
    )


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
    assert len(ranked) == len(CANDIDATE_SYMBOLS)
    transparency = result.get("transparency") or {}
    assert transparency.get("todays_opportunities")
    assert primary.get("win_prob_breakdown")
    assert primary.get("level_reasons")
    assert primary.get("trade_economics")
    assert primary.get("why_not_alternatives")


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


def test_short_expected_return_from_entry_target():
    """SHORT TSLA entry 412 target 402.7 → ER ≈ 2.26%."""
    er = _expected_return_from_prices("SHORT", 412.0, 402.7)
    assert er is not None
    assert abs(er - 2.26) < 0.01
    calc = _return_calculation_string("SHORT", 412.0, 402.7, er)
    assert calc == "(412.0-402.7)/412.0=2.26%"


def test_long_expected_return_from_entry_target():
    """LONG entry 100 target 103 → ER 3%."""
    er = _expected_return_from_prices("LONG", 100.0, 103.0)
    assert er == 3.0
    calc = _return_calculation_string("LONG", 100.0, 103.0, er)
    assert calc == "(103.0-100.0)/100.0=3.00%"


def test_apply_price_based_return_overrides_heuristic():
    slot = {
        "symbol": "TSLA",
        "entry_price": 412.0,
        "target_price": 402.7,
        "stop_price": 419.6,
        "expected_return_pct": 2.67,
        "expected_move": "+2.67%",
        "risk_reward": 1.5,
    }
    out = _apply_price_based_return(slot, direction="SHORT", heuristic_er=2.67)
    assert out["expected_return_pct"] == 2.26
    assert out["expected_move"] == "+2.26%"
    assert out["return_calculation"] == "(412.0-402.7)/412.0=2.26%"
    assert out["trade_summary_cn"] == "做空 TSLA：在 412.0 附近入场，目标 402.7，预期 +2.26%"
    rr = _risk_reward_from_levels(412.0, 419.6, 2.26)
    assert out["risk_reward"] == rr
    assert out.get("rr_display")
    assert out["rr_display"]["reward_risk_ratio"] == rr


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
def test_primary_trade_er_matches_entry_target(_prior, _obs):
    """Primary slot ER must match (entry-target)/entry, not momentum heuristic."""
    _prior.return_value = {
        "stocks": {"quotes": {"TSLA": {"change_pct": -3.0, "close": 420.0}}},
    }
    raw = _bullish_raw()
    raw["trading_date"] = "2026-07-07"
    raw["stocks"]["quotes"]["TSLA"] = {
        "close": 410.0,
        "prev_close": 418.0,
        "open": 412.0,
        "high": 420.0,
        "low": 400.0,
        "change_pct": -1.9,
        "volume": 5000,
        "session_type": "premarket",
    }
    edges = compute_edges(
        raw,
        catalysts_today=[],
        qqq_pct=-0.5,
        smh_pct=-1.0,
        spy_pct=-0.3,
        sym_pcts={"TSLA": -1.9},
        driver_type="Macro",
    )
    rule_bundle = {
        "bias": "Bearish Bias",
        "total": -3,
        "driver_type": "Macro",
        "daily_driver": "Risk-off",
        "catalysts_today": [],
        "edges": edges,
    }
    parts = {
        "P9": {"buy_options": "No", "zero_dte": "No", "buy_call": "No", "buy_put": "No"},
        "P11": {"scores": {"VIX": 2, "Bond": 1}},
        "P13": format_p13_from_edges(edges),
        "P16": {"judgment": "Trade SHORT TSLA", "one_liner": "做空 TSLA"},
    }
    result = compute_trade_decision(
        raw,
        rule_bundle=rule_bundle,
        parts=parts,
        edges=edges,
    )
    primary = result["best_trades"].get("primary")
    best = result["best_opportunity"]
    if primary and primary.get("entry_price") and primary.get("target_price"):
        entry = primary["entry_price"]
        target = primary["target_price"]
        expected = _expected_return_from_prices(primary["direction"], entry, target)
        assert primary["expected_return_pct"] == expected
        assert best["expected_return_pct"] == expected
        assert primary["return_calculation"]
        assert "trade_summary_cn" in primary


def test_geo_penalty_on_nvda_short():
    penalty, reason = _geo_context_penalty(
        "NVDA",
        "SHORT",
        macro_calendar={
            "catalysts": [
                {"name": "Geopolitical Risk (Iran)", "category": "breaking", "theme": "geopolitics"}
            ]
        },
        driver_tree={"primary": {"type": "Political", "label": "Geopolitical Risk (Iran)"}},
    )
    assert penalty >= 15
    assert reason is not None


def test_compute_trade_decision_top5_and_watchlist():
    raw = _bullish_raw()
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
        "catalysts_today": [],
        "macro_calendar": {"catalysts": [], "has_material_catalyst": False},
        "driver_tree": {"primary": {"type": "Momentum", "label": "AI Momentum"}},
    }
    parts = {
        "P9": {"buy_options": "Yes", "zero_dte": "No", "buy_call": "Yes", "buy_put": "No"},
        "P16": {"judgment": "Wait", "trade_plan": {"gate": "Wait", "conditions_wait": ["wait"]}},
    }
    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs):
        result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)
    assert "top_trades" in result
    assert len(result["top_trades"]) <= 5
    assert "transparency" in result
    assert "top_trades" in result["transparency"]

def test_trade_candidates_direction_from_bias():
    """Every ranked trade_candidates row carries bias-picked LONG/SHORT."""
    raw = _bullish_raw()
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
        "catalysts_today": [],
        "macro_calendar": {"catalysts": [], "has_material_catalyst": False},
        "driver_tree": {"primary": {"type": "Momentum", "label": "AI Momentum"}},
    }
    parts = {
        "P9": {"buy_options": "Yes", "zero_dte": "No", "buy_call": "Yes", "buy_put": "No"},
        "P16": {"judgment": "Wait", "trade_plan": {"gate": "Wait", "conditions_wait": ["wait"]}},
    }
    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs):
        result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)
    for row in result["trade_candidates"]:
        assert row.get("direction") == "LONG"
    for slot in result["top_trades"]:
        assert slot.get("direction") == "LONG"

    bear_bundle = {**rule_bundle, "bias": "Bearish Bias", "total": -3}
    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs):
        bear = compute_trade_decision(raw, rule_bundle=bear_bundle, parts=parts)
    for row in bear["trade_candidates"]:
        assert row.get("direction") == "SHORT"


def test_enforce_rejects_long_with_target_below_entry():
    """ARM-like LONG slot with entry > target must become Pass."""
    slot = {
        "symbol": "ARM",
        "direction": "LONG",
        "trade_action": "BUY",
        "trade": "BUY",
        "entry_price": 341.0,
        "stop_price": 322.22,
        "target_price": 333.69,
        "entry_zone": {"low": 340.0, "high": 342.0, "mid": 341.0},
        "expected_return_pct": -2.01,
        "why_factors": ["RS weak vs QQQ -2.42%"],
        "levels_valid": False,
    }
    out = _enforce_level_invariants(slot)
    assert out["trade_action"] == "Pass"
    assert out["levels_valid"] is False
    assert not _slot_is_actionable(out)


def test_enforce_rejects_short_with_target_above_entry():
    slot = {
        "symbol": "TSLA",
        "direction": "SHORT",
        "trade_action": "Small",
        "trade": "Small",
        "entry_price": 400.0,
        "stop_price": 410.0,
        "target_price": 420.0,  # above entry — invalid short
        "entry_zone": {"low": 398.0, "high": 402.0, "mid": 400.0},
        "expected_return_pct": -5.0,
        "why_factors": [],
        "levels_valid": False,
    }
    out = _enforce_level_invariants(slot)
    assert out["trade_action"] == "Pass"
    assert not _slot_is_actionable(out)


def test_enforce_keeps_valid_long():
    slot = {
        "symbol": "NVDA",
        "direction": "LONG",
        "trade_action": "BUY",
        "trade": "BUY",
        "entry_price": 100.0,
        "stop_price": 95.0,
        "target_price": 105.0,
        "entry_zone": {"low": 99.0, "high": 101.0, "mid": 100.0},
        "expected_return_pct": 5.0,
        "why_factors": [],
        "levels_valid": True,
    }
    out = _enforce_level_invariants(slot)
    assert out["trade_action"] == "BUY"
    assert _slot_is_actionable(out)


def test_actionable_top_trades_never_violate_geometry():
    """Integration: every actionable top_trade satisfies direction geometry + ER≥0."""
    raw = _bullish_raw()
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 3,
        "driver_type": "Momentum",
        "daily_driver": "AI Momentum",
        "catalysts_today": [],
        "macro_calendar": {"catalysts": [], "has_material_catalyst": False},
        "driver_tree": {"primary": {"type": "Momentum", "label": "AI Momentum"}},
    }
    parts = {
        "P9": {"buy_options": "Yes", "zero_dte": "No", "buy_call": "Yes", "buy_put": "No"},
        "P16": {"judgment": "计划 Trade", "one_liner": "做多"},
    }
    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs):
        result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)

    for slot in result["top_trades"]:
        if not _slot_is_actionable(slot):
            continue
        entry = slot["entry_price"]
        stop = slot["stop_price"]
        target = slot["target_price"]
        assert entry is not None and stop is not None and target is not None
        if slot["direction"] == "LONG":
            assert stop < entry <= target
            assert slot["expected_return_pct"] >= 0
        elif slot["direction"] == "SHORT":
            assert target <= entry < stop
            assert slot["expected_return_pct"] >= 0

    primary = result["best_trades"].get("primary")
    if primary and _slot_is_actionable(primary):
        assert primary["entry_price"] <= primary["target_price"] or primary["direction"] == "SHORT"
        if primary["direction"] == "LONG":
            assert primary["target_price"] >= primary["entry_price"]
            assert primary["expected_return_pct"] >= 0

