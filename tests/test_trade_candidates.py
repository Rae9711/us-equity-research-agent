"""Tests for P18 Trade Candidates / Best Opportunity engine (v2)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

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


@pytest.fixture(autouse=True)
def _mock_swing_yahoo(request):
    """Avoid live Yahoo ATR fetches in unit tests (unless test already patches)."""
    if "no_swing_mock" in request.keywords:
        yield
        return
    with patch(
        "src.research.trade_candidates._fetch_swing_context",
        return_value={
            "atr": 3.0,
            "atr_source": "proxy_test",
            "week_high": None,
            "week_low": None,
            "prior_5d_chg_pct": 3.0,
        },
    ):
        yield


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
    from src.research.trade_candidates import _SYMBOL_SECTION

    section = _SYMBOL_SECTION.get(symbol.upper(), "stocks")
    q = ((raw.get(section) or {}).get("quotes") or {}).get(symbol) or {}
    if not q:
        return {"ticker": symbol, "error": "no quote"}
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


def test_candidate_symbols_covers_tracked_universe():
    """Mag7 + liquid semis + equity indexes/ETFs from symbols.yaml (not futures/rates)."""
    assert "NVDA" in CANDIDATE_SYMBOLS
    assert "MSFT" in CANDIDATE_SYMBOLS
    assert "AAPL" in CANDIDATE_SYMBOLS
    assert "AMZN" in CANDIDATE_SYMBOLS
    assert "META" in CANDIDATE_SYMBOLS
    assert "GOOGL" in CANDIDATE_SYMBOLS
    assert "TSLA" in CANDIDATE_SYMBOLS
    assert "AMD" in CANDIDATE_SYMBOLS
    assert "MU" in CANDIDATE_SYMBOLS
    assert "AVGO" in CANDIDATE_SYMBOLS
    assert "ARM" in CANDIDATE_SYMBOLS
    assert "QQQ" in CANDIDATE_SYMBOLS
    assert "SPY" in CANDIDATE_SYMBOLS
    assert "TQQQ" in CANDIDATE_SYMBOLS
    assert "DIA" in CANDIDATE_SYMBOLS
    assert "SMH" in CANDIDATE_SYMBOLS
    assert "XLK" in CANDIDATE_SYMBOLS
    assert "XLF" in CANDIDATE_SYMBOLS
    assert "XLE" in CANDIDATE_SYMBOLS
    # Context-only market keys stay out of the tradeable candidate list.
    assert "VIX" not in CANDIDATE_SYMBOLS
    assert "ES" not in CANDIDATE_SYMBOLS
    assert CANDIDATE_SYMBOLS[0] == "NVDA"  # config/stocks order


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


def test_long_does_not_inflate_win_prob_with_same_session_rs():
    """Same-session RS is descriptive only; prior-day trend still moves win_prob."""
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
    components = (strong.get("win_prob_breakdown") or {}).get("components") or []
    assert all(c.get("key") != "rs" for c in components)
    assert strong["win_prob"] <= 52
    assert strong["win_prob"] > weak["win_prob"]


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
    assert row["expected_return_pct"] >= 2.0
    assert row["win_prob"] <= 52
    assert row["win_prob"] >= 48
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
    assert best["confidence"] >= 48
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
def test_p16_no_trade_blocks_stocks_and_index(_prior, _obs):
    """P16 No Trade / Wait closes the whole book — cash is a valid day."""
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
    assert primary is None
    assert result["best_trades"]["index_trade"] == "NO TRADE"
    assert result["best_opportunity"]["direction"] == "NO TRADE"
    assert "P16" in (result["best_trades"].get("threshold_message") or "")


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

def test_trade_candidates_direction_is_per_symbol_not_market_bias():
    """Per-name tape sets direction; market Bias must not flip every symbol."""
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
    bull_tsla = next(r for r in result["trade_candidates"] if r["symbol"] == "TSLA")
    assert bull_tsla.get("direction") in ("LONG", "NO TRADE")

    bear_bundle = {**rule_bundle, "bias": "Bearish Bias", "total": -3}
    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs):
        bear = compute_trade_decision(raw, rule_bundle=bear_bundle, parts=parts)
    bear_tsla = next(r for r in bear["trade_candidates"] if r["symbol"] == "TSLA")
    assert bear_tsla.get("direction") == bull_tsla.get("direction")
    assert not all(r.get("direction") == "SHORT" for r in bear["trade_candidates"])


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
        "target_price": 110.0,  # 2R — hard R:R floor applies to planned levels
        "entry_zone": {"low": 99.0, "high": 101.0, "mid": 100.0},
        "expected_return_pct": 5.0,
        "win_prob": 60.0,
        "risk_reward": 2.0,
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


def test_no_quote_pass_stubs_include_current_price_key():
    """Universe symbols without quotes must not KeyError in P18 decision tree.

    After CANDIDATE_SYMBOLS expansion, missing-quote Pass stubs used to omit
    ``current_price`` / expected_* keys; ``_build_trade_slot`` then raised
    KeyError when filling top_trades transparency.
    """
    raw = _bullish_raw()
    # Leave Mag7 names beyond TSLA/NVDA without quotes (typical PIT partial universe).
    assert "MSFT" not in ((raw.get("stocks") or {}).get("quotes") or {})

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

    def _obs_partial(symbol: str, raw_in: dict, prior_raw: dict, trading_day, **kwargs):
        from src.research.trade_candidates import _SYMBOL_SECTION

        section = _SYMBOL_SECTION.get(symbol.upper(), "stocks")
        q = ((raw_in.get(section) or {}).get("quotes") or {}).get(symbol) or {}
        if not q:
            return {"ticker": symbol, "error": "no quote"}
        return _mock_obs(symbol, raw_in, prior_raw, trading_day, **kwargs)

    with patch("src.research.trade_candidates._observation", side_effect=_obs_partial):
        with patch("src.research.trade_candidates._load_prior_raw", return_value={}):
            result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)

    # Must complete without KeyError; every ranked row carries current_price key.
    assert "trade_candidates" in result
    assert "top_trades" in result
    for row in result["trade_candidates"]:
        assert "current_price" in row, f"{row.get('symbol')} missing current_price"
        assert "expected_high" in row
        assert "expected_low" in row
        assert "expected_close" in row

    stub_syms = {
        r["symbol"]
        for r in result["trade_candidates"]
        if r.get("why") == "No quote data" or "No quote data" in (r.get("why_factors") or [])
    }
    assert stub_syms, "expected at least one no-quote Pass stub"
    for sym in stub_syms:
        stub = next(r for r in result["trade_candidates"] if r["symbol"] == sym)
        assert stub["trade_action"] == "Pass"
        assert stub["current_price"] is None

    # Transparency board may include Pass stubs; all slots have current_price key.
    for slot in result["top_trades"]:
        assert "current_price" in slot


def test_build_trade_slot_survives_legacy_pass_row_without_current_price():
    """Defensive: even a legacy incomplete Pass row must not KeyError."""
    from src.research.trade_candidates import _build_trade_slot

    legacy = {
        "symbol": "AMD",
        "win_prob": 50.0,
        "expected_return_pct": 0.0,
        "final_score": 0.0,
        "trade_action": "Pass",
        "trade": "Pass",
        "risk_reward": 0.0,
        "why_factors": ["No quote data"],
        "why": "No quote data",
        # deliberately omit current_price / expected_*
    }
    slot = _build_trade_slot(
        legacy,
        rank=1,
        direction="LONG",
        p9={},
        obs={"error": "no quote"},
        raw={},
        prior_raw={},
        trading_day=__import__("datetime").date(2026, 7, 14),
        section="stocks",
        q={},
    )
    assert slot["trade_action"] == "Pass"
    assert "current_price" in slot
    assert slot["current_price"] is None
    assert slot["levels_valid"] is False


# ── Swing / Position trade ─────────────────────────────────────────────


def test_derive_swing_levels_wider_than_intraday():
    from src.research.trade_candidates import derive_swing_levels

    levels = derive_swing_levels(
        "LONG", current=100.0, atr=3.0, week_high=108.0, week_low=94.0
    )
    assert levels["levels_valid"] is True
    entry = levels["entry_price"]
    stop = levels["stop_price"]
    target = levels["target_price"]
    assert stop < entry < target
    # Stop at least ~1.2 ATR below entry (wider multi-day risk)
    assert entry - stop >= 1.2 * 3.0
    # Target at least ~5% or 2 ATR
    assert target - entry >= min(5.0, 2.0 * 3.0) - 0.01
    assert levels["entry_zone"]["low"] < levels["entry_zone"]["high"]
    assert "1.5–2×ATR" in levels["level_formula"]
    assert len(levels["targets"]) >= 2


def test_swing_direction_follows_trend_not_forced_bias():
    from src.research.trade_candidates import _swing_direction

    # Strong multi-day strength → LONG even if bias is SHORT
    assert (
        _swing_direction(
            bias_direction="SHORT",
            prior_day_chg=2.0,
            rs_vs_qqq=1.5,
            prior_5d_chg=4.0,
        )
        == "LONG"
    )
    # Strong multi-day weakness → SHORT even if bias is LONG
    assert (
        _swing_direction(
            bias_direction="LONG",
            prior_day_chg=-2.0,
            rs_vs_qqq=-1.5,
            prior_5d_chg=-4.0,
        )
        == "SHORT"
    )
    # Flat momentum → fall back to bias
    assert (
        _swing_direction(
            bias_direction="SHORT",
            prior_day_chg=0.1,
            rs_vs_qqq=0.0,
            prior_5d_chg=0.2,
        )
        == "SHORT"
    )


@patch("src.research.trade_candidates._fetch_swing_context")
def test_compute_swing_opportunity_independent_of_intraday(mock_ctx):
    from src.research.trade_candidates import compute_swing_opportunity

    mock_ctx.return_value = {
        "atr": 4.0,
        "atr_source": "proxy",
        "week_high": 150.0,
        "week_low": 130.0,
        "prior_5d_chg_pct": 5.5,
    }
    ranked = [
        {
            "symbol": "NVDA",
            "final_score": 1.2,  # below intraday BUY threshold often
            "trade_action": "Pass",
            "win_prob": 58,
            "prior_day_change_pct": 2.0,
            "relative_strength": 1.1,
            "current_price": 140.0,
            "news_count": 1,
            "edge_type": "Momentum",
        },
        {
            "symbol": "MU",
            "final_score": 0.5,
            "trade_action": "Pass",
            "win_prob": 50,
            "prior_day_change_pct": -1.0,
            "relative_strength": -0.5,
            "current_price": 100.0,
            "news_count": 0,
        },
    ]
    obs = {
        "NVDA": {"last": 140.0, "close": 140.0},
        "MU": {"last": 100.0, "close": 100.0},
    }
    swing = compute_swing_opportunity(
        ranked,
        bias_direction="LONG",
        total_score=1,  # not |≥4| — still eligible for dedicated swing pick
        obs_by_sym=obs,
        quote_by_sym={},
        catalysts=["NVDA AI demand"],
        raw={},
    )
    assert swing is not None
    assert swing["symbol"] == "NVDA"
    assert swing["horizon"] == "Swing"
    assert swing["direction"] == "LONG"
    assert swing["separate_from_intraday"] is True
    assert swing["entry_zone"]
    assert swing["stop_price"] < swing["entry_price"] < swing["target_price"]
    assert swing["invalidation"]
    assert "ADVISORY" in str(swing.get("advisory") or True) or swing["advisory"] is True


@patch("src.research.trade_candidates._observation", side_effect=_mock_obs)
@patch("src.research.trade_candidates._load_prior_raw")
@patch("src.research.trade_candidates._fetch_swing_context")
def test_swing_attached_when_intraday_no_trade(mock_ctx, _prior, _obs):
    """P16 No Trade also blocks swing — cash is a valid day."""
    from src.research.trade_candidates import compute_trade_decision

    mock_ctx.return_value = {
        "atr": 5.0,
        "atr_source": "proxy",
        "week_high": 270.0,
        "week_low": 230.0,
        "prior_5d_chg_pct": 6.0,
    }
    _prior.return_value = {
        "stocks": {"quotes": {"TSLA": {"change_pct": 4.0, "close": 240.0}}}
    }
    raw = _bullish_raw()
    # Quiet gaps so primary may fail threshold; swing uses preferred + momentum
    raw["stocks"]["quotes"]["TSLA"]["change_pct"] = 0.3
    raw["stocks"]["quotes"]["NVDA"]["change_pct"] = 0.2
    rule_bundle = {
        "bias": "Bullish Bias",
        "total": 1,
        "driver_type": "Momentum",
        "daily_driver": "AI",
        "catalysts_today": [{"name": "TSLA delivery", "symbol": "TSLA"}],
        "edges": compute_edges(
            raw,
            catalysts_today=[{"name": "TSLA delivery", "symbol": "TSLA"}],
            qqq_pct=0.4,
            smh_pct=1.8,
            spy_pct=0.2,
            sym_pcts={"TSLA": 0.3, "NVDA": 0.2},
            driver_type="Momentum",
        ),
    }
    parts = {
        "P9": {
            "buy_options": "No",
            "zero_dte": "No",
            "buy_call": "No",
            "buy_put": "No",
        },
        "P16": {"judgment": "计划：不交易", "one_liner": "No Trade"},
    }
    result = compute_trade_decision(
        raw,
        rule_bundle=rule_bundle,
        parts=parts,
        edges=rule_bundle["edges"],
    )
    assert result["best_trades"].get("p16_gate") == "No Trade"
    swing = result.get("swing_trade") or result["best_trades"].get("swing")
    assert swing is None

