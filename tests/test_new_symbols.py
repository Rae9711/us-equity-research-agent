"""Tests for the 2026-11-17 universe extension (JPM/XOM/UNH/COST/HD).

Liquid mega-caps in sectors the old 11-name universe did not cover
(financials / energy / healthcare / consumer staples / discretionary),
added to trade more per day. Each new name must be: in the research
universe, classified as a stock, covered by per-ticker news, and able
to complete a full tick entry -> exit flow.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.paper.account import default_account, get_position
from src.paper.execution_agent import decide_and_act
from src.paper.news_gate import FALLBACK_NEWS_TICKERS, news_universe
from src.research.trade_candidates import (
    CANDIDATE_SYMBOLS,
    _STOCK_SYMBOLS,
    _SWING_PREFERRED,
    _SYMBOL_SECTION,
    compute_trade_decision,
)

NEW_SYMBOLS = ["JPM", "XOM", "UNH", "COST", "HD"]


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


@pytest.fixture(autouse=True)
def _mock_swing_yahoo():
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


def test_new_symbols_in_universe_and_news_coverage():
    assert len(CANDIDATE_SYMBOLS) == 16
    for sym in NEW_SYMBOLS:
        assert sym in CANDIDATE_SYMBOLS
        assert sym in _STOCK_SYMBOLS  # instrument = Stock, not ETF
        assert _SYMBOL_SECTION[sym] == "stocks"
        assert sym in _SWING_PREFERRED
        assert sym in FALLBACK_NEWS_TICKERS
    assert set(NEW_SYMBOLS) <= news_universe()  # config/symbols.yaml


def test_new_symbols_scored_by_research():
    """Each new name gets a per-symbol row from compute_trade_decision."""
    raw = {
        "trading_date": "2026-07-06",
        "market": {"quotes": {
            "QQQ": {"close": 520.0, "prev_close": 518.0, "change_pct": 0.4},
            "SPY": {"close": 600.0, "prev_close": 599.0, "change_pct": 0.2},
            "TQQQ": {"close": 80.0, "prev_close": 79.0, "change_pct": 1.2},
            "^VIX": {"close": 14.0, "prev_close": 15.0, "change_pct": -6.0},
        }},
        "sector": {"quotes": {
            "SMH": {"close": 280.0, "prev_close": 275.0, "change_pct": 1.8},
        }},
        "stocks": {"quotes": {
            sym: {
                "close": 100.0, "prev_close": 98.0, "high": 101.0, "low": 97.0,
                "change_pct": 2.0, "session_type": "premarket", "volume": 1000,
            }
            for sym in NEW_SYMBOLS
        }},
        "news": {"polygon": [], "rss": []},
        "options": {"avg_implied_volatility": 0.20},
        "macro": {"economic_calendar": []},
    }
    rule_bundle = {
        "bias": "Bullish Bias", "total": 3, "driver_type": "Momentum",
        "daily_driver": "Momentum", "catalysts_today": [],
        "macro_calendar": {"catalysts": [], "has_material_catalyst": False},
        "driver_tree": {"primary": {"type": "Momentum", "label": "Momentum"}},
    }
    parts = {
        "P9": {"buy_options": "Yes", "zero_dte": "No", "buy_call": "Yes", "buy_put": "No"},
        "P16": {"judgment": "Wait", "trade_plan": {"gate": "Wait", "conditions_wait": ["wait"]}},
    }

    def _mock_obs(symbol, raw_arg, prior_raw, trading_day, **kwargs):
        section = "market" if symbol in ("QQQ", "SPY", "TQQQ") else (
            "sector" if symbol == "SMH" else "stocks"
        )
        q = ((raw_arg.get(section) or {}).get("quotes") or {}).get(symbol) or {}
        if not q:
            return {"ticker": symbol, "error": "no quote"}
        close = float(q.get("close") or 100)
        prev = float(q.get("prev_close") or close * 0.99)
        return {
            "ticker": symbol, "last": close, "close": close,
            "open": float(q.get("open") or prev * 1.01), "prev_close": prev,
            "change_pct": q.get("change_pct"), "gap_pct": 0.0,
        }

    with patch("src.research.trade_candidates._observation", side_effect=_mock_obs):
        result = compute_trade_decision(raw, rule_bundle=rule_bundle, parts=parts)
    by_symbol = {r.get("symbol"): r for r in result["trade_candidates"]}
    for sym in NEW_SYMBOLS:
        row = by_symbol.get(sym)
        assert row is not None, f"{sym} missing from trade_candidates"
        assert row.get("direction") in ("LONG", "SHORT")
        assert row.get("direction_source") == "per-symbol"


def _write_morning(root, trading_date, primary):
    day = root / "reports" / trading_date
    day.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_trades": {"primary": primary, "top_trades": [primary]},
        "best_opportunity": primary,
        "advisory": True,
    }
    (day / "morning.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("symbol", NEW_SYMBOLS)
def test_new_symbol_tick_entry_and_exit(data_root, symbol):
    """Full flow per new symbol: tick entry at 100, target exit at 110."""
    primary = {
        "symbol": symbol,
        "direction": "LONG",
        "entry_price": 100.0,
        "entry_zone": {"low": 99.0, "mid": 100.0, "high": 101.0},
        "stop_price": 95.0,
        "target_price": 110.0,
        "win_prob": 70,
        "expected_return_pct": 5.0,
        "horizon": "Intraday",
    }
    _write_morning(data_root, "2026-07-08", primary)
    acct = default_account()
    entry = decide_and_act(acct, "2026-07-08", session_phase="open", force_price=100.0)
    assert entry["action"] == "ENTRY"
    pos = get_position(acct, "intraday")
    assert pos is not None and pos["symbol"] == symbol
    assert pos["shares"] > 0
    bars = [{"ts": "2026-07-08 10:00", "open": 100.0, "high": 111.0,
             "low": 100.0, "close": 110.0}]
    # Exit tick at 105: above the 99-101 entry zone, so the primary is
    # MISSED and the tick does not re-enter right after the target exit
    # (target exits set no cooldown, by design).
    exit_decision = decide_and_act(
        acct, "2026-07-08", session_phase="open", force_price=105.0,
        bars_by_symbol={symbol: bars},
    )
    assert get_position(acct, "intraday") is None
    assert acct["realized_pnl"] > 0
    assert any(
        (b.get("action") == "EXIT" and (b.get("trade") or {}).get("symbol") == symbol)
        for b in (exit_decision.get("books") or [])
    )
