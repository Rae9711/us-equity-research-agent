"""Tests for the fail-safe news gate (2026-11-17).

The gate reads ONLY the background news cache:
* a 429 / missing fetch never blocks an entry (the 2026-11-16 outage
  blocked all intraday entries for a day — that must not recur);
* only a FRESH headline that BOTH matches a high-signal keyword AND
  carries negative sentiment can veto (per-symbol, or market-wide when
  untagged);
* a failed fetch never evicts the last good articles for a ticker.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.paper.news_cache import (
    NEWS_FRESHNESS_MINUTES,
    load_news_cache,
    merge_news_bundle,
    refresh_news_cache,
    refresh_news_cache_if_stale,
    save_news_cache,
)
from src.paper.news_gate import news_block_reason, news_gate_for_date
from src.utils.trading_calendar import ET

UNIVERSE = frozenset({"AAPL", "MSFT", "NVDA"})


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def _now():
    return datetime.now(ET)


def _iso(dt):
    return dt.astimezone(ET).isoformat()


def _article(title, *, ticker="AAPL", sentiment="negative", minutes_ago=10):
    return {
        "title": title,
        "ticker": ticker,
        "published_utc": _iso(_now() - timedelta(minutes=minutes_ago)),
        "sentiment": sentiment,
        "url": "https://example.com/x",
    }


def _cache_with_articles(articles, *, fetched_minutes_ago=5):
    """A cache dict whose ticker slots hold the given articles."""
    now = _now()
    tickers = {}
    for a in articles:
        t = a.get("ticker")
        if t:
            tickers.setdefault(t, {"status": "ok", "fetched_at": _iso(now - timedelta(minutes=fetched_minutes_ago)), "articles": [], "last_error": None})
            tickers[t]["articles"].append(a)
    return {
        "version": 1,
        "updated_at": _iso(now - timedelta(minutes=fetched_minutes_ago)),
        "last_refresh_error": None,
        "broad": {"status": "ok", "fetched_at": _iso(now), "articles": [a for a in articles if not a.get("ticker")], "last_error": None},
        "tickers": tickers,
        "errors": [],
    }


# --- merge ---------------------------------------------------------------

def test_merge_replaces_successful_ticker():
    bundle = {
        "polygon": [_article("breaking: AAPL guidance cut", ticker="AAPL")],
        "errors": [],
    }
    merged = merge_news_bundle(None, bundle, _now(), universe=UNIVERSE)
    slot = merged["tickers"]["AAPL"]
    assert slot["status"] == "ok"
    assert len(slot["articles"]) == 1
    assert slot["articles"][0]["title"] == "breaking: AAPL guidance cut"
    assert slot["last_error"] is None
    assert merged["tickers"]["MSFT"]["articles"] == []


def test_merge_failed_ticker_keeps_old_articles():
    old = _article("breaking: AAPL guidance cut", ticker="AAPL", minutes_ago=90)
    cache = _cache_with_articles([old], fetched_minutes_ago=90)
    old_fetched = cache["tickers"]["AAPL"]["fetched_at"]
    bundle = {"polygon": [], "errors": ["AAPL:Client error '429 Too Many Requests' for url ..."]}
    merged = merge_news_bundle(cache, bundle, _now(), universe=UNIVERSE)
    slot = merged["tickers"]["AAPL"]
    assert slot["status"] == "failed"
    assert slot["articles"][0]["title"] == "breaking: AAPL guidance cut"
    assert slot["fetched_at"] == old_fetched  # last good fetch time kept
    assert "429" in (slot["last_error"] or "")


def test_merge_broad_failure_keeps_old_broad():
    old = _article("breaking: CPI came in hot", ticker=None, minutes_ago=60)
    cache = _cache_with_articles([old], fetched_minutes_ago=60)
    bundle = {"polygon": [], "errors": ["broad:Client error '429 Too Many Requests' for url ..."]}
    merged = merge_news_bundle(cache, bundle, _now(), universe=UNIVERSE)
    assert merged["broad"]["status"] == "failed"
    assert merged["broad"]["articles"][0]["title"] == "breaking: CPI came in hot"


def test_merge_none_bundle_keeps_everything():
    cache = _cache_with_articles([_article("breaking: AAPL guidance cut")])
    merged = merge_news_bundle(cache, None, _now(), universe=UNIVERSE)
    assert merged["tickers"]["AAPL"]["articles"]
    assert merged["last_refresh_error"]


# --- refresh -------------------------------------------------------------

def test_refresh_if_stale_no_cache_file_no_collect(data_root):
    calls = []
    cache, refreshed = refresh_news_cache_if_stale(_now(), collect_fn=lambda since: calls.append(since) or {"polygon": [], "errors": []})
    assert refreshed is False
    assert calls == []
    assert cache == {}


def test_refresh_if_stale_fresh_cache_no_collect(data_root):
    save_news_cache(_cache_with_articles([_article("breaking: AAPL guidance cut")], fetched_minutes_ago=5))
    calls = []
    cache, refreshed = refresh_news_cache_if_stale(_now(), collect_fn=lambda since: calls.append(since) or {"polygon": [], "errors": []})
    assert refreshed is False
    assert calls == []


def test_refresh_if_stale_old_cache_collects(data_root):
    save_news_cache(_cache_with_articles([], fetched_minutes_ago=90))
    bundle = {"polygon": [_article("breaking: MSFT guidance raised", ticker="MSFT", sentiment="positive")], "errors": []}
    cache, refreshed = refresh_news_cache_if_stale(_now(), collect_fn=lambda since: bundle)
    assert refreshed is True
    assert cache["tickers"]["MSFT"]["articles"][0]["title"] == "breaking: MSFT guidance raised"


def test_refresh_collect_raises_keeps_cache(data_root):
    save_news_cache(_cache_with_articles([_article("breaking: AAPL guidance cut")], fetched_minutes_ago=90))

    def _boom(since):
        raise RuntimeError("polygon down")

    cache = refresh_news_cache(_now(), collect_fn=_boom)
    assert cache["tickers"]["AAPL"]["articles"]
    assert "polygon down" in (cache.get("last_refresh_error") or "")
    assert load_news_cache()["tickers"]["AAPL"]["articles"]


# --- gate: fail-safe -----------------------------------------------------

def test_gate_missing_cache_is_open():
    gate = news_gate_for_date("2026-11-17", cache={})
    assert gate["source"] == "cache-missing"
    assert gate["blocked_symbols"] == []
    assert gate["global_block"] is False
    assert news_block_reason(gate, "AAPL") is None


def test_gate_429_errors_do_not_block():
    """The 2026-11-16 outage: errors with no fresh articles must NOT block."""
    cache = _cache_with_articles([], fetched_minutes_ago=200)
    cache["errors"] = ["AAPL:Client error '429 Too Many Requests' for url ..."]
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["blocked_symbols"] == []
    assert gate["global_block"] is False
    assert news_block_reason(gate, "AAPL") is None


def test_gate_fresh_negative_high_signal_blocks_symbol():
    cache = _cache_with_articles([_article("breaking: AAPL guidance cut", minutes_ago=10)])
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["source"] == "cache"
    assert gate["blocked_symbols"] == ["AAPL"]
    assert gate["global_block"] is False
    reason = news_block_reason(gate, "AAPL")
    assert reason is not None and "新鲜负面新闻" in reason
    assert news_block_reason(gate, "MSFT") is None


def test_gate_stale_negative_headline_no_block():
    old = _article("breaking: AAPL guidance cut", minutes_ago=NEWS_FRESHNESS_MINUTES + 60)
    cache = _cache_with_articles([old], fetched_minutes_ago=NEWS_FRESHNESS_MINUTES + 60)
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["blocked_symbols"] == []
    assert news_block_reason(gate, "AAPL") is None


def test_gate_negative_without_keyword_no_block():
    cache = _cache_with_articles([_article("AAPL shares slip 1% on light volume", minutes_ago=10)])
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["blocked_symbols"] == []
    assert news_block_reason(gate, "AAPL") is None


def test_gate_positive_keyword_no_block():
    cache = _cache_with_articles([
        _article("breaking: AAPL guidance raised", sentiment="positive", minutes_ago=10)
    ])
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["blocked_symbols"] == []
    assert news_block_reason(gate, "AAPL") is None


def test_gate_broad_fresh_negative_blocks_all():
    cache = _cache_with_articles([
        _article("breaking: CPI came in hot", ticker=None, minutes_ago=10)
    ])
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["global_block"] is True
    assert news_block_reason(gate, "AAPL") is not None
    assert news_block_reason(gate, "MSFT") is not None


def test_gate_unparseable_published_never_vetoes():
    article = _article("breaking: AAPL guidance cut", minutes_ago=10)
    article["published_utc"] = "not-a-date"
    cache = _cache_with_articles([article])
    gate = news_gate_for_date("2026-11-17", cache=cache, now=_now())
    assert gate["blocked_symbols"] == []
