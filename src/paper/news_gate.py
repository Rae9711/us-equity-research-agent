"""News gate for new entries — fail-safe edition (2026-11-17).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

History: the original gate (fix 6) read Step 3 / Step 0 *error lists*
and blocked new entries for any symbol whose Polygon news fetch had
failed. On 2026-11-16 Polygon 429'd all day, so the gate blocked ALL
intraday entries (28 SKIP decisions) — a data outage became a trading
halt.

New rule: the gate reads ONLY the background news cache
(``state/paper/news_cache.json``, see ``src/paper/news_cache.py``).

* Missing news is NEUTRAL: no cache, or a symbol whose cached news is
  stale / never fetched, never blocks an entry.
* A fetch failure (429) is NEUTRAL too: the cache keeps the last good
  articles and the gate does not care that the refresh failed.
* Only *fresh negative* news can veto: a cached headline published
  within ``NEWS_FRESHNESS_MINUTES`` that (a) matches a high-signal
  keyword (the same L2 detection Step 3 uses) AND (b) carries Polygon
  sentiment ``negative``. Ticker-tagged headlines veto that symbol;
  untagged (market-wide) headlines veto everything.

Position management (stop / target exits) is unaffected and keeps
running, as before.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from src.paper.news_cache import NEWS_FRESHNESS_MINUTES, load_news_cache
from src.utils.trading_calendar import ET

logger = logging.getLogger(__name__)

# Fallback universe when config/symbols.yaml cannot be read.
FALLBACK_NEWS_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    # Keep in sync with config/symbols.yaml polygon_tickers (extended
    # 2026-11-12 with the traded names AMD/MU/AVGO/ARM; extended 2026-11-17
    # with JPM/XOM/UNH/COST/HD for more trades/day).
    "AMD",
    "MU",
    "AVGO",
    "ARM",
    "JPM",
    "XOM",
    "UNH",
    "COST",
    "HD",
)

WHOLE_COLLECTION_MARKER = "collect_intraday_news raised"


def news_universe() -> frozenset[str]:
    """Tickers whose per-ticker Polygon news fetch is tracked."""
    try:
        from src.collectors.config import load_symbols

        tickers = (load_symbols().get("news") or {}).get("polygon_tickers") or []
        out = {str(t).upper() for t in tickers if str(t).strip()}
        if out:
            return frozenset(out)
    except Exception:
        logger.debug("news universe config load failed", exc_info=True)
    return frozenset(FALLBACK_NEWS_TICKERS)


def parse_news_error_entries(
    errors: list[Any] | None,
    *,
    universe: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Parse a news-collection error list into a gate status.

    Kept for the cache merge (``news_cache.merge_news_bundle``) and for
    diagnostics. The entry gate itself no longer blocks on these errors —
    see the module docstring.

    Returns ``blocked_symbols`` (set), ``global_block`` (bool),
    ``broad_failed`` (bool), ``whole_collection_failed`` (bool) and a
    human-readable ``reasons`` list. ``global_block`` is True only when the
    whole collection failed, or the broad market-wide query failed *and*
    every universe ticker's fetch failed too.
    """
    universe = universe if universe is not None else news_universe()
    blocked: set[str] = set()
    broad_failed = False
    whole_failed = False
    for entry in errors or []:
        text = str(entry or "")
        if not text:
            continue
        if text.startswith("collect_intraday_news"):
            whole_failed = True
            continue
        prefix = text.split(":", 1)[0].strip()
        if prefix == "broad":
            broad_failed = True
        elif prefix.upper() in universe:
            blocked.add(prefix.upper())
    global_block = bool(whole_failed) or bool(
        broad_failed and universe and universe <= blocked
    )
    reasons: list[str] = []
    if global_block:
        reasons.append("全局新闻数据缺失（Polygon 新闻接口失败）")
    for sym in sorted(blocked):
        reasons.append(f"{sym} 新闻数据缺失（Polygon 429/失败）")
    return {
        "blocked_symbols": blocked,
        "global_block": global_block,
        "broad_failed": broad_failed,
        "whole_collection_failed": whole_failed,
        "reasons": reasons,
    }


def _parse_published(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(ET)


def _fresh_articles(cache: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    """Cached articles published within the freshness window."""
    window = timedelta(minutes=NEWS_FRESHNESS_MINUTES)
    out: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    broad = cache.get("broad") or {}
    if isinstance(broad, dict):
        slots.append(broad)
    tickers = cache.get("tickers") or {}
    if isinstance(tickers, dict):
        slots.extend(v for v in tickers.values() if isinstance(v, dict))
    for slot in slots:
        for article in slot.get("articles") or []:
            if not isinstance(article, dict):
                continue
            published = _parse_published(article.get("published_utc"))
            if published is None:
                continue  # cannot verify freshness -> not a veto
            if now - published <= window and now - published >= timedelta(0):
                out.append(article)
    return out


def _veto_headlines(
    fresh: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fresh negative high-signal headlines, with their veto scope.

    One entry per headline: ``symbol`` (None = market-wide), ``title``,
    ``published_utc``, ``sentiment``, ``keyword``, ``severity``.
    A headline vetoes only when it BOTH matches a high-signal keyword
    (same L2 detection Step 3 uses) AND carries Polygon sentiment
    ``negative`` — a merely negative headline, or a keyword match with
    neutral/positive sentiment, does not veto.
    """
    from src.utils.news_signals import detect_high_signal_news

    vetoes: list[dict[str, Any]] = []
    for article in fresh:
        if str(article.get("sentiment") or "").lower() != "negative":
            continue
        matches = detect_high_signal_news([article])
        if not matches:
            continue
        top = matches[0]
        vetoes.append(
            {
                "symbol": article.get("ticker"),
                "title": article.get("title"),
                "published_utc": article.get("published_utc"),
                "sentiment": article.get("sentiment"),
                "keyword": top.get("keyword"),
                "severity": top.get("severity"),
                "theme": top.get("theme"),
            }
        )
    return vetoes


def news_gate_for_date(
    trading_date: str,
    *,
    step3: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """News gate status for ``trading_date`` — reads the cache ONLY.

    ``step3`` / ``raw`` are accepted for backward compatibility and are
    deliberately IGNORED: since 2026-11-17 the gate no longer reads
    collection error lists (a 429 must never block an entry again).
    ``cache`` / ``now`` inject the cache payload and the tick time
    (tests); otherwise the cache is loaded from disk and ``now`` is now.

    Returns ``blocked_symbols`` (sorted list), ``global_block``,
    ``source`` (``"cache"`` | ``"cache-missing"`` | ``"error"``),
    ``reasons``, ``veto_headlines`` and cache-freshness observability
    (``cache_updated_at``, ``stale_tickers``).
    """
    del trading_date, step3, raw  # cache-only since 2026-11-17
    now_et = now.astimezone(ET) if now is not None else datetime.now(ET)
    if now_et.tzinfo is None:
        now_et = ET.localize(now_et)
    try:
        loaded = cache if cache is not None else load_news_cache()
        if not loaded:
            return {
                "blocked_symbols": [],
                "global_block": False,
                "broad_failed": False,
                "whole_collection_failed": False,
                "source": "cache-missing",
                "reasons": ["无新闻缓存，按中性处理（不拦开仓）"],
                "veto_headlines": [],
                "cache_updated_at": None,
                "stale_tickers": [],
            }
        fresh = _fresh_articles(loaded, now_et)
        vetoes = _veto_headlines(fresh)
        blocked = sorted({v["symbol"] for v in vetoes if v.get("symbol")})
        global_block = any(v.get("symbol") is None for v in vetoes)
        reasons: list[str] = []
        if global_block:
            titles = "; ".join(
                str(v.get("title") or "")[:60] for v in vetoes if v.get("symbol") is None
            )
            reasons.append(f"突发负面新闻（新鲜），暂停全部新开仓：{titles}")
        for sym in blocked:
            titles = "; ".join(
                str(v.get("title") or "")[:60]
                for v in vetoes
                if v.get("symbol") == sym
            )
            reasons.append(f"{sym} 有新鲜负面新闻，暂停新开仓：{titles}")
        stale: list[str] = []
        tickers = loaded.get("tickers") or {}
        if isinstance(tickers, dict):
            window = timedelta(minutes=NEWS_FRESHNESS_MINUTES)
            for sym, slot in tickers.items():
                if not isinstance(slot, dict):
                    continue
                fetched = slot.get("fetched_at")
                fresh_fetch = False
                if fetched:
                    try:
                        dt = datetime.fromisoformat(str(fetched))
                        if dt.tzinfo is None:
                            dt = ET.localize(dt)
                        fresh_fetch = (now_et - dt.astimezone(ET)) <= window
                    except ValueError:
                        fresh_fetch = False
                if not fresh_fetch:
                    stale.append(str(sym))
        return {
            "blocked_symbols": blocked,
            "global_block": global_block,
            "broad_failed": False,
            "whole_collection_failed": False,
            "source": "cache",
            "reasons": reasons,
            "veto_headlines": vetoes,
            "cache_updated_at": loaded.get("updated_at"),
            "stale_tickers": sorted(stale),
        }
    except Exception:
        # Never let gate plumbing halt trading: fail open, loudly.
        logger.warning(
            "news gate load failed; gate open", exc_info=True
        )
        return {
            "blocked_symbols": [],
            "global_block": False,
            "broad_failed": False,
            "whole_collection_failed": False,
            "source": "error",
            "reasons": [],
            "veto_headlines": [],
            "cache_updated_at": None,
            "stale_tickers": [],
        }


def news_block_reason(gate: dict[str, Any] | None, symbol: str) -> str | None:
    """Reason to block a new entry in ``symbol``, or None when allowed.

    Only a fresh-negative-news veto blocks now. (A gate dict injected
    with legacy ``blocked_symbols`` / ``global_block`` keys — e.g. in
    tests — is still honored, with its own reason text.)
    """
    if not gate:
        return None
    vetoes = gate.get("veto_headlines") or []
    if gate.get("global_block"):
        if vetoes:
            titles = "; ".join(
                str(v.get("title") or "")[:60] for v in vetoes if v.get("symbol") is None
            )
            return f"突发负面新闻（新鲜），暂停全部新开仓：{titles}"
        return "全局新闻数据缺失（Polygon 新闻接口失败），暂停全部新开仓"
    sym = (symbol or "").upper()
    if sym and sym in (gate.get("blocked_symbols") or []):
        mine = [v for v in vetoes if v.get("symbol") == sym]
        if mine:
            titles = "; ".join(str(v.get("title") or "")[:60] for v in mine)
            return f"{sym} 有新鲜负面新闻，暂停新开仓：{titles}"
        return f"{sym} 新闻数据缺失（Polygon 429/失败），暂停新开仓"
    return None
