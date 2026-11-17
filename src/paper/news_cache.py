"""Background news cache for the fail-safe news gate (2026-11-17).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Problem: Polygon's intraday news endpoint 429s frequently on the free
tier. The old gate read Step 3 / Step 0 *error lists* and blocked new
entries for any symbol whose fetch had failed — on 2026-11-16 that
blocked ALL intraday entries for the whole day (28 SKIP decisions).

Fix: decouple collection from gating.

* Collection runs in the background (Step 0 morning collection, Step 3
  10:00 intraday collection, and a best-effort refresh from the paper
  tick) and merges into one cache file,
  ``state/paper/news_cache.json``, stamped with a freshness timestamp
  per ticker (plus a broad/market-wide slot).
* A failed fetch NEVER evicts the last good articles for that ticker:
  the old articles stay, their ``fetched_at`` ages out naturally, and
  the failure is recorded as ``last_error``.
* The entry gate (``src/paper/news_gate.py``) reads ONLY this cache.
  Missing or stale news is neutral — it never blocks an entry. Only a
  *fresh negative* headline (see ``news_gate``) can veto.

Cache schema (version 1)::

    {
      "version": 1,
      "updated_at": "<ISO ET, last merge>",
      "last_refresh_error": "<str | None>",
      "broad": {"status": "ok|failed|unknown", "fetched_at": iso|None,
                "articles": [...], "last_error": str|None},
      "tickers": {"AAPL": {"status": ..., "fetched_at": ...,
                            "articles": [...], "last_error": ...}, ...},
      "errors": ["<last bundle error strings, capped>"],
    }

An article is ``{"title", "ticker"|None, "published_utc", "sentiment",
"url"}`` — the same shape Step 3 stores in ``headlines_since_open``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Any

from src.utils.paths import news_cache_path
from src.utils.trading_calendar import ET, market_open_et

logger = logging.getLogger(__name__)

NEWS_CACHE_VERSION = 1

# A cached article / fetch counts as fresh for this long. After it, the
# gate treats the symbol as "no fresh news" (neutral), never as blocked.
NEWS_FRESHNESS_MINUTES = 120

# Minimum gap between tick-triggered background refreshes. Step 0 / Step 3
# collections always merge (they already paid the API calls); the tick
# only tops the cache up when it is older than this.
NEWS_CACHE_REFRESH_MIN_MINUTES = 30

MAX_ARTICLES_PER_TICKER = 20
MAX_ERROR_STRINGS = 8


def _now_et(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(ET)
    if now.tzinfo is None:
        return ET.localize(now)
    return now.astimezone(ET)


def _empty_cache() -> dict[str, Any]:
    return {
        "version": NEWS_CACHE_VERSION,
        "updated_at": None,
        "last_refresh_error": None,
        "broad": {"status": "unknown", "fetched_at": None, "articles": [], "last_error": None},
        "tickers": {},
        "errors": [],
    }


def _normalize_article(article: dict[str, Any]) -> dict[str, Any] | None:
    title = str(article.get("title") or "").strip()
    if not title:
        return None
    ticker = article.get("ticker")
    return {
        "title": title,
        "ticker": (str(ticker).upper() if ticker else None),
        "published_utc": article.get("published_utc"),
        "sentiment": (str(article.get("sentiment") or "").lower() or None),
        "url": article.get("url"),
    }


def _parse_error_tickers(errors: list[Any]) -> tuple[set[str], bool]:
    """Return (failed_ticker_prefixes, broad_failed) from a bundle's errors.

    Error strings look like ``"AAPL:Client error '429 ...'"`` (per-ticker)
    or ``"broad:..."`` (market-wide query).
    """
    failed: set[str] = set()
    broad_failed = False
    for entry in errors or []:
        text = str(entry or "")
        if not text:
            continue
        prefix = text.split(":", 1)[0].strip()
        if prefix == "broad":
            broad_failed = True
        elif prefix:
            failed.add(prefix.upper())
    return failed, broad_failed


def load_news_cache() -> dict[str, Any]:
    """Read the cache from disk; {} when missing or corrupt (fail-open)."""
    path = news_cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("news cache unreadable: %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def save_news_cache(cache: dict[str, Any]) -> None:
    """Atomic write of the cache (tmp file + os.replace)."""
    path = news_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def merge_news_bundle(
    cache: dict[str, Any] | None,
    bundle: dict[str, Any] | None,
    now: datetime | None = None,
    *,
    universe: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Merge one collection bundle into the cache (pure apart from time).

    * A ticker whose fetch succeeded (no error entry for it) gets its
      articles replaced — even with an empty list (a clean fetch with no
      fresh headlines is real information).
    * A ticker whose fetch failed keeps its previous articles untouched;
      only ``status`` / ``last_error`` change, so the data ages out
      instead of disappearing.
    * Broad (untagged) articles merge the same way.
    * ``bundle`` may be None (collection raised) — then nothing is
      replaced and the error is recorded.
    """
    now_et = _now_et(now)
    now_s = now_et.isoformat()
    out: dict[str, Any] = dict(cache or _empty_cache())
    out["version"] = NEWS_CACHE_VERSION
    out["updated_at"] = now_s
    broad = dict(out.get("broad") or _empty_cache()["broad"])
    tickers = dict(out.get("tickers") or {})

    if bundle is None:
        out["last_refresh_error"] = "collection raised (no bundle)"
        out["broad"] = broad
        out["tickers"] = tickers
        return out

    articles = [
        a for a in (_normalize_article(a) for a in (bundle.get("polygon") or []))
        if a is not None
    ]
    errors = [str(e) for e in (bundle.get("errors") or [])][:MAX_ERROR_STRINGS]
    failed_tickers, broad_failed = _parse_error_tickers(errors)

    if universe is None:
        from src.paper.news_gate import news_universe

        universe = news_universe()

    by_ticker: dict[str, list[dict[str, Any]]] = {}
    broad_articles: list[dict[str, Any]] = []
    for a in articles:
        t = a.get("ticker")
        if t and t in universe:
            by_ticker.setdefault(t, []).append(a)
        elif not t:
            broad_articles.append(a)

    if broad_failed:
        broad["status"] = "failed"
        broad["last_error"] = next(
            (e for e in errors if e.startswith("broad:")), "broad fetch failed"
        )
    else:
        broad["status"] = "ok"
        broad["fetched_at"] = now_s
        broad["articles"] = broad_articles[:MAX_ARTICLES_PER_TICKER]
        broad["last_error"] = None
    out["broad"] = broad

    for ticker in sorted(universe):
        prev = dict(tickers.get(ticker) or {"status": "unknown", "fetched_at": None, "articles": [], "last_error": None})
        if ticker in failed_tickers:
            prev["status"] = "failed"
            prev["last_error"] = next(
                (e for e in errors if e.split(":", 1)[0].strip().upper() == ticker),
                f"{ticker} fetch failed",
            )
            # Articles + fetched_at deliberately untouched: last good data stays.
        else:
            prev["status"] = "ok"
            prev["fetched_at"] = now_s
            prev["articles"] = by_ticker.get(ticker, [])[:MAX_ARTICLES_PER_TICKER]
            prev["last_error"] = None
        tickers[ticker] = prev
    out["tickers"] = tickers
    out["errors"] = errors
    out["last_refresh_error"] = None
    return out


def refresh_news_cache(
    now: datetime | None = None,
    *,
    collect_fn: Any = None,
) -> dict[str, Any]:
    """Run one background collection and merge it into the cache.

    Never raises: any failure leaves the previous cache in place (with
    the error recorded) and returns it. ``collect_fn`` injects the
    collector (tests); default is ``collect_intraday_news`` since market
    open.
    """
    now_et = _now_et(now)
    cache = load_news_cache()
    try:
        if collect_fn is None:
            from src.collectors.news import collect_intraday_news

            def _default_collect(since: Any) -> dict[str, Any]:
                return collect_intraday_news(since=since)

            collect_fn = _default_collect
        bundle = collect_fn(market_open_et(now_et.date()))
    except Exception as exc:  # noqa: BLE001 - refresh must never break the tick
        logger.warning("news cache refresh failed: %s", exc)
        cache = dict(cache or _empty_cache())
        cache["last_refresh_error"] = str(exc)[:300]
        cache["updated_at"] = now_et.isoformat()
        try:
            save_news_cache(cache)
        except Exception:
            logger.debug("news cache save failed", exc_info=True)
        return cache
    merged = merge_news_bundle(cache, bundle, now_et)
    try:
        save_news_cache(merged)
    except Exception:
        logger.debug("news cache save failed", exc_info=True)
    return merged


def refresh_news_cache_if_stale(
    now: datetime | None = None,
    *,
    min_interval_minutes: int = NEWS_CACHE_REFRESH_MIN_MINUTES,
    collect_fn: Any = None,
) -> tuple[dict[str, Any], bool]:
    """Refresh only when the cache is missing or older than the interval.

    Returns ``(cache, refreshed)``. Deliberately does NOT refresh when no
    cache file exists yet: the tick must not pay for a first collection
    (Step 0 / Step 3 seed the cache); with no cache the gate is neutral
    anyway. Never raises.
    """
    now_et = _now_et(now)
    path = news_cache_path()
    if not path.exists():
        return {}, False
    cache = load_news_cache()
    updated_at = cache.get("updated_at")
    if updated_at:
        try:
            last = datetime.fromisoformat(str(updated_at))
            if last.tzinfo is None:
                last = ET.localize(last)
            if (now_et - last.astimezone(ET)) < timedelta(minutes=min_interval_minutes):
                return cache, False
        except ValueError:
            pass
    return refresh_news_cache(now_et, collect_fn=collect_fn), True


def cache_freshness(cache: dict[str, Any] | None, now: datetime | None = None) -> dict[str, Any]:
    """Report which tickers have fresh cached news (observability only)."""
    now_et = _now_et(now)
    cache = cache or {}
    tickers = cache.get("tickers") or {}
    fresh: list[str] = []
    stale: list[str] = []
    for sym, slot in tickers.items():
        fetched = slot.get("fetched_at")
        ok = False
        if fetched:
            try:
                dt = datetime.fromisoformat(str(fetched))
                if dt.tzinfo is None:
                    dt = ET.localize(dt)
                ok = (now_et - dt.astimezone(ET)) <= timedelta(minutes=NEWS_FRESHNESS_MINUTES)
            except ValueError:
                ok = False
        (fresh if ok else stale).append(str(sym))
    return {
        "updated_at": cache.get("updated_at"),
        "fresh_tickers": sorted(fresh),
        "stale_tickers": sorted(stale),
    }
