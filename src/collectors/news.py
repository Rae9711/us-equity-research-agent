from __future__ import annotations

from datetime import datetime
from typing import Any

import feedparser
import httpx

from src.collectors.config import load_symbols
from src.collectors.polygon_client import PolygonClient
from src.utils.trading_calendar import ET, market_open_et, prior_close_utc_iso, today_et


def _parse_rss(url: str) -> feedparser.FeedParserDict:
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={"User-Agent": "DailyTradingOS/1.0 (+https://rae-trading.com)"},
            )
            response.raise_for_status()
            return feedparser.parse(response.text)
    except Exception:
        return feedparser.parse(url)


def _normalize_polygon_article(article: dict[str, Any], ticker: str | None = None) -> dict[str, Any]:
    insights = article.get("insights") or []
    sentiment = insights[0].get("sentiment") if insights else None
    return {
        "source": "polygon",
        "publisher": (article.get("publisher") or {}).get("name"),
        "ticker": ticker,
        "title": article.get("title"),
        "description": article.get("description"),
        "published_utc": article.get("published_utc"),
        "url": article.get("article_url"),
        "keywords": article.get("keywords") or [],
        "sentiment": sentiment,
    }


def _collect_rss(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rss_articles: list[dict[str, Any]] = []
    rss_errors: list[str] = []
    per_feed = int(cfg.get("rss_per_feed_limit", 8))
    for feed in cfg.get("rss", []):
        try:
            parsed = _parse_rss(feed["url"])
            for entry in (parsed.entries or [])[:per_feed]:
                rss_articles.append(
                    {
                        "source": feed["name"],
                        "title": entry.get("title"),
                        "published": entry.get("published"),
                        "link": entry.get("link"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            rss_errors.append(f"{feed['name']}:{exc}")
    return rss_articles, rss_errors


def _collect_polygon(
    cfg: dict[str, Any],
    *,
    published_gte: str | None = None,
    storage_limit: int | None = None,
    broad_first: bool = False,
    pause_s: float = 0.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    import time

    polygon = PolygonClient()
    polygon_articles: list[dict[str, Any]] = []
    poly_errors: list[str] = []
    seen_titles: set[str] = set()
    storage_cap = storage_limit or int(cfg.get("polygon_storage_limit", 60))

    def _ingest(article: dict[str, Any], ticker: str | None = None) -> None:
        title = str(article.get("title") or "")
        if not title or title in seen_titles:
            return
        seen_titles.add(title)
        polygon_articles.append(_normalize_polygon_article(article, ticker))

    def _fetch_broad() -> None:
        try:
            broad_limit = int(cfg.get("polygon_broad_limit", 40))
            broad_params: dict[str, Any] = {"limit": broad_limit, "order": "desc"}
            if published_gte:
                broad_params["published_utc.gte"] = published_gte
            broad = polygon.get("/v2/reference/news", broad_params)
            for article in broad.get("results") or []:
                _ingest(article)
        except Exception as exc:  # noqa: BLE001
            poly_errors.append(f"broad:{exc}")

    def _fetch_tickers() -> None:
        per_ticker_limit = int(cfg.get("polygon_per_ticker_limit", 8))
        for i, ticker in enumerate(cfg.get("polygon_tickers", [])):
            if pause_s and i:
                time.sleep(pause_s)
            try:
                data = polygon.news(ticker, limit=per_ticker_limit, published_gte=published_gte)
                for article in data.get("results") or []:
                    _ingest(article, ticker)
            except Exception as exc:  # noqa: BLE001
                poly_errors.append(f"{ticker}:{exc}")

    if broad_first:
        _fetch_broad()
        if pause_s:
            time.sleep(pause_s)
        _fetch_tickers()
    else:
        _fetch_tickers()
        _fetch_broad()

    return polygon_articles[:storage_cap], poly_errors


def collect_news(*, published_gte: str | None = None) -> dict[str, Any]:
    """Polygon News is the primary headline source; RSS is supplemental."""
    cfg = load_symbols()["news"]
    if published_gte is None:
        published_gte = prior_close_utc_iso()

    polygon_articles, poly_errors = _collect_polygon(cfg, published_gte=published_gte)
    rss_articles, rss_errors = _collect_rss(cfg)
    rss_cap = int(cfg.get("rss_storage_limit", 15))

    has_polygon = len(polygon_articles) >= 5
    has_rss_fallback = len(rss_articles) >= 5 and bool(poly_errors)
    # bloomberg/wsj are supplemental RSS — tracked for UI but do not gate ok/data_ready.
    checklist = {
        "polygon_news": has_polygon or has_rss_fallback,
        "bloomberg": any(a["source"] == "Bloomberg" for a in rss_articles),
        "wsj": any(a["source"] == "WSJ" for a in rss_articles),
    }

    return {
        "primary": "polygon",
        "published_gte": published_gte,
        "polygon": polygon_articles,
        "rss": rss_articles[:rss_cap],
        "checklist": checklist,
        "errors": poly_errors + rss_errors,
        "ok": checklist["polygon_news"],
    }


def collect_intraday_news(since: datetime | None = None) -> dict[str, Any]:
    """Headlines since cash session open (default 9:30 AM ET today)."""
    cfg = load_symbols()["news"]
    since_dt = since or market_open_et(today_et())
    if since_dt.tzinfo is None:
        since_dt = ET.localize(since_dt)
    from src.utils.trading_calendar import UTC

    published_gte = since_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Broad first: post-open window is sparse; avoid burning rate limit on tickers
    # before the market-wide query (free-tier Polygon often 429s after ~5 calls).
    intraday_cap = min(int(cfg.get("polygon_storage_limit", 60)), 40)
    polygon_articles, poly_errors = _collect_polygon(
        cfg,
        published_gte=published_gte,
        storage_limit=intraday_cap,
        broad_first=True,
        pause_s=0.15,
    )

    return {
        "since": since_dt.isoformat(),
        "published_gte": published_gte,
        "polygon": polygon_articles,
        "count": len(polygon_articles),
        "errors": poly_errors,
        "ok": len(polygon_articles) > 0,
    }
