from __future__ import annotations

from typing import Any

import feedparser
import httpx

from src.collectors.config import load_symbols
from src.collectors.polygon_client import PolygonClient


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


def collect_news() -> dict[str, Any]:
    """Polygon News is the primary headline source; RSS is supplemental."""
    cfg = load_symbols()
    polygon = PolygonClient()
    polygon_articles: list[dict[str, Any]] = []
    poly_errors: list[str] = []
    seen_titles: set[str] = set()

    per_ticker_limit = int(cfg["news"].get("polygon_per_ticker_limit", 5))
    for ticker in cfg["news"]["polygon_tickers"]:
        try:
            data = polygon.news(ticker, limit=per_ticker_limit)
            for article in data.get("results") or []:
                title = str(article.get("title") or "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                polygon_articles.append(_normalize_polygon_article(article, ticker))
        except Exception as exc:  # noqa: BLE001
            poly_errors.append(f"{ticker}:{exc}")

    try:
        broad_limit = int(cfg["news"].get("polygon_broad_limit", 30))
        broad = polygon.get("/v2/reference/news", {"limit": broad_limit, "order": "desc"})
        for article in broad.get("results") or []:
            title = str(article.get("title") or "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            polygon_articles.append(_normalize_polygon_article(article))
    except Exception as exc:  # noqa: BLE001
        poly_errors.append(f"broad:{exc}")

    rss_articles: list[dict[str, Any]] = []
    rss_errors: list[str] = []
    for feed in cfg["news"].get("rss", []):
        try:
            parsed = _parse_rss(feed["url"])
            for entry in (parsed.entries or [])[:5]:
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

    checklist = {
        "polygon_news": len(polygon_articles) >= 5,
        "bloomberg": any(a["source"] == "Bloomberg" for a in rss_articles),
        "wsj": any(a["source"] == "WSJ" for a in rss_articles),
    }

    return {
        "primary": "polygon",
        "polygon": polygon_articles[:40],
        "rss": rss_articles[:10],
        "checklist": checklist,
        "errors": poly_errors + rss_errors,
        "ok": checklist["polygon_news"],
    }
