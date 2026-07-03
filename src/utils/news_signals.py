"""Lightweight headline theme/sentiment tallies for regime features."""

from __future__ import annotations

from typing import Any

_THEME_KEYWORDS: dict[str, list[str]] = {
    "ai": ["ai", "nvidia", "semiconductor", "chip", "gpu", "openai", "anthropic"],
    "macro": ["fed", "fomc", "cpi", "inflation", "jobs", "payroll", "treasury", "rate", "gdp"],
    "oil": ["oil", "opec", "crude", "energy", "gasoline"],
    "risk": ["selloff", "rally", "volatility", "vix", "risk-off", "risk on"],
}


def _headline_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title") or ""),
        str(item.get("description") or ""),
        " ".join(str(k) for k in (item.get("keywords") or [])),
    ]
    return " ".join(parts).lower()


def extract_news_signals(raw_news: dict[str, Any] | None) -> dict[str, Any]:
    """Count themes and bearish ratio from Step 0 polygon + RSS headlines."""
    if not raw_news:
        return {
            "news_headline_count": 0,
            "news_bearish_ratio": None,
            "news_ai_mentions": 0,
            "news_macro_mentions": 0,
        }

    articles: list[dict[str, Any]] = []
    articles.extend(raw_news.get("polygon") or [])
    for item in raw_news.get("rss") or []:
        articles.append(
            {
                "title": item.get("title"),
                "description": None,
                "keywords": [],
                "sentiment": None,
            }
        )

    theme_counts = {k: 0 for k in _THEME_KEYWORDS}
    bearish = 0
    bullish = 0
    for article in articles:
        text = _headline_text(article)
        if not text.strip():
            continue
        for theme, kws in _THEME_KEYWORDS.items():
            if any(kw in text for kw in kws):
                theme_counts[theme] += 1
        sentiment = str(article.get("sentiment") or "").lower()
        if sentiment == "negative":
            bearish += 1
        elif sentiment == "positive":
            bullish += 1

    labeled = bearish + bullish
    bearish_ratio = (bearish / labeled) if labeled else None

    return {
        "news_headline_count": len(articles),
        "news_bearish_ratio": round(bearish_ratio, 3) if bearish_ratio is not None else None,
        "news_ai_mentions": theme_counts["ai"],
        "news_macro_mentions": theme_counts["macro"],
        "news_oil_mentions": theme_counts["oil"],
        "news_risk_mentions": theme_counts["risk"],
    }
