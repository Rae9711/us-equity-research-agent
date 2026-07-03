"""Lightweight headline theme/sentiment tallies + L2 突发关键词检测。"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_THEME_KEYWORDS: dict[str, list[str]] = {
    "ai": ["ai", "nvidia", "semiconductor", "chip", "gpu", "openai", "anthropic"],
    "macro": ["fed", "fomc", "cpi", "inflation", "jobs", "payroll", "treasury", "rate", "gdp"],
    "oil": ["oil", "opec", "crude", "energy", "gasoline"],
    "risk": ["selloff", "rally", "volatility", "vix", "risk-off", "risk on"],
}

_KEYWORDS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "news_keywords.yaml"
)

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


@lru_cache(maxsize=1)
def _load_keyword_config() -> dict[str, Any]:
    """加载 news_keywords.yaml；缺失时返回空结构。"""
    if not _KEYWORDS_CONFIG_PATH.exists():
        return {"high_signal": {}, "force_high_signal_prefixes": []}
    try:
        return yaml.safe_load(_KEYWORDS_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.exception("news_keywords.yaml parse failed")
        return {"high_signal": {}, "force_high_signal_prefixes": []}


def _compile_patterns() -> list[tuple[str, str, re.Pattern[str]]]:
    """展开成 [(theme, severity, pattern), ...]，忽略大小写。"""
    cfg = _load_keyword_config()
    compiled: list[tuple[str, str, re.Pattern[str]]] = []

    def _add(theme: str, severity: str, patterns: list[str]) -> None:
        for pat in patterns or []:
            try:
                compiled.append((theme, severity, re.compile(pat, re.IGNORECASE)))
            except re.error:
                logger.warning("跳过非法关键词: %s", pat)

    high_signal = cfg.get("high_signal") or {}
    if isinstance(high_signal, dict):
        for theme, block in high_signal.items():
            if isinstance(block, dict):
                _add(theme, str(block.get("severity") or "high"), block.get("patterns") or [])
            elif isinstance(block, list):
                _add(theme, "high", block)

    for extra_key in ("trade", "earnings"):
        block = cfg.get(extra_key)
        if isinstance(block, dict):
            theme = str(block.get("theme") or extra_key)
            _add(theme, str(block.get("severity") or "medium"), block.get("patterns") or [])
        elif isinstance(block, list):
            _add(extra_key, "medium", block)

    return compiled


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


def detect_high_signal_news(headlines: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """L2 · 检测突发/高信号新闻。

    Args:
        headlines: [{title, description?, published_utc?, sentiment?, url?}, ...]

    Returns:
        [{keyword, severity, theme, title, published_utc, sentiment, url}, ...]
        按 severity（high→medium）与 published_utc 逆序排列。
    """
    if not headlines:
        return []

    patterns = _compile_patterns()
    cfg = _load_keyword_config()
    force_prefixes = [str(p).lower() for p in (cfg.get("force_high_signal_prefixes") or [])]

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # (title, keyword) 去重

    for h in headlines:
        title = str(h.get("title") or "")
        if not title:
            continue
        text = _headline_text(h)

        title_lower = title.lower().strip()
        force_hit = any(title_lower.startswith(p) for p in force_prefixes)

        for theme, severity, pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            key = (title, m.group(0).lower())
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "keyword": m.group(0),
                    "severity": "high" if force_hit else severity,
                    "theme": theme,
                    "title": title,
                    "published_utc": h.get("published_utc"),
                    "sentiment": h.get("sentiment"),
                    "url": h.get("url") or h.get("link"),
                }
            )

        if force_hit and not any(m["title"] == title for m in matches):
            matches.append(
                {
                    "keyword": title_lower.split(":", 1)[0] + ":",
                    "severity": "high",
                    "theme": "breaking",
                    "title": title,
                    "published_utc": h.get("published_utc"),
                    "sentiment": h.get("sentiment"),
                    "url": h.get("url") or h.get("link"),
                }
            )

    matches.sort(
        key=lambda m: (
            -_SEVERITY_RANK.get(str(m.get("severity")), 0),
            str(m.get("published_utc") or ""),
        ),
        reverse=False,
    )
    matches.sort(key=lambda m: str(m.get("published_utc") or ""), reverse=True)
    matches.sort(key=lambda m: _SEVERITY_RANK.get(str(m.get("severity")), 0), reverse=True)
    return matches


def summarize_signals(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """便于 LLM/UI 一眼看到严重度分布。"""
    high = [m for m in matches if m.get("severity") == "high"]
    medium = [m for m in matches if m.get("severity") == "medium"]
    themes: dict[str, int] = {}
    for m in matches:
        t = str(m.get("theme") or "misc")
        themes[t] = themes.get(t, 0) + 1
    return {
        "total": len(matches),
        "high": len(high),
        "medium": len(medium),
        "themes": themes,
        "has_high": bool(high),
    }
