"""Macro calendar — scheduled releases, breaking geo, and commodity catalysts.

Sorted by market impact (not just presence). Used by P13 macro edge, P10 driver tree,
P16 trade plan, and trade-candidate gating.
"""

from __future__ import annotations

from typing import Any, Literal

from src.utils.news_signals import detect_high_signal_news, extract_news_signals

CatalystCategory = Literal["scheduled", "breaking", "commodity"]
ImpactLevel = Literal["high", "medium", "low"]

# Order matters: labor/inflation before generic Fed/FOMC.
_CATALYST_KEYWORDS: list[tuple[str, str]] = [
    ("employment situation", "NFP"),
    ("nonfarm payrolls", "NFP"),
    ("nonfarm", "NFP"),
    ("consumer price index", "CPI"),
    ("cpi", "CPI"),
    ("producer price index", "PPI"),
    ("personal income and outlays", "PCE"),
    ("pce price", "PCE"),
    ("retail sales", "Retail Sales"),
    ("ism manufacturing", "ISM Mfg"),
    ("ism services", "ISM Services"),
    ("gross domestic product", "GDP"),
    ("gdp", "GDP"),
    ("jolts", "JOLTS"),
    ("initial claims", "Jobless Claims"),
    ("jobless claims", "Jobless Claims"),
    ("adp employment", "ADP"),
    ("fomc minutes", "FOMC Minutes"),
    ("fomc meeting", "FOMC"),
    ("fomc statement", "FOMC"),
    ("federal open market committee meeting", "FOMC"),
    ("federal open market committee", "FOMC"),
]


def extract_catalysts(
    calendar: list[dict[str, Any]],
    *,
    target_date: str | None = None,
) -> list[dict[str, Any]]:
    """Pick major macro releases from Step 0 economic_calendar."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in calendar or []:
        event_date = str(event.get("date") or "")
        if target_date and event_date != target_date:
            continue
        release = str(event.get("release_name") or "").strip()
        if not release:
            continue
        lowered = release.lower()
        label: str | None = None
        for needle, tag in _CATALYST_KEYWORDS:
            if needle in lowered:
                label = tag
                break
        if not label:
            continue
        key = f"{label}|{event_date}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": label,
                "release": release,
                "date": event.get("date"),
                "consensus": event.get("consensus"),
                "forecast": event.get("forecast"),
                "estimate": event.get("estimate"),
            }
        )
    return out


def catalysts_on_date(calendar: list[dict[str, Any]], target_date: str) -> list[dict[str, Any]]:
    """Macro catalysts from economic_calendar scheduled on target_date."""
    return [c for c in extract_catalysts(calendar) if c.get("date") == target_date]

# release_name substring → (label, base impact score)
_SCHEDULED_IMPACT: list[tuple[str, str, int]] = [
    ("employment situation", "NFP", 95),
    ("nonfarm payrolls", "NFP", 95),
    ("nonfarm", "NFP", 95),
    ("consumer price index", "CPI", 92),
    ("cpi", "CPI", 90),
    ("producer price index", "PPI", 75),
    ("personal income and outlays", "PCE", 88),
    ("pce price", "PCE", 88),
    ("fomc statement", "FOMC", 95),
    ("fomc meeting", "FOMC", 95),
    ("federal open market committee meeting", "FOMC", 95),
    ("fomc minutes", "FOMC Minutes", 82),
    ("minutes of the federal open market", "FOMC Minutes", 82),
    ("meeting minutes", "FOMC Minutes", 78),
    ("gross domestic product", "GDP", 80),
    ("gdp", "GDP", 78),
    ("retail sales", "Retail Sales", 72),
    ("ism manufacturing", "ISM Mfg", 68),
    ("ism services", "ISM Services", 65),
    ("jolts", "JOLTS", 60),
    ("initial claims", "Jobless Claims", 58),
    ("jobless claims", "Jobless Claims", 58),
    ("adp employment", "ADP", 62),
    ("treasury auction", "Treasury Auction", 55),
    ("treasury note auction", "Treasury Auction", 55),
    ("treasury bill auction", "Treasury Auction", 50),
    ("10-year note auction", "Treasury Auction", 58),
    ("30-year bond auction", "Treasury Auction", 55),
]

_GEO_THEME_LABELS: dict[str, str] = {
    "geopolitics": "Geopolitical Risk",
    "breaking": "Breaking News",
    "fed": "Fed Headlines",
    "macro_release": "Macro Surprise",
    "trade": "Trade Policy",
}

_OIL_MOVE_THRESHOLD = 1.5
_OIL_NEWS_THRESHOLD = 2


def _impact_level(score: int) -> ImpactLevel:
    if score >= 70:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _match_scheduled(release: str) -> tuple[str, int] | None:
    lowered = release.lower()
    best: tuple[str, int] | None = None
    for needle, label, score in _SCHEDULED_IMPACT:
        if needle in lowered:
            if best is None or score > best[1]:
                best = (label, score)
    return best


def _headlines_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    news = raw.get("news") or {}
    out: list[dict[str, Any]] = []
    for article in (news.get("polygon") or []):
        out.append(article)
    for item in news.get("rss") or []:
        out.append(
            {
                "title": item.get("title"),
                "description": None,
                "published_utc": item.get("published"),
                "url": item.get("link"),
            }
        )
    return out


def _oil_shock(
    raw: dict[str, Any],
    *,
    oil_news: int,
    prior_raw: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    sector_q = ((raw.get("sector") or {}).get("quotes") or {})
    xle = sector_q.get("XLE") or {}
    xle_pct = _safe_pct(xle.get("change_pct"))
    if xle_pct is None and prior_raw:
        prior_xle = ((prior_raw.get("sector") or {}).get("quotes") or {}).get("XLE") or {}
        xle_pct = _safe_pct(prior_xle.get("change_pct"))

    market_q = ((raw.get("market") or {}).get("quotes") or {})
    uso = market_q.get("USO") or {}
    uso_pct = _safe_pct(uso.get("change_pct"))

    move = max(abs(xle_pct or 0), abs(uso_pct or 0))
    if move < _OIL_MOVE_THRESHOLD and oil_news < _OIL_NEWS_THRESHOLD:
        return None

    direction = "Rally" if (xle_pct or uso_pct or 0) > 0 else "Selloff"
    score = 68 + min(15, int(move * 3)) + min(10, oil_news * 2)
    evidence_parts = []
    if xle_pct is not None:
        evidence_parts.append(f"XLE {xle_pct:+.1f}%")
    if uso_pct is not None:
        evidence_parts.append(f"USO {uso_pct:+.1f}%")
    if oil_news:
        evidence_parts.append(f"{oil_news} oil headlines")

    return {
        "name": f"Oil {direction}",
        "category": "commodity",
        "impact": _impact_level(score),
        "impact_score": score,
        "release": f"Energy / crude {direction.lower()}",
        "evidence": " · ".join(evidence_parts) or "Oil theme in news",
        "source": "market+news",
    }


def _safe_pct(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _breaking_catalysts(headlines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = detect_high_signal_news(headlines)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    _GEO_HEADLINE_HINTS = (
        "iran",
        "israel",
        "ukraine",
        "russia",
        "middle east",
        "geopolit",
        "war ",
        "missile",
        "airstrike",
        "sanction",
    )

    for h in headlines:
        title = str(h.get("title") or "")
        text = f"{title} {h.get('description') or ''}".lower()
        if not title:
            continue
        if any(k in text for k in _GEO_HEADLINE_HINTS):
            label = "Geopolitical Risk (Iran)" if "iran" in text else "Geopolitical Risk"
            key = f"geo|{label}|{title[:30]}"
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "name": label,
                        "category": "breaking",
                        "impact": "high",
                        "impact_score": 88,
                        "release": title[:120],
                        "evidence": title[:80],
                        "source": "news",
                        "theme": "geopolitics",
                        "severity": "high",
                    }
                )

    for m in matches:
        theme = str(m.get("theme") or "breaking")
        if theme not in ("geopolitics", "breaking", "fed", "macro_release", "trade"):
            continue
        label = _GEO_THEME_LABELS.get(theme, theme.title())
        if theme == "geopolitics":
            kw = str(m.get("keyword") or "").lower()
            if "iran" in kw or "iran" in str(m.get("title") or "").lower():
                label = "Geopolitical Risk (Iran)"
            elif "israel" in str(m.get("title") or "").lower():
                label = "Geopolitical Risk (Middle East)"
            else:
                label = "Geopolitical Risk"
        severity = str(m.get("severity") or "medium")
        base = 88 if severity == "high" else 72 if theme == "geopolitics" else 65
        key = f"{label}|{m.get('title', '')[:40]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": label,
                "category": "breaking",
                "impact": _impact_level(base),
                "impact_score": base,
                "release": str(m.get("title") or "")[:120],
                "evidence": f"{m.get('keyword', '')} — {str(m.get('title') or '')[:80]}",
                "source": "news",
                "theme": theme,
                "severity": severity,
            }
        )
    return out


def build_macro_calendar(
    raw: dict[str, Any],
    *,
    trading_date: str | None = None,
    prior_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """All material catalysts for the day, sorted by impact_score descending."""
    td = trading_date or str(raw.get("trading_date") or "")
    calendar = (raw.get("macro") or {}).get("economic_calendar") or []
    scheduled_raw = extract_catalysts(calendar, target_date=td or None)

    catalysts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in scheduled_raw:
        release = str(row.get("release") or row.get("name") or "")
        matched = _match_scheduled(release) or (str(row.get("name")), 60)
        label, score = matched
        key = f"scheduled|{label}|{row.get('date')}"
        if key in seen:
            continue
        seen.add(key)
        catalysts.append(
            {
                "name": label,
                "category": "scheduled",
                "impact": _impact_level(score),
                "impact_score": score,
                "release": release,
                "date": row.get("date"),
                "consensus": row.get("consensus") or row.get("forecast") or row.get("estimate"),
                "evidence": f"Scheduled: {release}",
                "source": "economic_calendar",
            }
        )

    # Catch calendar rows missed by _extract_catalysts (e.g. alternate FOMC Minutes naming)
    for event in calendar or []:
        if td and str(event.get("date") or "") != td:
            continue
        release = str(event.get("release_name") or "").strip()
        if not release:
            continue
        matched = _match_scheduled(release)
        if not matched:
            continue
        label, score = matched
        key = f"scheduled|{label}|{event.get('date')}"
        if key in seen:
            continue
        seen.add(key)
        catalysts.append(
            {
                "name": label,
                "category": "scheduled",
                "impact": _impact_level(score),
                "impact_score": score,
                "release": release,
                "date": event.get("date"),
                "evidence": f"Scheduled: {release}",
                "source": "economic_calendar",
            }
        )

    news_signals = extract_news_signals(raw.get("news"))
    oil_news = int(news_signals.get("news_oil_mentions") or 0)
    oil_cat = _oil_shock(raw, oil_news=oil_news, prior_raw=prior_raw)
    if oil_cat:
        key = f"commodity|{oil_cat['name']}"
        if key not in seen:
            seen.add(key)
            catalysts.append(oil_cat)

    for brk in _breaking_catalysts(_headlines_from_raw(raw)):
        key = f"breaking|{brk['name']}|{brk.get('release', '')[:30]}"
        if key in seen:
            continue
        seen.add(key)
        catalysts.append(brk)

    catalysts.sort(key=lambda c: (-int(c.get("impact_score") or 0), c.get("name", "")))

    has_material = any(int(c.get("impact_score") or 0) >= 50 for c in catalysts)
    high_names = [c["name"] for c in catalysts if c.get("impact") == "high"][:3]
    headline = " + ".join(high_names) if high_names else (
        "No major scheduled catalyst" if not catalysts else catalysts[0]["name"]
    )

    return {
        "catalysts": catalysts,
        "has_material_catalyst": has_material or bool(catalysts),
        "headline": headline,
        "summary": _calendar_summary(catalysts),
        "scheduled": [c for c in catalysts if c.get("category") == "scheduled"],
        "breaking": [c for c in catalysts if c.get("category") == "breaking"],
        "commodity": [c for c in catalysts if c.get("category") == "commodity"],
    }


def _calendar_summary(catalysts: list[dict[str, Any]]) -> str:
    if not catalysts:
        return "今日无重大宏观/地缘/商品催化剂"
    parts: list[str] = []
    for cat, label in (
        ("scheduled", "Scheduled"),
        ("breaking", "Breaking"),
        ("commodity", "Commodity"),
    ):
        names = [c["name"] for c in catalysts if c.get("category") == cat]
        if names:
            parts.append(f"{label}: {', '.join(names[:3])}")
    return " · ".join(parts)


def catalysts_for_edges(macro_calendar: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize macro calendar entries for edges / P13 catalyst list."""
    out: list[dict[str, Any]] = []
    for c in macro_calendar.get("catalysts") or []:
        out.append(
            {
                "name": c.get("name"),
                "release": c.get("release") or c.get("name"),
                "date": c.get("date"),
                "category": c.get("category"),
                "impact_score": c.get("impact_score"),
            }
        )
    return out


def top_catalyst_name(macro_calendar: dict[str, Any]) -> str | None:
    cats = macro_calendar.get("catalysts") or []
    return cats[0]["name"] if cats else None


def has_geopolitical_risk(macro_calendar: dict[str, Any]) -> bool:
    for c in macro_calendar.get("catalysts") or []:
        if c.get("category") == "breaking" and "geo" in str(c.get("name", "")).lower():
            return True
        if c.get("theme") == "geopolitics":
            return True
    return False
