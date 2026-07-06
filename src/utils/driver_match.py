"""Fuzzy driver matching — Morning P10 vs S7 actual_driver."""

from __future__ import annotations

import re

OVERLAP_KEYWORDS = [
    "ai",
    "semiconductor",
    "semi",
    "macro",
    "employment",
    "nfp",
    "payroll",
    "nonfarm",
    "chip",
    "bond",
    "fed",
    "fomc",
    "risk",
    "oil",
    "energy",
    "geo",
    "dollar",
    "liquidity",
    "inflation",
    "cpi",
    "momentum",
    "positioning",
    "rebalance",
    "earnings",
    "political",
    "rates",
    "catalyst",
]

# driver_type taxonomy → keywords that match S7 actual_driver text
DRIVER_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Macro": ["macro", "nfp", "cpi", "pce", "ppi", "gdp", "employment", "payroll", "inflation", "jobs"],
    "Fed": ["fed", "fomc", "powell", "rate cut", "rate hike"],
    "Rates": ["bond", "yield", "10y", "rates", "treasury"],
    "AI": ["ai", "semiconductor", "semi", "chip", "smh", "nvda", "mag7"],
    "Momentum": ["momentum", "breakout", "trend", "index momentum", "ai momentum"],
    "Positioning": ["positioning", "rotation", "rebalance", "flow"],
    "Earnings": ["earnings", "guidance", "eps"],
    "Liquidity": ["liquidity", "risk appetite", "risk-on", "risk-off"],
    "Political": ["political", "election", "tariff", "trade war", "geo"],
    "Rebalance": ["rebalance", "quarter end", "window dressing"],
    "No Catalyst": ["no catalyst", "no dominant", "quiet", "range"],
}

_VAGUE_MACRO_RE = re.compile(r"^macro$", re.I)


def _normalize_driver_text(text: str) -> str:
    """Strip P10 judgment prefix to get comparable driver label."""
    t = text.strip()
    t = re.sub(r"^type[：:]\s*\w+\s*[·•]\s*driver[：:]\s*", "", t, flags=re.I)
    t = re.sub(r"^driver[：:]\s*", "", t, flags=re.I)
    return t.strip()


def driver_match_level(
    morning_driver: str,
    actual_driver: str,
    *,
    morning_driver_type: str | None = None,
) -> str:
    """Return 对 | 部分对 | 错 | N/A."""
    if not morning_driver or not morning_driver.strip():
        return "N/A"
    if not actual_driver or actual_driver == "Unknown":
        return "N/A"

    morning_lower = _normalize_driver_text(morning_driver).lower()
    actual_lower = actual_driver.lower()

    if _VAGUE_MACRO_RE.match(morning_lower) and "macro" not in actual_lower:
        return "错"

    if morning_driver_type:
        type_kws = DRIVER_TYPE_KEYWORDS.get(morning_driver_type, [])
        type_hit = any(kw in actual_lower for kw in type_kws)
        label_hit = any(kw in morning_lower and kw in actual_lower for kw in OVERLAP_KEYWORDS)
        if type_hit and label_hit:
            return "对"
        if type_hit or label_hit:
            return "部分对"

    if any(kw in morning_lower for kw in actual_lower.replace("/", " ").split()):
        return "对"

    for kw in OVERLAP_KEYWORDS:
        if kw in morning_lower and kw in actual_lower:
            return "部分对"

    for part in actual_lower.replace("/", " ").split():
        part = part.strip()
        if len(part) >= 2 and part in morning_lower:
            return "部分对"

    # driver_type-only fallback
    if morning_driver_type:
        type_kws = DRIVER_TYPE_KEYWORDS.get(morning_driver_type, [])
        if any(kw in actual_lower for kw in type_kws):
            return "部分对"

    return "错"


def driver_hit(
    morning_driver: str,
    actual_driver: str,
    *,
    morning_driver_type: str | None = None,
) -> bool:
    """True when morning driver fuzzy-matches S7 actual (对 or 部分对)."""
    level = driver_match_level(
        morning_driver, actual_driver, morning_driver_type=morning_driver_type
    )
    return level in ("对", "部分对")
