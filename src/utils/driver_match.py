"""Fuzzy driver matching — Morning P10 vs S7 actual_driver."""

from __future__ import annotations

OVERLAP_KEYWORDS = [
    "ai",
    "semiconductor",
    "semi",
    "macro",
    "employment",
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
]


def driver_match_level(morning_driver: str, actual_driver: str) -> str:
    """Return 对 | 部分对 | 错 | N/A."""
    if not morning_driver or not morning_driver.strip():
        return "N/A"
    if not actual_driver or actual_driver == "Unknown":
        return "N/A"

    morning_lower = morning_driver.lower()
    actual_lower = actual_driver.lower()

    if any(kw in morning_lower for kw in actual_lower.replace("/", " ").split()):
        return "对"

    for kw in OVERLAP_KEYWORDS:
        if kw in morning_lower and kw in actual_lower:
            return "部分对"

    # slash-separated actual (e.g. AI/Semi)
    for part in actual_lower.replace("/", " ").split():
        part = part.strip()
        if len(part) >= 2 and part in morning_lower:
            return "部分对"

    return "错"


def driver_hit(morning_driver: str, actual_driver: str) -> bool:
    """True when morning driver fuzzy-matches S7 actual (对 or 部分对)."""
    level = driver_match_level(morning_driver, actual_driver)
    return level in ("对", "部分对")
