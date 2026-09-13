"""Correlation / concentration guard for the dual-book paper trader.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

The candidate universe mixes Mag7, sector ETFs, and broad indexes, so the two
books can silently stack the *same* factor bet. When the swing core and the
intraday satellite are the same factor **and** the same direction, we treat them
as one concentrated position and scale the second leg down (or block it) so a
single factor shock cannot blow through the intended per-trade risk.
"""

from __future__ import annotations

# Coarse factor buckets. Symbols in the same group are assumed highly correlated
# for risk purposes (this is a risk ASSUMPTION, not a measured correlation).
FACTOR_GROUPS: dict[str, set[str]] = {
    "semis_ai": {
        "NVDA", "AMD", "MU", "AVGO", "ARM", "SMH", "TSM", "SOXL", "SOXX", "MRVL",
    },
    "megacap_tech": {
        "META", "MSFT", "AAPL", "AMZN", "GOOGL", "GOOG", "TSLA", "XLK",
    },
    "broad_beta": {"QQQ", "SPY", "TQQQ", "IWM", "DIA"},
    "financials": {"XLF"},
    "energy": {"XLE"},
}

# Cross-group correlation: broad_beta / semis / megacap all load on "risk-on".
RISK_ON_GROUPS = {"semis_ai", "megacap_tech", "broad_beta"}

# How much to scale the *second* leg when it duplicates the factor+direction.
SAME_GROUP_SCALE = 0.5
RELATED_GROUP_SCALE = 0.75


def factor_group(symbol: str | None) -> str | None:
    if not symbol:
        return None
    s = symbol.upper()
    for group, members in FACTOR_GROUPS.items():
        if s in members:
            return group
    return None


def correlation_scale(
    *,
    new_symbol: str | None,
    new_direction: str | None,
    open_symbol: str | None,
    open_direction: str | None,
) -> tuple[float, str | None]:
    """Return (size_multiplier, note) for a new leg given an existing position.

    1.0 = no concentration overlap. < 1.0 = same-factor same-direction stacking.
    """
    if not open_symbol or not new_symbol:
        return 1.0, None
    nd = (new_direction or "LONG").upper()
    od = (open_direction or "LONG").upper()
    ng = factor_group(new_symbol)
    og = factor_group(open_symbol)
    if ng is None or og is None:
        return 1.0, None

    if new_symbol.upper() == open_symbol.upper() and nd == od:
        return 0.0, f"已持有同标的同方向 {open_symbol}，不重复加仓"

    if ng == og and nd == od:
        return (
            SAME_GROUP_SCALE,
            f"与在手 {open_symbol} 同因子({ng})同向 → 第二腿减半以控集中度",
        )

    if ng in RISK_ON_GROUPS and og in RISK_ON_GROUPS and nd == od:
        return (
            RELATED_GROUP_SCALE,
            f"与在手 {open_symbol} 同属风险资产且同向 → 第二腿降至 75%",
        )

    # Opposite directions in the same factor = a hedge; leave it alone.
    return 1.0, None
