"""Reject nonsense fill prices for paper trading.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Prevents stop/target/entry fills when the quote is absurd vs known anchors
(entry, stop, target, last mark) — e.g. a global ``--force-price 100``
stamping every open position.
"""

from __future__ import annotations

from typing import Any

# Max relative distance from any positive reference anchor.
DEFAULT_MAX_DEVIATION_PCT = 30.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def collect_price_anchors(
    *,
    entry: float | None = None,
    stop: float | None = None,
    target: float | None = None,
    last_price: float | None = None,
    signal_entry: float | None = None,
    extra: list[float | None] | None = None,
) -> list[float]:
    """Positive reference prices used for sanity checks."""
    out: list[float] = []
    for v in (entry, stop, target, last_price, signal_entry, *(extra or [])):
        px = _safe_float(v)
        if px is not None and px > 0:
            out.append(px)
    return out


def price_deviation_pct(price: float, anchor: float) -> float:
    """Absolute percent distance of ``price`` from ``anchor``."""
    if anchor <= 0:
        return float("inf")
    return abs(price - anchor) / anchor * 100.0


def is_sane_fill_price(
    price: float | None,
    *,
    anchors: list[float] | None = None,
    max_deviation_pct: float = DEFAULT_MAX_DEVIATION_PCT,
) -> tuple[bool, str | None]:
    """Return (ok, reject_reason).

    Rules:
      - price must be finite and > 0
      - if anchors exist, price must be within ``max_deviation_pct`` of *at least
        one* anchor (closest-anchor rule). This allows a true gap through stop
        as long as the fill is still near the stop / entry / last mark, while
        rejecting absolute nonsense like META @ $100 with entry ~$670.
    """
    px = _safe_float(price)
    if px is None or px <= 0:
        return False, "报价无效（空/≤0）"

    refs = [a for a in (anchors or []) if a and a > 0]
    if not refs:
        # No anchors — only hard reject non-positive (already passed).
        return True, None

    best = min(refs, key=lambda a: price_deviation_pct(px, a))
    dev = price_deviation_pct(px, best)
    if dev > max_deviation_pct:
        return (
            False,
            (
                f"报价异常 ${px:.4g} 距参考价 ${best:.4g} "
                f"{dev:.1f}% > {max_deviation_pct:.0f}% — 拒绝成交"
            ),
        )
    return True, None


def anchors_from_position(pos: dict[str, Any] | None) -> list[float]:
    if not pos:
        return []
    return collect_price_anchors(
        entry=_safe_float(pos.get("avg_entry")),
        stop=_safe_float(pos.get("stop")),
        target=_safe_float(pos.get("target")),
        last_price=_safe_float(pos.get("last_price")),
    )


def anchors_from_signal(signal: dict[str, Any] | None) -> list[float]:
    """Anchors from the trade plan — never include the live quote itself."""
    sig = signal or {}
    zone = sig.get("entry_zone") if isinstance(sig.get("entry_zone"), dict) else {}
    return collect_price_anchors(
        entry=_safe_float(sig.get("entry_price") or sig.get("entry")),
        stop=_safe_float(sig.get("stop_price") or sig.get("stop")),
        target=_safe_float(sig.get("target_price") or sig.get("target")),
        extra=[
            _safe_float(zone.get("mid")),
            _safe_float(zone.get("low")),
            _safe_float(zone.get("high")),
        ],
    )
