"""Driver tree — primary / secondary / tertiary session drivers with evidence."""

from __future__ import annotations

from typing import Any


def _driver_node(
    driver_type: str,
    label: str,
    evidence: str,
    *,
    score: int,
    source: str = "rules",
) -> dict[str, Any]:
    return {
        "type": driver_type,
        "label": label,
        "evidence": evidence,
        "score": score,
        "source": source,
    }


def build_driver_tree(
    *,
    macro_calendar: dict[str, Any],
    chip_selloff: bool,
    smh_pct: float | None,
    strongest_sector: str,
    weakest_sector: str,
    qqq_pct: float | None,
    news_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank up to three concurrent drivers; highest impact wins primary slot."""
    news_signals = news_signals or {}
    candidates: list[dict[str, Any]] = []

    for c in macro_calendar.get("breaking") or []:
        name = str(c.get("name") or "")
        theme = str(c.get("theme") or "")
        if "geo" in name.lower() or theme == "geopolitics":
            dtype = "Political"
        elif theme == "fed":
            dtype = "Fed"
        elif theme == "trade":
            dtype = "Political"
        else:
            dtype = "Macro"
        score = int(c.get("impact_score") or 70) + 8
        candidates.append(
            _driver_node(dtype, name, str(c.get("evidence") or c.get("release") or ""), score=score)
        )

    for c in macro_calendar.get("commodity") or []:
        score = int(c.get("impact_score") or 65)
        candidates.append(
            _driver_node(
                "Macro",
                str(c.get("name") or "Commodity Shock"),
                str(c.get("evidence") or ""),
                score=score,
            )
        )

    for c in macro_calendar.get("scheduled") or []:
        name = str(c.get("name") or "")
        if name in ("FOMC", "FOMC Minutes"):
            dtype = "Fed"
        else:
            dtype = "Macro"
        score = int(c.get("impact_score") or 60)
        candidates.append(
            _driver_node(dtype, name, str(c.get("evidence") or c.get("release") or ""), score=score)
        )

    if chip_selloff:
        smh_txt = f"{smh_pct:+.1f}%" if smh_pct is not None else "weak"
        candidates.append(
            _driver_node(
                "AI",
                "AI Chip Rotation",
                f"SMH {smh_txt} · semis lagging · AI headline flow",
                score=72,
            )
        )

    if strongest_sector not in ("N/A", "", "SMH") and strongest_sector:
        sec_pct = ""
        candidates.append(
            _driver_node(
                "Positioning",
                f"{strongest_sector} Rotation",
                f"{strongest_sector} leading vs {weakest_sector or 'peers'}",
                score=55,
            )
        )
    elif strongest_sector == "SMH" and smh_pct is not None and smh_pct > 0 and not chip_selloff:
        candidates.append(
            _driver_node(
                "Momentum",
                "AI Momentum",
                f"SMH {smh_pct:+.1f}% leading",
                score=58,
            )
        )

    if qqq_pct is not None and abs(qqq_pct) > 0.5 and not chip_selloff:
        candidates.append(
            _driver_node(
                "Momentum",
                "Index Momentum",
                f"QQQ {qqq_pct:+.1f}%",
                score=50,
            )
        )

    oil_news = int(news_signals.get("news_oil_mentions") or 0)
    if oil_news >= 3 and not any("Oil" in c["label"] for c in candidates):
        candidates.append(
            _driver_node("Macro", "Oil / Energy Theme", f"{oil_news} oil headlines", score=62)
        )

    if not candidates:
        candidates.append(
            _driver_node("No Catalyst", "No dominant catalyst", "Quiet session — range rules", score=10)
        )

    # De-dupe by label, keep highest score
    by_label: dict[str, dict[str, Any]] = {}
    for c in candidates:
        lbl = c["label"]
        if lbl not in by_label or c["score"] > by_label[lbl]["score"]:
            by_label[lbl] = c
    ranked = sorted(by_label.values(), key=lambda x: -x["score"])

    tree: dict[str, Any] = {
        "primary": ranked[0] if len(ranked) > 0 else None,
        "secondary": ranked[1] if len(ranked) > 1 else None,
        "tertiary": ranked[2] if len(ranked) > 2 else None,
        "all_ranked": ranked[:6],
        "session_update_slot": True,
    }
    return tree


def driver_tree_display(tree: dict[str, Any]) -> str:
    parts: list[str] = []
    for slot in ("primary", "secondary", "tertiary"):
        node = tree.get(slot)
        if node:
            parts.append(f"{node['type']}: {node['label']}")
    return " → ".join(parts) if parts else "—"


def primary_from_tree(tree: dict[str, Any]) -> tuple[str, str]:
    """Backward-compatible (driver_type, daily_driver) from tree primary."""
    primary = tree.get("primary") or {}
    return str(primary.get("type") or "No Catalyst"), str(primary.get("label") or "No dominant catalyst")
