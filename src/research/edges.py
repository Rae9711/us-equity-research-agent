"""Split Edge model — macro / index / sector / stock (Decision Agent v2).

ADVISORY ONLY — 不构成投资建议.
"""

from __future__ import annotations

from typing import Any

SECTOR_RS_THRESHOLD = 0.15
INDEX_RS_THRESHOLD = 0.10
STOCK_RS_THRESHOLD = 0.30

_STOCK_SYMBOLS = ("NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL")


def _edge_row(
    edge: str,
    *,
    why: str,
    catalyst: str | None = None,
    sector: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"edge": edge, "why": why}
    if catalyst is not None:
        row["catalyst"] = catalyst
    if sector is not None:
        row["sector"] = sector
    if symbol is not None:
        row["symbol"] = symbol
    return row


def _news_hits(raw: dict[str, Any], symbol: str) -> int:
    sym = symbol.upper()
    count = 0
    news = raw.get("news") or {}
    for article in (news.get("polygon") or []) + (news.get("rss") or []):
        ticker = str(article.get("ticker") or "").upper()
        title = str(article.get("title") or "").upper()
        if ticker == sym or sym in title:
            count += 1
    return count


def _symbol_session_pct(
    symbol: str,
    raw: dict[str, Any],
    sym_pcts: dict[str, float | None],
) -> float | None:
    if symbol in sym_pcts:
        return sym_pcts[symbol]
    section = "market" if symbol in ("QQQ", "SPY", "TQQQ") else (
        "sector" if symbol == "SMH" else "stocks"
    )
    q = ((raw.get(section) or {}).get("quotes") or {}).get(symbol) or {}
    chg = q.get("change_pct")
    try:
        return float(chg) if chg is not None else None
    except (TypeError, ValueError):
        return None


def _volume_signal(q: dict[str, Any], sym_pct: float | None) -> bool:
    vol = q.get("volume")
    try:
        vol_i = int(vol) if vol is not None else 0
    except (TypeError, ValueError):
        vol_i = 0
    session = str(q.get("session_type") or "")
    if session == "premarket" and vol_i > 0:
        return True
    return sym_pct is not None and sym_pct > 0.15


def compute_edges(
    raw: dict[str, Any],
    *,
    catalysts_today: list[dict[str, Any]] | None = None,
    macro_calendar: dict[str, Any] | None = None,
    qqq_pct: float | None = None,
    smh_pct: float | None = None,
    spy_pct: float | None = None,
    sym_pcts: dict[str, float | None] | None = None,
    driver_type: str = "",
) -> dict[str, Any]:
    """Compute four independent edges for P13 + P18 decision tree."""
    sym_pcts = sym_pcts or {}
    catalysts = catalysts_today or []
    cal = macro_calendar or {}

    # Macro edge — scheduled releases, breaking geo, or commodity shocks
    material = cal.get("has_material_catalyst") if cal else bool(catalysts)
    cal_cats = cal.get("catalysts") or catalysts
    if material and cal_cats:
        labels: list[str] = []
        seen: set[str] = set()
        for c in cal_cats:
            name = c.get("name") or ""
            if name and name not in seen:
                seen.add(name)
                labels.append(name)
        catalyst_str = "、".join(labels[:4]) if labels else "宏观/地缘催化剂"
        categories = {c.get("category") for c in cal_cats if c.get("category")}
        if "breaking" in categories:
            why = f"今日有重大催化剂（含地缘/突发）：{catalyst_str}"
        elif "commodity" in categories:
            why = f"今日有商品/能源冲击 + 宏观事件：{catalyst_str}"
        else:
            why = f"今日有 {catalyst_str} 等宏观催化剂"
        macro = _edge_row("YES", why=why, catalyst=catalyst_str)
    else:
        macro = _edge_row("NO", why="今日无重大宏观/地缘/商品催化剂", catalyst=None)

    # Index edge — QQQ/SPY session direction at open
    if qqq_pct is not None and qqq_pct > INDEX_RS_THRESHOLD:
        index = _edge_row("YES", why=f"QQQ 盘前/开盘 +{qqq_pct:.2f}%")
    elif spy_pct is not None and spy_pct > INDEX_RS_THRESHOLD:
        index = _edge_row("YES", why=f"SPY 盘前/开盘 +{spy_pct:.2f}%")
    elif qqq_pct is not None and qqq_pct < -INDEX_RS_THRESHOLD:
        index = _edge_row("YES", why=f"QQQ 走弱 {qqq_pct:.2f}%（做空 edge）")
    else:
        index = _edge_row("NO", why="指数方向不明或幅度不足")

    # Sector edge — SMH vs QQQ relative strength
    smh_vs = None
    if smh_pct is not None and qqq_pct is not None:
        smh_vs = smh_pct - qqq_pct
    sector_name = "AI" if (driver_type or "").upper() in ("AI", "MOMENTUM") else "Semiconductor"
    if smh_vs is not None and smh_vs >= SECTOR_RS_THRESHOLD:
        sector = _edge_row(
            "YES",
            why=f"SMH 领先 QQQ {smh_vs:+.2f}%",
            sector=sector_name,
        )
    else:
        sector = _edge_row("NO", why="板块相对强度不足", sector=None)

    # Stock edge — single-name RS + news + volume
    stock_q = (raw.get("stocks") or {}).get("quotes") or {}
    best_sym: str | None = None
    best_rs = -999.0
    for sym in _STOCK_SYMBOLS:
        sym_pct = _symbol_session_pct(sym, raw, sym_pcts)
        q = stock_q.get(sym) or {}
        if sym_pct is None or qqq_pct is None:
            continue
        rs = sym_pct - qqq_pct
        news_n = _news_hits(raw, sym)
        vol_ok = _volume_signal(q, sym_pct)
        if rs >= STOCK_RS_THRESHOLD and news_n >= 1 and vol_ok:
            if rs > best_rs:
                best_rs = rs
                best_sym = sym

    if best_sym:
        stock = _edge_row(
            "YES",
            why=f"{best_sym} RS+新闻+量能",
            symbol=best_sym,
        )
    else:
        stock = _edge_row("NO", why="无单票 RS+新闻+量能共振", symbol=None)

    return {
        "macro_edge": macro,
        "index_edge": index,
        "sector_edge": sector,
        "stock_edge": stock,
        "advisory": "ADVISORY — 不构成投资建议",
    }


def format_p13_from_edges(edges: dict[str, Any]) -> dict[str, Any]:
    """Build P13 part dict with four-edge display."""
    macro = edges.get("macro_edge") or {}
    index = edges.get("index_edge") or {}
    sector = edges.get("sector_edge") or {}
    stock = edges.get("stock_edge") or {}

    yes_count = sum(
        1 for e in (macro, index, sector, stock) if e.get("edge") == "YES"
    )
    judgment = f"Edge：{yes_count}/4 YES"

    body_lines = [
        "- **Macro Edge**：" + macro.get("edge", "NO")
        + (f" · {macro.get('catalyst')}" if macro.get("catalyst") else ""),
        f"  - {macro.get('why', '')}",
        f"- **Index Edge**：{index.get('edge', 'NO')}",
        f"  - {index.get('why', '')}",
        "- **Sector Edge**：" + sector.get("edge", "NO")
        + (f" · {sector.get('sector')}" if sector.get("sector") else ""),
        f"  - {sector.get('why', '')}",
        "- **Stock Edge**：" + stock.get("edge", "NO")
        + (f" · {stock.get('symbol')}" if stock.get("symbol") else ""),
        f"  - {stock.get('why', '')}",
    ]

    one_parts = []
    if macro.get("edge") == "YES":
        one_parts.append(f"宏观 {macro.get('catalyst', '')}")
    if stock.get("edge") == "YES":
        one_parts.append(f"个股 {stock.get('symbol')}")
    if sector.get("edge") == "YES":
        one_parts.append(f"板块 {sector.get('sector')}")
    one_liner = " · ".join(one_parts) if one_parts else "四路 Edge 均偏弱，依赖个股评分"

    catalysts_out = edges.get("macro_edge", {}).get("catalyst")
    catalyst_list: list[dict[str, Any]] = []
    if catalysts_out:
        for part in str(catalysts_out).split("、"):
            part = part.strip()
            if part:
                catalyst_list.append({"name": part, "release": part})

    return {
        "judgment": judgment,
        "confidence": 0.55 + yes_count * 0.08,
        "one_liner": one_liner,
        "body_md": "\n".join(body_lines),
        "edges": edges,
        "catalysts": catalyst_list,
        "scores": {},
    }
