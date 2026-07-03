from __future__ import annotations

from typing import Any

# Mapping from macro release keywords → short catalyst label used in P13.
# Order matters: first match wins to prefer specific labels (e.g. "FOMC minutes"
# before generic "FOMC").
_CATALYST_KEYWORDS: list[tuple[str, str]] = [
    ("fomc minutes", "FOMC Minutes"),
    ("federal open market committee", "FOMC"),
    ("fomc", "FOMC"),
    ("employment situation", "NFP"),
    ("nonfarm", "NFP"),
    ("consumer price index", "CPI"),
    ("cpi", "CPI"),
    ("producer price index", "PPI"),
    ("personal income and outlays", "PCE"),
    ("pce", "PCE"),
    ("retail sales", "Retail Sales"),
    ("ism manufacturing", "ISM Mfg"),
    ("ism services", "ISM Services"),
    ("gross domestic product", "GDP"),
    ("gdp", "GDP"),
    ("jolts", "JOLTS"),
    ("adp", "ADP"),
]


def _extract_catalysts(calendar: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick major macro releases from Step 0 economic_calendar.

    Returns a de-duplicated list preserving calendar order. Each entry:
    {"name": "CPI", "release": "Consumer Price Index", "date": "2026-07-15"}
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in calendar or []:
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
        key = f"{label}|{event.get('date')}"
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
    return [c for c in _extract_catalysts(calendar) if c.get("date") == target_date]


def _pct(q: dict[str, Any]) -> float | None:
    v = q.get("change_pct")
    return float(v) if v is not None else None


def _fred_value(raw: dict[str, Any], key: str) -> float | None:
    series = (raw.get("macro") or {}).get("series") or {}
    row = series.get(key) or {}
    val = row.get("value")
    if val in (None, ".", ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _score_direction(pct: float | None, threshold: float = 0.15) -> str:
    if pct is None:
        return "Flat"
    if pct > threshold:
        return "Up"
    if pct < -threshold:
        return "Down"
    return "Flat"


def compute_rule_parts(raw: dict[str, Any]) -> dict[str, Any]:
    market_quotes = (raw.get("market") or {}).get("quotes") or {}
    dxy = market_quotes.get("DX-Y.NYB") or {}
    vix = market_quotes.get("^VIX") or {}

    dgs2 = _fred_value(raw, "DGS2")
    dgs10 = _fred_value(raw, "DGS10")

    bond_direction = "Flat"
    bond_score = 0
    if dgs10 is not None:
        # MVP: level-based proxy when we lack prior-day FRED delta
        if dgs10 >= 4.5:
            bond_direction, bond_score = "Up", -1
        elif dgs10 <= 4.0:
            bond_direction, bond_score = "Down", 1

    dxy_pct = _pct(dxy)
    dxy_dir = _score_direction(dxy_pct)
    dollar_score = 0
    if dxy_dir == "Up":
        dollar_score = -1
    elif dxy_dir == "Down":
        dollar_score = 1
    dollar_qqq = (
        "Negative" if dollar_score < 0 else "Positive" if dollar_score > 0 else "Neutral"
    )

    vix_pct = _pct(vix)
    vix_dir = _score_direction(vix_pct, threshold=2.0)
    vix_score = 0
    panic = "Low"
    if vix_dir == "Down":
        vix_score = 1
        panic = "Low"
    elif vix_dir == "Up":
        vix_score = -1
        panic = "Medium" if (vix.get("close") or 0) < 25 else "High"

    sector_quotes = (raw.get("sector") or {}).get("quotes") or {}
    sector_rows = []
    for name, q in sector_quotes.items():
        if "error" in q:
            continue
        sector_rows.append(
            {
                "sector": name,
                "change_pct": q.get("change_pct"),
                "close": q.get("close"),
            }
        )
    sector_rows.sort(key=lambda r: r.get("change_pct") or 0, reverse=True)
    strongest = sector_rows[0]["sector"] if sector_rows else "N/A"
    weakest = sector_rows[-1]["sector"] if sector_rows else "N/A"
    avg_sector = (
        sum(r.get("change_pct") or 0 for r in sector_rows) / len(sector_rows)
        if sector_rows
        else 0
    )
    rotation = "分化"
    if avg_sector > 0.3:
        rotation = "普涨"
    elif avg_sector < -0.3:
        rotation = "普跌"

    options = raw.get("options") or {}
    avg_iv = options.get("avg_implied_volatility")
    iv_level = "Medium"
    if avg_iv is not None:
        if avg_iv < 0.25:
            iv_level = "Low"
        elif avg_iv > 0.45:
            iv_level = "High"
    nonfarm_days = (raw.get("macro") or {}).get("nonfarm_days_until")
    zero_dte = "No" if nonfarm_days is not None and nonfarm_days <= 3 else "Yes"
    buy_options = "Yes" if iv_level in ("Low", "Medium") else "No"

    calendar = (raw.get("macro") or {}).get("economic_calendar") or []
    catalysts = _extract_catalysts(calendar)
    catalyst_today = bool(catalysts)

    stock_quotes = (raw.get("stocks") or {}).get("quotes") or {}
    smh = sector_quotes.get("SMH") or {}
    xlk = sector_quotes.get("XLK") or {}
    nvda = stock_quotes.get("NVDA") or {}

    ai_score = 0
    smh_pct = _pct(smh)
    nvda_pct = _pct(nvda)
    if smh_pct is not None and smh_pct > 0:
        ai_score += 1
    if nvda_pct is not None and nvda_pct > 0:
        ai_score += 1

    mag7_syms = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    mag7_rows: list[dict[str, Any]] = []
    for sym in mag7_syms:
        q = stock_quotes.get(sym) or {}
        pct = _pct(q)
        if pct is None:
            continue
        mag7_rows.append({"symbol": sym, "change_pct": pct})
    mag7_rows.sort(key=lambda r: r["change_pct"], reverse=True)
    mag7_up = sum(1 for r in mag7_rows if r["change_pct"] > 0)
    mag7_total = len(mag7_rows)
    mag7_breadth = (mag7_up / mag7_total) if mag7_total else None
    mag7_avg = (
        sum(r["change_pct"] for r in mag7_rows) / mag7_total
        if mag7_total
        else None
    )

    ai_signals = 0
    ai_signals += 1 if smh_pct is not None and smh_pct > 0 else -1 if smh_pct is not None and smh_pct < 0 else 0
    ai_signals += 1 if nvda_pct is not None and nvda_pct > 0 else -1 if nvda_pct is not None and nvda_pct < 0 else 0
    if mag7_breadth is not None:
        if mag7_breadth >= 0.6:
            ai_signals += 1
        elif mag7_breadth <= 0.4:
            ai_signals -= 1
    if mag7_avg is not None:
        if mag7_avg > 0.3:
            ai_signals += 1
        elif mag7_avg < -0.3:
            ai_signals -= 1

    if ai_signals >= 2:
        ai_stance = "Bullish"
    elif ai_signals <= -2:
        ai_stance = "Bearish"
    else:
        ai_stance = "Neutral"

    ai_leaders = [r for r in mag7_rows if r["change_pct"] > 0][:3]
    ai_laggers = [r for r in reversed(mag7_rows) if r["change_pct"] < 0][:2]

    def _fmt_row(r: dict[str, Any]) -> str:
        return f"{r['symbol']} {r['change_pct']:+.2f}%"

    scores = {
        "Macro": 0,
        "Fed": 0,
        "Bond": bond_score,
        "Dollar": dollar_score,
        "VIX": vix_score,
        "AI": ai_score,
        "Earnings": 0,
        "Breadth": 1 if avg_sector > 0 else -1 if avg_sector < 0 else 0,
        "Momentum": 1 if _pct(market_quotes.get("QQQ") or {}) and _pct(market_quotes.get("QQQ") or {}) > 0 else -1,
    }
    total = sum(scores.values())
    if total >= 5:
        bias = "Bull"
    elif total >= 2:
        bias = "Bullish Bias"
    elif total <= -5:
        bias = "Strong Bear"
    elif total <= -2:
        bias = "Bear"
    else:
        bias = "Neutral"

    parts: dict[str, Any] = {
        "P4": {
            "judgment": f"Growth：{'Bearish' if bond_score < 0 else 'Bullish' if bond_score > 0 else 'Neutral'} · Score：{bond_score}",
            "confidence": 0.7,
            "one_liner": f"10Y {dgs10 or 'N/A'}%，方向 {bond_direction}",
            "body_md": f"2Y：{dgs2 or 'N/A'}%\n10Y：{dgs10 or 'N/A'}%\nDirection：{bond_direction}\nMorning Score：Bond → {bond_score}",
            "scores": {"bond": bond_score},
        },
        "P5": {
            "judgment": f"对 QQQ：{dollar_qqq} · Score：{dollar_score}",
            "confidence": 0.65,
            "one_liner": f"Dollar {dxy_dir}，{'轻微利空' if dollar_score < 0 else '利好' if dollar_score > 0 else '中性'} QQQ",
            "body_md": f"Dollar：{dxy_dir}\nMorning Score：Dollar → {dollar_score}",
            "scores": {"dollar": dollar_score},
        },
        "P6": {
            "judgment": f"恐慌：{panic} · Score：{vix_score}",
            "confidence": 0.75,
            "one_liner": f"VIX {vix_dir}，{'Risk-on 可信' if vix_score > 0 else '谨慎' if vix_score < 0 else '中性'}",
            "body_md": f"VIX：{vix_dir}\n情绪：{panic}\nMorning Score：VIX → {vix_score}",
            "scores": {"vix": vix_score},
        },
        "P7": {
            "judgment": f"最强板块：{strongest} · 轮动：{rotation}",
            "confidence": 0.7,
            "one_liner": f"{strongest} 相对最强，{weakest} 最弱",
            "body_md": "\n".join(
                f"{r['sector']}：{r.get('change_pct', 'N/A')}%"
                for r in sector_rows
            ),
            "scores": {},
        },
        "P8": _build_p8(
            ai_stance=ai_stance,
            ai_signals=ai_signals,
            smh_pct=smh_pct,
            nvda_pct=nvda_pct,
            xlk_pct=_pct(xlk),
            mag7_rows=mag7_rows,
            mag7_breadth=mag7_breadth,
            mag7_avg=mag7_avg,
            ai_leaders=ai_leaders,
            ai_laggers=ai_laggers,
            fmt_row=_fmt_row,
        ),
        "P9": {
            "judgment": f"买期权：{buy_options} · 0DTE：{zero_dte}",
            "confidence": None,
            "one_liner": f"IV {iv_level}，P/C {options.get('put_call_ratio', 'N/A')}",
            "body_md": (
                f"QQQ IV：{iv_level}\n"
                f"是否适合买期权：{buy_options}\n"
                f"是否做 0DTE：{zero_dte}\n"
                f"数据源：{options.get('source', 'N/A')}"
            ),
            "scores": {},
        },
        "P11": {
            "judgment": f"Bias：{bias} · Total：{total:+d}",
            "confidence": min(0.85, 0.55 + abs(total) * 0.03),
            "one_liner": f"{bias}，综合得分 {total:+d}",
            "body_md": "\n".join(f"{k} | {v:+d}" for k, v in scores.items()) + f"\n\n**Total**：{total:+d}",
            "scores": scores,
            "total": total,
            "bias": bias,
        },
        "P13": _build_p13(catalysts),
    }
    return {"parts": parts, "scores": scores, "total": total, "bias": bias}


def _build_p13(catalysts: list[dict[str, Any]]) -> dict[str, Any]:
    if not catalysts:
        return {
            "judgment": "Edge：NO",
            "confidence": None,
            "one_liner": "今天无 CPI/PCE/FOMC 等重大数据，无明显 Edge",
            "body_md": "- **Edge**：NO\n- **催化剂**：无重大宏观数据",
            "catalysts": [],
            "scores": {},
        }

    labels = [c["name"] for c in catalysts]
    unique_labels: list[str] = []
    seen: set[str] = set()
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            unique_labels.append(lbl)

    def _fmt_when(date_str: str | None) -> str:
        return f" · {date_str}" if date_str else ""

    body_lines = ["- **Edge**：YES", "- **今日催化剂**："]
    for c in catalysts:
        body_lines.append(f"  - {c['name']} — {c['release']}{_fmt_when(c.get('date'))}")

    label_summary = "、".join(unique_labels)
    one_liner = f"今日有 {label_summary} 等宏观催化剂"

    return {
        "judgment": f"Edge：YES · {label_summary}",
        "confidence": 0.65,
        "one_liner": one_liner,
        "body_md": "\n".join(body_lines),
        "catalysts": catalysts,
        "scores": {},
    }


def _build_p8(
    *,
    ai_stance: str,
    ai_signals: int,
    smh_pct: float | None,
    nvda_pct: float | None,
    xlk_pct: float | None,
    mag7_rows: list[dict[str, Any]],
    mag7_breadth: float | None,
    mag7_avg: float | None,
    ai_leaders: list[dict[str, Any]],
    ai_laggers: list[dict[str, Any]],
    fmt_row,
) -> dict[str, Any]:
    def _pct_str(v: float | None) -> str:
        return f"{v:+.2f}%" if v is not None else "N/A"

    if mag7_rows:
        leader_txt = "、".join(fmt_row(r) for r in ai_leaders) or "无领涨"
        lagger_txt = "、".join(fmt_row(r) for r in ai_laggers) or "无明显拖累"
    else:
        leader_txt = lagger_txt = "数据缺失"

    stance_hint = {
        "Bullish": "AI 主题偏多",
        "Bearish": "AI 主题偏空",
        "Neutral": "AI 主题中性",
    }[ai_stance]

    one_liner = f"{stance_hint}：SMH {_pct_str(smh_pct)} · NVDA {_pct_str(nvda_pct)}"
    body_lines = [
        f"- **AI 判断**：{ai_stance}",
        f"- **SMH 半导体**：{_pct_str(smh_pct)}",
        f"- **XLK 科技**：{_pct_str(xlk_pct)}",
        f"- **NVDA**：{_pct_str(nvda_pct)}",
    ]
    if mag7_rows:
        breadth_pct = f"{(mag7_breadth or 0) * 100:.0f}%"
        avg_pct = _pct_str(mag7_avg)
        body_lines += [
            f"- **Mag7 广度**：{breadth_pct} 上涨（均值 {avg_pct}）",
            f"- **领涨**：{leader_txt}",
            f"- **拖累**：{lagger_txt}",
        ]
    else:
        body_lines.append("- **Mag7**：数据缺失")

    return {
        "judgment": f"AI 主题：{ai_stance}",
        "confidence": 0.6 if mag7_rows else 0.4,
        "one_liner": one_liner,
        "body_md": "\n".join(body_lines),
        "scores": {"ai_stance_signals": ai_signals},
    }
