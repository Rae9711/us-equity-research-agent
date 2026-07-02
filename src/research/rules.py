from __future__ import annotations

from typing import Any


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
    catalyst_today = False
    for event in calendar:
        name = str(event.get("release_name") or "").lower()
        if any(k in name for k in ("employment", "cpi", "pce", "fomc", "fed")):
            catalyst_today = True
            break

    ai_score = 0
    smh = sector_quotes.get("SMH") or {}
    nvda = ((raw.get("stocks") or {}).get("quotes") or {}).get("NVDA") or {}
    if _pct(smh) is not None and _pct(smh) > 0:
        ai_score += 1
    if _pct(nvda) is not None and _pct(nvda) > 0:
        ai_score += 1

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
        "P13": {
            "judgment": f"Edge：{'YES' if catalyst_today else 'NO'}",
            "confidence": 0.6 if catalyst_today else None,
            "one_liner": (
                "今日有重大宏观催化剂"
                if catalyst_today
                else "今天无 CPI/PCE/FOMC 等重大数据，无明显 Edge"
            ),
            "body_md": f"Do we have an edge today? {'YES' if catalyst_today else 'NO'}",
            "scores": {},
        },
    }
    return {"parts": parts, "scores": scores, "total": total, "bias": bias}
