from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from src.utils.paths import raw_data_path
from src.utils.quote_resolve import session_change_pct
from src.utils.trading_calendar import prior_trading_day

# Driver taxonomy — P10 driver_type must be one of these.
DRIVER_TYPES: list[str] = [
    "Macro",
    "Positioning",
    "Momentum",
    "Earnings",
    "AI",
    "Fed",
    "Rates",
    "Political",
    "Liquidity",
    "Rebalance",
    "No Catalyst",
]

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


def _extract_catalysts(
    calendar: list[dict[str, Any]],
    *,
    target_date: str | None = None,
) -> list[dict[str, Any]]:
    """Pick major macro releases from Step 0 economic_calendar.

    When target_date is set, only events scheduled on that date are included.
    """
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
    return [c for c in _extract_catalysts(calendar) if c.get("date") == target_date]


def _pct(q: dict[str, Any]) -> float | None:
    v = q.get("change_pct")
    return float(v) if v is not None else None


def _load_prior_raw(trading_day: date) -> dict[str, Any]:
    prior_path = raw_data_path(prior_trading_day(trading_day).isoformat())
    if not prior_path.exists():
        return {}
    try:
        return json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _session_pct(
    ticker: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date | None,
    q: dict[str, Any],
    *,
    section: str,
) -> float | None:
    if trading_day is not None:
        pct = session_change_pct(ticker, raw, prior_raw, trading_day, section=section)
        if pct is not None:
            return pct
    return _pct(q)


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


def _index_divergence(
    dow_pct: float | None,
    spy_pct: float | None,
    qqq_pct: float | None,
    smh_pct: float | None,
) -> tuple[str, bool]:
    """Return (P1 label, is_divergence)."""
    if dow_pct is None or qqq_pct is None:
        return "Neutral", False
    dow_up = dow_pct > 0.1
    qqq_down = qqq_pct < -0.1
    smh_weak = smh_pct is not None and smh_pct <= -3.0
    spy_flat = spy_pct is not None and abs(spy_pct) <= 0.25
    if dow_up and qqq_down and (smh_weak or spy_flat):
        return "分化 · Divergence", True
    if qqq_pct > 0.15 and smh_pct is not None and smh_pct > 0.15:
        return "Risk-on", False
    if qqq_pct < -0.15 and (smh_pct or 0) < -0.15:
        return "Risk-off", False
    if abs(qqq_pct) <= 0.15 and abs(dow_pct) <= 0.15:
        return "Neutral", False
    return "Neutral", False


def _chip_selloff_signal(
    smh_pct: float | None,
    nvda_pct: float | None,
    news_ai_mentions: int,
    news_macro_mentions: int,
) -> bool:
    if smh_pct is not None and smh_pct <= -3.0:
        return True
    if nvda_pct is not None and nvda_pct <= -2.0 and (news_ai_mentions or 0) >= 2:
        return True
    if smh_pct is not None and smh_pct <= -2.0 and (news_ai_mentions or 0) > (news_macro_mentions or 0):
        return True
    return False


def _daily_macro_driver(catalysts_today: list[dict[str, Any]]) -> str | None:
    if not catalysts_today:
        return None
    priority = ["NFP", "CPI", "PCE", "FOMC", "FOMC Minutes", "GDP", "Retail Sales", "Jobless Claims"]
    names = [c.get("name") for c in catalysts_today]
    for tag in priority:
        if tag in names:
            return tag
    return catalysts_today[0].get("name")


def _daily_driver_type_and_driver(
    catalysts_today: list[dict[str, Any]],
    chip_selloff: bool,
    smh_pct: float | None,
    strongest_sector: str,
    qqq_pct: float | None,
) -> tuple[str, str]:
    """Return (driver_type, specific_driver). Never Macro when no catalyst today."""
    macro = _daily_macro_driver(catalysts_today)

    if macro == "NFP":
        if chip_selloff and smh_pct is not None and smh_pct <= -3.0:
            return "Macro", "NFP + AI Chip Selloff"
        return "Macro", "NFP"

    if macro in ("FOMC", "FOMC Minutes"):
        return "Fed", macro

    if macro:
        return "Macro", macro

    # No macro catalyst — driver_type must NOT be Macro
    if chip_selloff:
        return "AI", "AI Chip Selloff"

    if strongest_sector == "SMH":
        if smh_pct is not None and smh_pct > 0:
            return "Momentum", "AI Momentum"
        return "AI", "AI Momentum"

    if qqq_pct is not None and abs(qqq_pct) > 0.5:
        return "Momentum", "Index Momentum"

    if strongest_sector not in ("N/A", "") and strongest_sector not in ("SMH",):
        return "Positioning", f"{strongest_sector} Rotation"

    return "No Catalyst", "No dominant catalyst"


def _bond_growth_lens(bond_direction: str, bond_score: int) -> str:
    if bond_score > 0:
        return "Tailwind"
    if bond_score < 0:
        return "Headwind"
    return "No Headwind"


def _p2_causal_chain(
    *,
    driver_type: str,
    driver: str,
    chip_selloff: bool,
    macro_driver: str | None,
    smh_pct: float | None,
) -> str:
    if chip_selloff:
        return (
            f"AI Chip Selloff → SMH {smh_pct or 'N/A'}% → NVDA/META 新闻 → Nasdaq 承压"
        )
    if macro_driver:
        return f"{macro_driver} → Bond → Dollar → Sector → Index"
    if driver_type in ("Momentum", "AI", "Positioning"):
        return f"{driver_type} ({driver}) → Sector → Index"
    return "No Macro Catalyst → Bond → Dollar → Sector → Index"


def _build_p9_options(
    *,
    catalyst_today: bool,
    pre_holiday: bool,
    iv_level: str,
    total: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Four explicit YES/NO fields; 0DTE only when buy_options is Yes."""
    event_day = catalyst_today or pre_holiday
    can_buy = not event_day and iv_level in ("Low", "Medium")
    buy_options = "Yes" if can_buy else "No"
    zero_dte = "Yes" if can_buy and iv_level == "Low" and not event_day else "No"
    if buy_options == "No":
        zero_dte = "No"
    buy_call = "Yes" if buy_options == "Yes" and total >= 0 else "No"
    buy_put = "Yes" if buy_options == "Yes" and total < 0 else "No"

    judgment = (
        f"买期权：{buy_options} · 0DTE：{zero_dte} · "
        f"Call：{buy_call} · Put：{buy_put}"
    )
    one_liner = (
        "事件日/节前 → 全部 NO"
        if event_day
        else f"IV {iv_level}，P/C {options.get('put_call_ratio', 'N/A')}"
    )
    body_md = "\n".join(
        [
            f"- **适合买期权？**：{buy_options}",
            f"- **适合做 0DTE？**：{zero_dte}",
            f"- **适合买 Call？**：{buy_call}",
            f"- **适合买 Put？**：{buy_put}",
            f"- **QQQ IV**：{iv_level}",
            f"- **事件日**：{'是' if catalyst_today else '否'} · **节前**：{'是' if pre_holiday else '否'}",
            f"- **数据源**：{options.get('source', 'N/A')}",
        ]
    )
    return {
        "judgment": judgment,
        "confidence": None,
        "one_liner": one_liner,
        "body_md": body_md,
        "buy_options": buy_options,
        "zero_dte": zero_dte,
        "buy_call": buy_call,
        "buy_put": buy_put,
        "scores": {},
    }


def _build_p14_preference(
    *,
    ai_stance: str,
    chip_selloff: bool,
    strongest: str,
    weakest: str,
    smh_pct: float | None,
    avg_sector: float,
    qqq_pct: float | None,
) -> dict[str, Any]:
    prefs: list[str] = []
    if chip_selloff or ai_stance == "Bearish":
        prefs.append("Defensive > AI")
    elif ai_stance == "Bullish" or strongest == "SMH":
        prefs.append("AI > Defensive")
    else:
        prefs.append("AI ≈ Defensive")

    if avg_sector < -0.3:
        prefs.append("Value > Growth")
    elif smh_pct is not None and smh_pct > 0 and avg_sector > 0:
        prefs.append("Growth > Value")
    else:
        prefs.append("Growth ≈ Value")

    if qqq_pct is not None and abs(qqq_pct) > 0.3:
        prefs.append("Momentum > Mean Reversion")
    else:
        prefs.append("Mean Reversion > Momentum")

    judgment = " · ".join(prefs)
    return {
        "judgment": judgment,
        "confidence": 0.65,
        "one_liner": f"板块：{strongest} 最强 / {weakest} 最弱",
        "body_md": "\n".join(f"- **{p}**" for p in prefs),
        "preferences": prefs,
        "scores": {},
    }


def _pre_holiday(raw: dict[str, Any]) -> bool:
    from src.utils.trading_calendar import is_trading_day, next_trading_day, today_et

    td = raw.get("trading_date")
    try:
        d = date.fromisoformat(str(td)) if td else today_et()
    except ValueError:
        d = today_et()
    return (next_trading_day(d) - d).days > 1


def compute_rule_parts(raw: dict[str, Any]) -> dict[str, Any]:
    trading_date = str(raw.get("trading_date") or "")
    trading_day: date | None = None
    if trading_date:
        try:
            trading_day = date.fromisoformat(trading_date)
        except ValueError:
            trading_day = None
    prior_raw = _load_prior_raw(trading_day) if trading_day else {}

    market_quotes = (raw.get("market") or {}).get("quotes") or {}
    dxy = market_quotes.get("DX-Y.NYB") or {}
    vix = market_quotes.get("^VIX") or {}
    spy_q = market_quotes.get("SPY") or {}
    qqq_q = market_quotes.get("QQQ") or {}
    dia_q = market_quotes.get("DIA") or {}

    dgs2 = _fred_value(raw, "DGS2")
    dgs10 = _fred_value(raw, "DGS10")

    bond_direction = "Flat"
    bond_score = 0
    if dgs10 is not None:
        if dgs10 >= 4.5:
            bond_direction, bond_score = "Up", -1
        elif dgs10 <= 4.0:
            bond_direction, bond_score = "Down", 1

    dxy_pct = _session_pct("DX-Y.NYB", raw, prior_raw, trading_day, dxy, section="market")
    dxy_dir = _score_direction(dxy_pct)
    dollar_score = 0
    if dxy_dir == "Up":
        dollar_score = -1
    elif dxy_dir == "Down":
        dollar_score = 1
    dollar_qqq = (
        "Negative" if dollar_score < 0 else "Positive" if dollar_score > 0 else "Neutral"
    )

    vix_pct = _session_pct("^VIX", raw, prior_raw, trading_day, vix, section="market")
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
                "change_pct": _session_pct(
                    name, raw, prior_raw, trading_day, q, section="sector"
                ),
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

    spy_pct = _session_pct("SPY", raw, prior_raw, trading_day, spy_q, section="market")
    qqq_pct = _session_pct("QQQ", raw, prior_raw, trading_day, qqq_q, section="market")
    dow_pct = _session_pct("DIA", raw, prior_raw, trading_day, dia_q, section="market")
    smh = sector_quotes.get("SMH") or {}
    xlk = sector_quotes.get("XLK") or {}
    smh_pct = _session_pct("SMH", raw, prior_raw, trading_day, smh, section="sector")

    p1_label, is_divergence = _index_divergence(dow_pct, spy_pct, qqq_pct, smh_pct)
    if is_divergence:
        rotation = "分化 · Divergence"
    elif avg_sector > 0.3:
        rotation = "普涨"
    elif avg_sector < -0.3:
        rotation = "普跌"
    else:
        rotation = "轮动"

    options = raw.get("options") or {}
    avg_iv = options.get("avg_implied_volatility")
    iv_level = "Medium"
    if avg_iv is not None:
        if avg_iv < 0.25:
            iv_level = "Low"
        elif avg_iv > 0.45:
            iv_level = "High"

    calendar = (raw.get("macro") or {}).get("economic_calendar") or []
    catalysts_today = _extract_catalysts(calendar, target_date=trading_date or None)
    catalyst_today = bool(catalysts_today)
    macro_driver = _daily_macro_driver(catalysts_today)

    from src.utils.news_signals import extract_news_signals

    news_signals = extract_news_signals(raw.get("news"))
    nvda_q = (raw.get("stocks") or {}).get("quotes", {}).get("NVDA") or {}
    chip_selloff = _chip_selloff_signal(
        smh_pct,
        _session_pct("NVDA", raw, prior_raw, trading_day, nvda_q, section="stocks"),
        int(news_signals.get("news_ai_mentions") or 0),
        int(news_signals.get("news_macro_mentions") or 0),
    )
    daily_driver_type, daily_driver = _daily_driver_type_and_driver(
        catalysts_today, chip_selloff, smh_pct, strongest, qqq_pct
    )

    pre_holiday = _pre_holiday(raw)

    stock_quotes = (raw.get("stocks") or {}).get("quotes") or {}
    nvda_pct = _session_pct("NVDA", raw, prior_raw, trading_day, nvda_q, section="stocks")

    ai_score = 0
    if smh_pct is not None and smh_pct > 0:
        ai_score += 1
    if nvda_pct is not None and nvda_pct > 0:
        ai_score += 1

    mag7_syms = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    mag7_rows: list[dict[str, Any]] = []
    for sym in mag7_syms:
        q = stock_quotes.get(sym) or {}
        pct = _session_pct(sym, raw, prior_raw, trading_day, q, section="stocks")
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
        "Momentum": 1 if qqq_pct is not None and qqq_pct > 0 else -1,
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

    bond_lens = _bond_growth_lens(bond_direction, bond_score)
    bond_causal = f"10Y {bond_direction} → {bond_lens}"
    if macro_driver and bond_score != 0:
        bond_causal = f"{macro_driver} → 10Y {bond_direction} → {bond_lens}"
    dollar_causal = (
        f"{macro_driver} → DXY {dxy_dir} → QQQ {dollar_qqq}"
        if macro_driver
        else f"DXY {dxy_dir} → QQQ {dollar_qqq}"
    )

    p2_chain = _p2_causal_chain(
        driver_type=daily_driver_type,
        driver=daily_driver,
        chip_selloff=chip_selloff,
        macro_driver=macro_driver,
        smh_pct=smh_pct,
    )
    p2_driver = daily_driver if not chip_selloff else "AI Chip Selloff"

    parts: dict[str, Any] = {
        "P1": {
            "judgment": p1_label,
            "confidence": 0.75 if is_divergence else 0.65,
            "one_liner": (
                f"DIA {dow_pct or 'N/A'}% · SPY {spy_pct or 'N/A'}% · QQQ {qqq_pct or 'N/A'}% · SMH {smh_pct or 'N/A'}%"
            ),
            "body_md": "\n".join(
                [
                    f"- **DIA (Dow proxy)**：{dow_pct or 'N/A'}%",
                    f"- **SPY**：{spy_pct or 'N/A'}%",
                    f"- **QQQ**：{qqq_pct or 'N/A'}%",
                    f"- **SMH**：{smh_pct or 'N/A'}%",
                    f"- **定性**：{p1_label}",
                ]
            ),
        },
        "P2": {
            "judgment": f"Driver：{p2_driver}",
            "confidence": 0.7,
            "one_liner": p2_chain[:200],
            "body_md": f"- **因果链**：{p2_chain}",
        },
        "P10": {
            "judgment": f"Type：{daily_driver_type} · Driver：{daily_driver}",
            "driver_type": daily_driver_type,
            "driver": daily_driver,
            "confidence": 0.75 if catalyst_today else 0.6,
            "one_liner": f"Primary Driver Type：{daily_driver_type} · {daily_driver}",
            "body_md": (
                f"- **Primary Driver Type**：{daily_driver_type}\n"
                f"- **Primary Driver**：{daily_driver}\n"
                f"- **催化剂**：{macro_driver or '无'}\n"
                f"- **半导体**：{'AI Chip Selloff' if chip_selloff else '非主导'}"
            ),
        },
        "P4": {
            "judgment": f"Growth：{'Bearish' if bond_score < 0 else 'Bullish' if bond_score > 0 else 'Neutral'} · Score：{bond_score}",
            "confidence": 0.7,
            "one_liner": bond_causal,
            "body_md": (
                f"- **10Y**：{dgs10 or 'N/A'}% ({bond_direction})\n"
                f"- **2Y**：{dgs2 or 'N/A'}%\n"
                f"- **Growth 镜头**：{bond_lens}\n"
                f"- **因果**：{bond_causal}\n"
                f"- **Morning Score**：Bond → {bond_score}"
            ),
            "scores": {"bond": bond_score},
        },
        "P5": {
            "judgment": f"对 QQQ：{dollar_qqq} · Score：{dollar_score}",
            "confidence": 0.65,
            "one_liner": dollar_causal,
            "body_md": (
                f"- **因果**：{dollar_causal}\n"
                f"- **Morning Score**：Dollar → {dollar_score}"
            ),
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
            xlk_pct=_session_pct("XLK", raw, prior_raw, trading_day, xlk, section="sector"),
            mag7_rows=mag7_rows,
            mag7_breadth=mag7_breadth,
            mag7_avg=mag7_avg,
            ai_leaders=ai_leaders,
            ai_laggers=ai_laggers,
            fmt_row=_fmt_row,
            chip_selloff=chip_selloff,
        ),
        "P9": _build_p9_options(
            catalyst_today=catalyst_today,
            pre_holiday=pre_holiday,
            iv_level=iv_level,
            total=total,
            options=options,
        ),
        "P11": {
            "judgment": f"Bias：{bias} · Total：{total:+d}",
            "confidence": min(0.85, 0.55 + abs(total) * 0.03),
            "one_liner": f"{bias}，综合得分 {total:+d}",
            "body_md": "\n".join(f"{k} | {v:+d}" for k, v in scores.items()) + f"\n\n**Total**：{total:+d}",
            "scores": scores,
            "total": total,
            "bias": bias,
        },
        "P13": _build_p13(catalysts_today),
        "P14": _build_p14_preference(
            ai_stance=ai_stance,
            chip_selloff=chip_selloff,
            strongest=strongest,
            weakest=weakest,
            smh_pct=smh_pct,
            avg_sector=avg_sector,
            qqq_pct=qqq_pct,
        ),
        "P15": _build_p15(
            catalysts_today,
            daily_driver_type,
            daily_driver,
            chip_selloff,
            smh_pct,
            qqq_pct,
        ),
    }
    return {
        "parts": parts,
        "scores": scores,
        "total": total,
        "bias": bias,
        "driver_type": daily_driver_type,
        "daily_driver": daily_driver,
        "catalysts_today": catalysts_today,
    }


def _build_p13(catalysts: list[dict[str, Any]]) -> dict[str, Any]:
    if not catalysts:
        return {
            "judgment": "Edge：NO",
            "confidence": None,
            "one_liner": "今天无重大宏观数据发布日，无明显 Edge",
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


def _build_p15(
    catalysts: list[dict[str, Any]],
    driver_type: str,
    driver: str,
    chip_selloff: bool,
    smh_pct: float | None,
    qqq_pct: float | None,
) -> dict[str, Any]:
    """Verifiable trigger scenarios — no vague 'Macro利好'."""
    catalyst = catalysts[0]["name"] if catalysts else None
    if catalyst == "NFP":
        body = "\n".join(
            [
                "Scenario A：若 NFP 弱于预期 + 10Y 下行 → Dow/价值走强，Growth 反弹",
                "Scenario B：若 NFP 符合预期但 SMH 续跌 → QQQ < 昨低，放弃追多",
                "Scenario C：若 QQQ 震荡（±0.25%）→ 不交易",
            ]
        )
        primary = "B" if chip_selloff else "A"
    elif catalyst:
        body = "\n".join(
            [
                f"Scenario A：若 {catalyst} 弱于预期 + QQQ > 昨高 → Call",
                f"Scenario B：若 {catalyst} 强于预期 + QQQ < 昨低 → Put / 放弃",
                "Scenario C：若 QQQ 震荡 → 不交易",
            ]
        )
        primary = "A"
    elif chip_selloff:
        body = "\n".join(
            [
                "Scenario A：若 SMH 反弹 + QQQ > 昨高 → 短线 Call",
                "Scenario B：若 SMH 续跌 + QQQ < 昨低 → 放弃 / Put",
                "Scenario C：若 QQQ 震荡 → 不交易",
            ]
        )
        primary = "B"
    else:
        body = "\n".join(
            [
                "Scenario A：若 QQQ > 昨高 + SMH 领涨 → Call",
                "Scenario B：若 QQQ < 昨低 → Put / 放弃",
                "Scenario C：若 QQQ 震荡 → 不交易",
            ]
        )
        primary = "A" if (qqq_pct or 0) >= 0 else "B"
    return {
        "judgment": f"最可能 Scenario：{primary}",
        "confidence": 0.65,
        "one_liner": f"围绕 {driver_type}（{driver}）的可验证情景",
        "body_md": body,
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
    chip_selloff: bool = False,
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

    long_term = "Bullish Long-term (AI Expansion Regime)"
    today_view = "Bearish Today" if chip_selloff or ai_stance == "Bearish" else ai_stance

    return {
        "judgment": f"{long_term} · {today_view}",
        "confidence": 0.6 if mag7_rows else 0.4,
        "one_liner": one_liner,
        "body_md": "\n".join(
            body_lines
            + [
                f"- **长期 Regime**：{long_term}",
                f"- **今日 AI 表现**：{today_view}",
            ]
        ),
        "scores": {"ai_stance_signals": ai_signals},
    }
