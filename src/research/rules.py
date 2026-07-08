from __future__ import annotations

import json
from datetime import date, time, timedelta
from typing import Any, Literal

from src.research.driver_tree import build_driver_tree, driver_tree_display, primary_from_tree
from src.research.edges import compute_edges, format_p13_from_edges
from src.research.level_sources import format_if_level
from src.research.macro_calendar import (
    build_macro_calendar,
    catalysts_for_edges,
    catalysts_on_date,
    extract_catalysts,
    top_catalyst_name,
)
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

def _extract_catalysts(
    calendar: list[dict[str, Any]],
    *,
    target_date: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible alias — prefer macro_calendar.extract_catalysts."""
    return extract_catalysts(calendar, target_date=target_date)


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
    as_of_et: time | Literal["now"] | None = None,
) -> float | None:
    if trading_day is not None:
        pct = session_change_pct(
            ticker, raw, prior_raw, trading_day, section=section, as_of_et=as_of_et
        )
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


def compute_rule_parts(
    raw: dict[str, Any],
    *,
    as_of_et: time | Literal["now"] | None = None,
) -> dict[str, Any]:
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

    dxy_pct = _session_pct("DX-Y.NYB", raw, prior_raw, trading_day, dxy, section="market", as_of_et=as_of_et)
    dxy_dir = _score_direction(dxy_pct)
    dollar_score = 0
    if dxy_dir == "Up":
        dollar_score = -1
    elif dxy_dir == "Down":
        dollar_score = 1
    dollar_qqq = (
        "Negative" if dollar_score < 0 else "Positive" if dollar_score > 0 else "Neutral"
    )

    vix_pct = _session_pct("^VIX", raw, prior_raw, trading_day, vix, section="market", as_of_et=as_of_et)
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
                    name, raw, prior_raw, trading_day, q, section="sector", as_of_et=as_of_et
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

    spy_pct = _session_pct("SPY", raw, prior_raw, trading_day, spy_q, section="market", as_of_et=as_of_et)
    qqq_pct = _session_pct("QQQ", raw, prior_raw, trading_day, qqq_q, section="market", as_of_et=as_of_et)
    dow_pct = _session_pct("DIA", raw, prior_raw, trading_day, dia_q, section="market", as_of_et=as_of_et)
    smh = sector_quotes.get("SMH") or {}
    xlk = sector_quotes.get("XLK") or {}
    smh_pct = _session_pct("SMH", raw, prior_raw, trading_day, smh, section="sector", as_of_et=as_of_et)

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
    macro_calendar = build_macro_calendar(
        raw, trading_date=trading_date or None, prior_raw=prior_raw
    )
    catalysts_today = catalysts_for_edges(macro_calendar)
    catalyst_today = bool(macro_calendar.get("has_material_catalyst"))
    macro_driver = top_catalyst_name(macro_calendar) or _daily_macro_driver(catalysts_today)

    from src.utils.news_signals import extract_news_signals

    news_signals = extract_news_signals(raw.get("news"))
    nvda_q = (raw.get("stocks") or {}).get("quotes", {}).get("NVDA") or {}
    chip_selloff = _chip_selloff_signal(
        smh_pct,
        _session_pct("NVDA", raw, prior_raw, trading_day, nvda_q, section="stocks", as_of_et=as_of_et),
        int(news_signals.get("news_ai_mentions") or 0),
        int(news_signals.get("news_macro_mentions") or 0),
    )
    driver_tree = build_driver_tree(
        macro_calendar=macro_calendar,
        chip_selloff=chip_selloff,
        smh_pct=smh_pct,
        strongest_sector=strongest,
        weakest_sector=weakest,
        qqq_pct=qqq_pct,
        news_signals=news_signals,
    )
    daily_driver_type, daily_driver = primary_from_tree(driver_tree)
    # Backward compat when tree is empty
    if daily_driver == "No dominant catalyst" and not macro_calendar.get("catalysts"):
        daily_driver_type, daily_driver = _daily_driver_type_and_driver(
            catalysts_today, chip_selloff, smh_pct, strongest, qqq_pct
        )
        driver_tree["primary"] = {
            "type": daily_driver_type,
            "label": daily_driver,
            "evidence": "Legacy single-driver fallback",
            "score": 40,
        }

    pre_holiday = _pre_holiday(raw)

    stock_quotes = (raw.get("stocks") or {}).get("quotes") or {}
    nvda_pct = _session_pct("NVDA", raw, prior_raw, trading_day, nvda_q, section="stocks", as_of_et=as_of_et)

    ai_score = 0
    if smh_pct is not None and smh_pct > 0:
        ai_score += 1
    if nvda_pct is not None and nvda_pct > 0:
        ai_score += 1

    mag7_syms = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]
    mag7_rows: list[dict[str, Any]] = []
    for sym in mag7_syms:
        q = stock_quotes.get(sym) or {}
        pct = _session_pct(sym, raw, prior_raw, trading_day, q, section="stocks", as_of_et=as_of_et)
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

    tsla_q = stock_quotes.get("TSLA") or {}
    tsla_pct = _session_pct("TSLA", raw, prior_raw, trading_day, tsla_q, section="stocks", as_of_et=as_of_et)
    edges = compute_edges(
        raw,
        catalysts_today=catalysts_today,
        macro_calendar=macro_calendar,
        qqq_pct=qqq_pct,
        smh_pct=smh_pct,
        spy_pct=spy_pct,
        sym_pcts={
            "QQQ": qqq_pct,
            "SMH": smh_pct,
            "SPY": spy_pct,
            "NVDA": nvda_pct,
            "TSLA": tsla_pct,
        },
        driver_type=daily_driver_type,
    )
    p13_part = format_p13_from_edges(edges)
    p13_part["catalysts"] = catalysts_today
    p13_part["macro_calendar"] = macro_calendar

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
            "judgment": f"Primary：{daily_driver_type} · {daily_driver}",
            "driver_type": daily_driver_type,
            "driver": daily_driver,
            "driver_tree": driver_tree,
            "confidence": 0.75 if catalyst_today else 0.6,
            "one_liner": driver_tree_display(driver_tree),
            "body_md": _format_driver_tree_md(driver_tree, macro_calendar),
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
            xlk_pct=_session_pct("XLK", raw, prior_raw, trading_day, xlk, section="sector", as_of_et=as_of_et),
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
        "P13": p13_part,
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
        "P16": _build_p16(
            qqq_q=qqq_q,
            prior_raw=prior_raw,
            total=total,
            bias=bias,
            catalyst_today=catalyst_today,
            chip_selloff=chip_selloff,
            p16_gate_hint="Trade" if total >= 2 else "Wait",
            macro_calendar=macro_calendar,
            driver_tree=driver_tree,
        ),
    }
    return {
        "parts": parts,
        "scores": scores,
        "total": total,
        "bias": bias,
        "driver_type": daily_driver_type,
        "daily_driver": daily_driver,
        "driver_tree": driver_tree,
        "macro_calendar": macro_calendar,
        "trade_plan": parts["P16"].get("trade_plan"),
        "catalysts_today": catalysts_today,
        "edges": edges,
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


def _format_driver_tree_md(
    driver_tree: dict[str, Any],
    macro_calendar: dict[str, Any],
) -> str:
    lines = ["- **Driver Tree**（盘中可更新）"]
    for slot, title in (
        ("primary", "Primary"),
        ("secondary", "Secondary"),
        ("tertiary", "Tertiary"),
    ):
        node = driver_tree.get(slot)
        if node:
            lines.append(
                f"  - **{title}** [{node.get('type')}]: {node.get('label')} — {node.get('evidence', '')}"
            )
    if macro_calendar.get("headline"):
        lines.append(f"- **Macro Calendar**：{macro_calendar['headline']}")
    if macro_calendar.get("summary"):
        lines.append(f"- **催化剂摘要**：{macro_calendar['summary']}")
    return "\n".join(lines)


def _build_p16(
    *,
    qqq_q: dict[str, Any],
    prior_raw: dict[str, Any],
    total: int,
    bias: str,
    catalyst_today: bool,
    chip_selloff: bool,
    p16_gate_hint: str,
    macro_calendar: dict[str, Any] | None = None,
    driver_tree: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trading plan with labeled IF/THEN levels and explicit LONG/SHORT/WAIT blocks."""
    prior_qqq = ((prior_raw.get("market") or {}).get("quotes") or {}).get("QQQ") or {}
    prev_high = prior_qqq.get("high") or qqq_q.get("high")
    prev_low = prior_qqq.get("low") or qqq_q.get("low")
    prior_close = prior_qqq.get("close") or qqq_q.get("prev_close")

    def _lvl(px: Any, src: str) -> str:
        try:
            return format_if_level(float(px), src)
        except (TypeError, ValueError):
            return "—"

    high_tag = _lvl(prev_high, "prev_high") if prev_high is not None else "—"
    low_tag = _lvl(prev_low, "prev_low") if prev_low is not None else "—"
    close_tag = _lvl(prior_close, "prior_close") if prior_close is not None else "—"

    macro_calendar = macro_calendar or {}
    driver_tree = driver_tree or {}
    primary = driver_tree.get("primary") or {}
    primary_label = str(primary.get("label") or "")

    conditions_long: list[str] = []
    conditions_short: list[str] = []
    conditions_wait: list[str] = []

    geo_day = any(
        c.get("category") == "breaking" and "geo" in str(c.get("name", "")).lower()
        for c in (macro_calendar.get("catalysts") or [])
    )
    fed_minutes = any(
        "FOMC Minutes" in str(c.get("name", ""))
        for c in (macro_calendar.get("catalysts") or [])
    )
    oil_day = any(c.get("category") == "commodity" for c in (macro_calendar.get("catalysts") or []))

    conditions_long.append(
        f"IF QQQ > {high_tag} AND breadth positive THEN LONG QQQ / Call（突破昨高）"
    )
    conditions_short.append(
        f"IF QQQ < {low_tag} AND risk-off confirmed THEN SHORT / Put（破位昨低）"
    )
    conditions_wait.append(
        f"IF QQQ between {low_tag} and {high_tag} THEN WAIT — no edge in range"
    )

    if geo_day:
        conditions_wait.insert(
            0,
            "IF geopolitical headlines escalate AND VIX spikes THEN WAIT — reduce size, no chase",
        )
        conditions_long.append(
            "IF de-escalation headline + QQQ reclaims prior high THEN LONG risk-on bounce"
        )
        conditions_short.append(
            "IF oil spike + SMH breaks prior low THEN SHORT SMH / defensive hedge"
        )

    if fed_minutes:
        conditions_wait.insert(
            0 if not geo_day else 1,
            "IF before 14:00 ET AND QQQ inside prior range THEN WAIT for FOMC Minutes",
        )
        conditions_long.append(
            f"IF post-Minutes dovish + QQQ > {high_tag} THEN LONG（数据利好突破）"
        )
        conditions_short.append(
            f"IF post-Minutes hawkish + QQQ < {low_tag} THEN SHORT / exit longs"
        )

    if oil_day:
        conditions_wait.append("IF oil whipsaw ±2% intraday THEN WAIT — energy volatility")
        conditions_long.append("IF oil stabilizes + XLE fades AND QQQ holds THEN LONG growth")
        conditions_short.append("IF oil breaks higher + semis weak THEN SHORT SMH")

    if chip_selloff:
        conditions_long.append(
            f"IF SMH reclaims strength + QQQ > {high_tag} THEN SHORT-cover / tactical LONG"
        )
        conditions_short.append(
            f"IF SMH continues lower + QQQ < {low_tag} THEN avoid dip-buy; SHORT semis"
        )
        conditions_wait.append("IF chip selloff continues without QQQ breakdown THEN WAIT")

    if catalyst_today and not fed_minutes:
        conditions_wait.insert(
            0,
            "IF major data not yet released THEN WAIT — no pre-release index bets",
        )

    gate = p16_gate_hint
    if chip_selloff and total < 2:
        gate = "Wait"
    elif catalyst_today and geo_day:
        gate = "Wait"
    elif catalyst_today:
        gate = p16_gate_hint
    else:
        gate = "Trade" if total >= 2 and "bull" in bias.lower() else "Wait"

    if total <= -2:
        gate = "No Trade"

    lines = conditions_long[:2] + conditions_short[:1] + conditions_wait[:1]

    trade_plan = {
        "gate": gate,
        "conditions_long": conditions_long,
        "conditions_short": conditions_short,
        "conditions_wait": conditions_wait,
        "primary_driver": primary_label,
        "macro_headline": macro_calendar.get("headline"),
    }

    body_sections = [
        "**Conditions to go LONG**",
        *[f"- {c}" for c in conditions_long],
        "",
        "**Conditions to go SHORT**",
        *[f"- {c}" for c in conditions_short],
        "",
        "**Conditions to WAIT / No Trade**",
        *[f"- {c}" for c in conditions_wait],
    ]

    return {
        "judgment": f"计划：{gate}",
        "confidence": 0.65,
        "one_liner": conditions_wait[0] if conditions_wait else lines[0],
        "body_md": "\n".join(body_sections),
        "trade_plan": trade_plan,
        "levels": {
            "qqq_prev_high": prev_high,
            "qqq_prev_low": prev_low,
            "qqq_prior_close": prior_close,
        },
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
