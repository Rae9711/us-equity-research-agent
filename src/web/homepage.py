"""Homepage helpers: decision card, hypothesis accuracy, relative strength."""

from __future__ import annotations

import json
import re
from datetime import date as date_type
from typing import Any

from src.db import ConclusionRecord, MarketCase
from src.db.session import get_session
from src.utils.paths import morning_json_path, raw_data_path, reports_dir, step_json_path
from src.utils.quote_resolve import session_change_pct
from src.utils.trading_calendar import prior_trading_day
from src.collectors.macro_releases import (
    event_config_by_id,
    load_macro_release_config,
    load_releases,
    releases_file_path,
    scheduled_event_configs_for_date,
)
from src.collectors.release_market_reaction import (
    compute_measured_reaction,
    format_measured_reaction,
)
from src.utils.news_signals import detect_high_signal_news
from src.utils.driver_match import driver_hit

# Driver keyword → tradable symbols (vs QQQ benchmark)
DRIVER_SYMBOL_MAP: dict[str, list[str]] = {
    "ai": ["SMH", "NVDA", "AMD"],
    "semiconductor": ["SMH", "NVDA", "AMD"],
    "macro": ["TLT", "IEF", "UUP"],
    "fed": ["TLT", "IEF", "UUP"],
    "fomc": ["TLT", "IEF", "UUP"],
    "bond": ["TLT", "IEF", "UUP"],
    "rate": ["TLT", "IEF", "UUP"],
    "inflation": ["TLT", "UUP", "XLE"],
    "cpi": ["TLT", "UUP", "XLE"],
    "employment": ["XLF", "SPY", "QQQ"],
    "nonfarm": ["XLF", "SPY", "QQQ"],
    "oil": ["XLE", "USO", "QQQ"],
    "energy": ["XLE", "USO", "QQQ"],
    "geo": ["XLE", "UUP", "QQQ"],
    "dollar": ["UUP", "DXY", "QQQ"],
    "risk": ["TQQQ", "SPY", "QQQ"],
    "liquidity": ["TQQQ", "SPY", "QQQ"],
}
DEFAULT_SYMBOLS = ["SMH", "NVDA", "SPY"]


def _format_data_as_of_et(iso_ts: str | None) -> str | None:
    """Format morning.json generated_at as 'HH:MM ET' for the decision card."""
    if not iso_ts:
        return None
    try:
        from datetime import datetime

        from pytz import timezone

        et = timezone("America/New_York")
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = et.localize(dt)
        else:
            dt = dt.astimezone(et)
        return dt.strftime("%H:%M ET")
    except (TypeError, ValueError):
        return iso_ts[:16] if iso_ts else None


def load_morning(trading_date: str) -> dict[str, Any] | None:
    path = morning_json_path(trading_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_step4(trading_date: str) -> dict[str, Any] | None:
    path = step_json_path(4, trading_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_step3(trading_date: str) -> dict[str, Any] | None:
    path = step_json_path(3, trading_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_session_trade_update(trading_date: str) -> dict[str, Any] | None:
    """Step 3 mid-session re-rank payload (session_trade_update / trade_reeval)."""
    step3 = load_step3(trading_date)
    if not step3:
        return None
    return step3.get("session_trade_update") or step3.get("trade_reeval")


# Backward-compatible name used by earlier WIP
_load_trade_reeval = _load_session_trade_update


def _part_text(part: dict[str, Any] | None) -> str:
    if not part:
        return ""
    return " ".join(
        str(part.get(k) or "")
        for k in ("judgment", "one_liner", "body_md")
    ).strip()


def _normalize_driver_key(driver: str) -> str:
    d = driver.lower()
    for key in DRIVER_SYMBOL_MAP:
        if key in d:
            return key
    return "default"


def _extract_driver(morning: dict[str, Any]) -> dict[str, str]:
    parts = morning.get("parts") or {}
    p10 = parts.get("P10") or {}
    driver_type = (
        p10.get("driver_type")
        or morning.get("driver_type")
        or ""
    ).strip()
    driver = (
        p10.get("driver")
        or morning.get("daily_driver")
        or ""
    ).strip()

    if not driver:
        judgment = p10.get("judgment") or p10.get("one_liner") or ""
        if judgment and judgment not in ("—", "N/A"):
            cleaned = re.sub(r"^(Driver[：:]\s*)", "", judgment, flags=re.I).strip()
            m_type = re.search(r"Type[：:]\s*([^·•]+)", cleaned, re.I)
            m_drv = re.search(r"Driver[：:]\s*(.+)$", cleaned, re.I)
            if m_type:
                driver_type = driver_type or m_type.group(1).strip()
            if m_drv:
                driver = m_drv.group(1).strip()
            elif not driver:
                driver = cleaned[:120]

    if not driver:
        hyp = morning.get("hypothesis") or {}
        stmt = (hyp.get("statement") or "").strip()
        if stmt:
            driver = stmt[:120]

    display = " — ".join(x for x in (driver_type, driver) if x) or "—"
    return {
        "driver_type": driver_type or "—",
        "driver": driver or "—",
        "display": display,
    }


def _extract_bias(morning: dict[str, Any]) -> str:
    bias = morning.get("bias")
    total = morning.get("total_score")
    parts = morning.get("parts") or {}
    p11 = parts.get("P11") or {}
    if bias is None and p11.get("judgment"):
        m = re.search(r"Bias[：:]\s*([^·]+)", p11["judgment"])
        if m:
            bias = m.group(1).strip()
    if total is None and p11.get("judgment"):
        m = re.search(r"Total[：:]\s*([+-]?\d+)", p11["judgment"])
        if m:
            try:
                total = int(m.group(1))
            except ValueError:
                pass

    if bias is None:
        return "—"
    if total is not None:
        return f"{bias} · Total {total:+d}"
    return str(bias)


def _resolve_p16_gate(morning: dict[str, Any]) -> str:
    """P16 index gate: Trade | Wait | No Trade (from morning artifacts or recompute)."""
    best = morning.get("best_opportunity") or {}
    if best.get("p16_gate") in ("Trade", "Wait", "No Trade"):
        return str(best["p16_gate"])
    bt = morning.get("best_trades") or {}
    if bt.get("p16_gate") in ("Trade", "Wait", "No Trade"):
        return str(bt["p16_gate"])
    from src.research.trade_candidates import _p16_gate

    total = morning.get("total_score")
    try:
        total_i = int(total) if total is not None else 0
    except (TypeError, ValueError):
        total_i = 0
    return _p16_gate(morning.get("parts") or {}, total_i)


def _has_stock_setup(morning: dict[str, Any], step4: dict[str, Any] | None = None) -> bool:
    primary = (morning.get("best_trades") or {}).get("primary")
    if primary and primary.get("direction") in ("LONG", "SHORT"):
        return True
    if step4 and step4.get("stock_trade"):
        return True
    return False


def _trade_action(morning: dict[str, Any], step4: dict[str, Any] | None) -> str:
    """Homepage badge: Trade | Wait | No Trade.

    P16 Wait / No Trade closes the whole book (index and stocks). Cash is a
    valid day — do not badge Watch and imply a stock fill is still on.
    """
    p16_gate = _resolve_p16_gate(morning)
    if p16_gate in ("No Trade", "Wait"):
        return "No Trade"

    if step4 is not None:
        index_open = bool(
            step4.get("index_trade") and step4.get("index_trade") != "NO TRADE"
        )
        if step4.get("should_trade") or step4.get("stock_trade") or index_open:
            return "Trade"
        return "No Trade"

    primary = (morning.get("best_trades") or {}).get("primary")
    if primary and primary.get("direction") in ("LONG", "SHORT"):
        return "Trade"

    parts = morning.get("parts") or {}
    p16_text = _part_text(parts.get("P16")).lower()
    if any(w in p16_text for w in ("不交易", "放弃", "不追", "no trade", "hold off")):
        return "No Trade"
    if any(w in p16_text for w in ("买", "call", "做多", "trade", "入场")):
        return "Trade"

    total = morning.get("total_score")
    if total is None:
        p11 = parts.get("P11") or {}
        m = re.search(r"Total[：:]\s*([+-]?\d+)", p11.get("judgment") or "")
        if m:
            try:
                total = int(m.group(1))
            except ValueError:
                total = 0
        else:
            total = 0

    edge = (parts.get("P13") or {}).get("judgment") or ""
    edges = (parts.get("P13") or {}).get("edges") or {}
    any_edge = any(
        (edges.get(k) or {}).get("edge") == "YES"
        for k in ("macro_edge", "index_edge", "sector_edge", "stock_edge")
    )
    if total >= 2 and (any_edge or "Edge：YES" in edge or "/4 YES" in edge):
        return "Trade"
    if total >= 0:
        return "Wait"
    return "No Trade"


def _extract_invalidation(morning: dict[str, Any]) -> str:
    parts = morning.get("parts") or {}
    p15 = parts.get("P15") or {}
    body = p15.get("body_md") or p15.get("one_liner") or ""
    inv_lines: list[str] = []
    for line in body.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.search(r"Scenario\s*[BC]\b", line, re.I):
            inv_lines.append(re.sub(r"^[-*•]\s*", "", line))
    if inv_lines:
        return " · ".join(inv_lines[:2])[:240]

    hyp = morning.get("hypothesis") or {}
    counter = hyp.get("counter_evidence") or []
    if counter:
        return "；".join(str(c) for c in counter[:2])[:240]

    p15_one = (p15.get("one_liner") or "").strip()
    if p15_one and p15_one not in ("—", "N/A"):
        return p15_one[:240]
    return "—"


def _watch_variables(morning: dict[str, Any], driver: str) -> list[str]:
    parts = morning.get("parts") or {}
    hyp = morning.get("hypothesis") or {}
    variables: list[str] = []

    p3_one = (parts.get("P3") or {}).get("one_liner") or ""
    if p3_one and p3_one not in ("—", "N/A"):
        variables.append(p3_one[:90])

    driver_key = _normalize_driver_key(driver)
    defaults = {
        "ai": ["NVDA 相对强弱", "SMH vs QQQ"],
        "semiconductor": ["NVDA 相对强弱", "SMH vs QQQ"],
        "macro": ["10Y 收益率", "DXY / UUP"],
        "fed": ["Fed 政策预期", "10Y 利率"],
        "bond": ["10Y 收益率", "TLT / IEF"],
        "oil": ["XLE 能源板块", "油价 proxy"],
        "energy": ["XLE 能源板块", "USO"],
        "employment": ["非农/就业数据", "XLF 金融"],
        "default": ["QQQ 方向", "VIX 波动"],
    }
    for v in defaults.get(driver_key, defaults["default"]):
        if len(variables) >= 2:
            break
        if v not in variables:
            variables.append(v)

    for e in hyp.get("evidence") or []:
        if len(variables) >= 2:
            break
        s = str(e)[:80]
        if s and s not in variables:
            variables.append(s)

    return variables[:2] if variables else ["—", "—"]


def build_decision_card(trading_date: str) -> dict[str, Any] | None:
    """Build Executive Summary decision card from morning JSON (+ Step 4 if available)."""
    d = date_type.fromisoformat(trading_date)
    from src.utils.trading_calendar import is_trading_day

    if not is_trading_day(d):
        return None

    from src.web.steps_status import step0_available

    if not step0_available(trading_date):
        return None

    morning = load_morning(trading_date)
    if not morning:
        return None

    step4 = load_step4(trading_date)
    driver_info = _extract_driver(morning)
    hyp = morning.get("hypothesis") or {}
    p17 = (morning.get("parts") or {}).get("P17") or {}

    hypothesis_line = (hyp.get("statement") or p17.get("one_liner") or "—").strip()
    if hyp.get("id"):
        hypothesis_line = f"{hyp['id']} · {hypothesis_line}"

    exec_sum = morning.get("executive_summary") or {}
    best = morning.get("best_opportunity") or {}
    best_trades = morning.get("best_trades") or {}
    primary = best_trades.get("primary")
    bias_raw = morning.get("bias") or "—"
    from src.research.trade_candidates import _bias_stars

    bias_stars = exec_sum.get("bias_stars") or _bias_stars(str(bias_raw))

    if not exec_sum and best:
        from src.research.trade_candidates import build_executive_summary

        exec_sum = build_executive_summary(
            bias=bias_raw,
            bias_stars=bias_stars,
            driver_type=driver_info["driver_type"],
            driver=driver_info["driver"],
            best=best,
            best_trades=best_trades,
        )

    # P16 Wait / No Trade always wins over a leftover stock primary.
    trade_action = _trade_action(morning, step4)
    p16_gate = _resolve_p16_gate(morning)
    if p16_gate in ("No Trade", "Wait"):
        trade_action = "No Trade"
    elif primary and primary.get("direction") in ("LONG", "SHORT"):
        trade_action = "Trade"
    elif best_trades.get("threshold_message"):
        trade_action = "Wait"
    elif best.get("direction") == "NO TRADE" and not primary:
        trade_action = "No Trade"
    elif best.get("direction") in ("LONG", "SHORT") and trade_action == "Wait":
        trade_action = "Trade"

    transparency = morning.get("transparency") or {}
    index_trade = morning.get("index_trade") or best_trades.get("index_trade")

    entry_zone = exec_sum.get("entry_zone") or (primary or {}).get("entry_zone")
    entry_price = (
        exec_sum.get("entry_price")
        or (primary or {}).get("entry_price")
        or best.get("entry_price")
    )
    direction = (
        exec_sum.get("direction")
        or (primary or {}).get("direction")
        or best.get("direction")
    )
    stop_price = (
        exec_sum.get("stop_price")
        or (primary or {}).get("stop_price")
        or best.get("stop_price")
    )

    # Live Entry Status vs frozen Ideal Entry (raw/session quote when available)
    from src.research.entry_status import (
        attach_entry_status,
        classify_entry_status,
        infer_session_phase,
        resolve_symbol_last,
    )

    phase = infer_session_phase(trading_date)
    primary_sym = (primary or {}).get("symbol") or best.get("symbol")
    live_price = resolve_symbol_last(str(primary_sym or ""), trading_date) if primary_sym else None
    anchors = (primary or {}).get("level_anchors") or {}
    if not isinstance(anchors, dict):
        anchors = {
            "vwap": getattr(anchors, "vwap", None),
            "orb_high": getattr(anchors, "orb_high", None),
            "orb_low": getattr(anchors, "orb_low", None),
        }
    target_price = (
        exec_sum.get("target_price")
        or (primary or {}).get("target_price")
        or best.get("target_price")
    )
    entry_status = classify_entry_status(
        direction=str(direction or ""),
        current_price=live_price if live_price is not None else (primary or {}).get("current_price"),
        entry_zone=entry_zone,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        session_phase=phase,
        vwap=anchors.get("vwap"),
        orb_high=anchors.get("orb_high"),
        orb_low=anchors.get("orb_low"),
    )

    top_trades = list(transparency.get("top_trades") or morning.get("top_trades") or [])
    refreshed_top: list[dict[str, Any]] = []
    for t in top_trades:
        row = dict(t)
        sym = row.get("symbol")
        px = resolve_symbol_last(str(sym or ""), trading_date) if sym else None
        attach_entry_status(row, current_price=px, session_phase=phase)
        refreshed_top.append(row)

    primary_out = dict(primary) if primary else None
    if primary_out is not None:
        primary_out["entry_status"] = entry_status
        if live_price is not None:
            primary_out["current_price"] = live_price

    return {
        "driver": driver_info["display"],
        "driver_type": exec_sum.get("driver_type") or driver_info["driver_type"],
        "driver_label": exec_sum.get("driver") or driver_info["driver"],
        "bias": f"{bias_raw} {bias_stars}".strip() if bias_stars else _extract_bias(morning),
        "bias_stars": bias_stars,
        "best_trade": exec_sum.get("best_trade") or (
            f"{best.get('symbol', '—')} · {best.get('direction', '—')} · "
            f"{best.get('instrument', '—')}"
        ),
        "confidence": exec_sum.get("confidence") or best.get("confidence"),
        "entry": exec_sum.get("entry") or best.get("entry", "—"),
        "entry_source": exec_sum.get("entry_source") or (primary or {}).get("entry_source"),
        "entry_price": entry_price,
        "stop": exec_sum.get("stop") or best.get("stop", "—"),
        "stop_source": exec_sum.get("stop_source") or (primary or {}).get("stop_source"),
        "stop_price": stop_price,
        "target": exec_sum.get("target") or best.get("target", "—"),
        "target_source": exec_sum.get("target_source") or (primary or {}).get("target_source"),
        "target_price": exec_sum.get("target_price") or (primary or {}).get("target_price") or best.get("target_price"),
        "target_action": exec_sum.get("target_action") or (primary or {}).get("target_action") or best.get("target_action"),
        "direction": direction,
        "horizon": (primary or {}).get("horizon") or best.get("horizon") or best.get("duration"),
        "why_chain": exec_sum.get("why_chain") or best.get("why_chain", "—"),
        "why_factors": exec_sum.get("why_factors") or (primary or {}).get("why_factors") or [],
        "why_vs_runner_up": exec_sum.get("why_vs_runner_up") or (primary or {}).get("why_vs_runner_up"),
        "edge_type": exec_sum.get("edge_type") or (primary or {}).get("edge_type"),
        "win_prob": (primary or {}).get("win_prob") or best.get("win_prob"),
        "win_prob_breakdown": exec_sum.get("win_prob_breakdown") or (primary or {}).get("win_prob_breakdown"),
        "level_reasons": exec_sum.get("level_reasons") or (primary or {}).get("level_reasons"),
        "trade_economics": exec_sum.get("trade_economics") or (primary or {}).get("trade_economics"),
        "position_sizing": exec_sum.get("position_sizing") or (primary or {}).get("position_sizing"),
        "rr_display": exec_sum.get("rr_display") or (primary or {}).get("rr_display"),
        "entry_zone": entry_zone,
        "entry_status": entry_status,
        "remaining_er_pct": (entry_status or {}).get("remaining_er_pct"),
        "live_expected_return_pct": (entry_status or {}).get("live_expected_return_pct")
        or (entry_status or {}).get("remaining_er_pct"),
        "alternate_entry": (entry_status or {}).get("alternate_entry")
        or (entry_status or {}).get("new_plan"),
        "session_trade_update": _load_session_trade_update(trading_date),
        "trade_reeval": _load_session_trade_update(trading_date),
        "rank_summary": exec_sum.get("rank_summary") or (primary or {}).get("rank_summary"),
        "win_prob_source": (primary or {}).get("win_prob_source"),
        "similar_days": (primary or {}).get("similar_days") or transparency.get("similar_days"),
        "ev_distribution": (primary or {}).get("ev_distribution") or transparency.get("ev_distribution"),
        "calibration": (primary or {}).get("calibration") or transparency.get("calibration"),
        "why_wins_today": exec_sum.get("why_wins_today") or transparency.get("why_wins_today") or [],
        "why_not_alternatives": exec_sum.get("why_not_alternatives") or transparency.get("why_not_alternatives") or [],
        "todays_opportunities": transparency.get("todays_opportunities") or [],
        "index_rejection_reasons": transparency.get("index_rejection_reasons") or [],
        "day_risks": transparency.get("day_risks") or {},
        "index_trade": index_trade,
        "expected_return_pct": exec_sum.get("expected_return_pct") or (primary or {}).get("expected_return_pct") or best.get("expected_return_pct"),
        "expected_move": exec_sum.get("expected_move") or (primary or {}).get("expected_move") or best.get("expected_move"),
        "return_calculation": exec_sum.get("return_calculation") or (primary or {}).get("return_calculation") or best.get("return_calculation"),
        "trade_summary_cn": exec_sum.get("trade_summary_cn") or (primary or {}).get("trade_summary_cn") or best.get("trade_summary_cn"),
        "risk_reward": (primary or {}).get("risk_reward") or best.get("risk_reward"),
        "score_formula_display": (primary or {}).get("score_formula_display") or best.get("score_formula_display"),
        "confidence_stars": (primary or {}).get("confidence_stars"),
        "one_liner": exec_sum.get("one_liner") or best.get("one_liner", "—"),
        "trade_action": trade_action,
        "invalidation": _extract_invalidation(morning),
        "watch_variables": _watch_variables(morning, driver_info["display"]),
        "hypothesis": hypothesis_line[:200],
        "has_morning": True,
        "generated_at": morning.get("generated_at"),
        "decision_as_of": morning.get("decision_as_of"),
        "data_as_of_et": morning.get("decision_as_of_et")
        or _format_data_as_of_et(morning.get("decision_as_of"))
        or _format_data_as_of_et(morning.get("generated_at")),
        "advisory": "ADVISORY — 不构成投资建议",
        "trade_candidates": morning.get("trade_candidates") or [],
        "best_opportunity": best,
        "primary_trade": primary_out,
        "swing_trade": morning.get("swing_trade")
        or best_trades.get("swing")
        or None,
        "macro_calendar": morning.get("macro_calendar") or transparency.get("macro_calendar"),
        "driver_tree": morning.get("driver_tree") or (morning.get("parts") or {}).get("P10", {}).get("driver_tree"),
        "trade_plan": morning.get("trade_plan") or transparency.get("trade_plan"),
        "top_trades": refreshed_top,
        "watchlist": transparency.get("watchlist") or morning.get("watchlist") or [],
    }


def build_holiday_news_brief(trading_date: str) -> dict[str, Any]:
    """Lightweight news snapshot for NYSE closed days — no trade decisions."""
    from src.utils.trading_calendar import is_trading_day, prior_trading_day

    d = date_type.fromisoformat(trading_date)
    if is_trading_day(d):
        return {"has_data": False, "headline_count": 0, "headlines": []}

    def _from_raw(raw: dict[str, Any], source_date: str, source_label: str) -> dict[str, Any]:
        news = raw.get("news") or {}
        polygon = news.get("polygon") or []
        rss = news.get("rss") or []
        headlines: list[dict[str, str | None]] = []
        for article in polygon[:5]:
            title = article.get("title")
            if title:
                headlines.append(
                    {"title": str(title), "url": article.get("url"), "source": "polygon"}
                )
        for article in rss[:3]:
            title = article.get("title")
            if title:
                headlines.append(
                    {
                        "title": str(title),
                        "url": article.get("link"),
                        "source": str(article.get("source") or "rss"),
                    }
                )
        total = len(polygon) + len(rss)
        return {
            "source_date": source_date,
            "source_label": source_label,
            "headline_count": total,
            "headlines": headlines,
            "has_data": total > 0,
        }

    holiday_raw = raw_data_path(trading_date)
    if holiday_raw.exists():
        try:
            raw = json.loads(holiday_raw.read_text(encoding="utf-8"))
            brief = _from_raw(raw, trading_date, "休市日 Raw")
            if brief["has_data"]:
                return brief
        except Exception:
            pass

    prior = prior_trading_day(d)
    prior_raw = raw_data_path(prior.isoformat())
    if prior_raw.exists():
        try:
            raw = json.loads(prior_raw.read_text(encoding="utf-8"))
            return _from_raw(raw, prior.isoformat(), "上一交易日 Raw")
        except Exception:
            pass

    return {
        "source_date": prior.isoformat(),
        "source_label": "无快照",
        "headline_count": 0,
        "headlines": [],
        "has_data": False,
    }


def _hypothesis_label_for_date(date_str: str) -> str | None:
    """Best available P17 correctness label for a trading day."""
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            try:
                data = json.loads(row.case_json)
                label = (data.get("labels") or {}).get("hypothesis_correct")
                if label and label not in ("", "N/A"):
                    return label
            except Exception:
                pass

        day = date_type.fromisoformat(date_str)
        p17 = (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == day,
                ConclusionRecord.part_id == "P17",
            )
            .first()
        )
        if p17 and p17.verification and p17.verification not in ("", "N/A"):
            return p17.verification
    finally:
        session.close()

    step7_path = step_json_path(7, date_str)
    if step7_path.exists():
        try:
            s7 = json.loads(step7_path.read_text(encoding="utf-8"))
            label = (s7.get("hypothesis_correct") or (s7.get("extra") or {}).get("hypothesis_correct"))
            if label and label not in ("", "N/A"):
                return label
        except Exception:
            pass
    return None


def hypothesis_accuracy_series() -> dict[str, Any]:
    """
    Cumulative P17 hypothesis accuracy over labeled trading days.
    对 = correct; 部分对/错 = not correct (matches agent stats convention).
    """
    session = get_session()
    dates: set[str] = set()
    try:
        for row in session.query(MarketCase.date).all():
            dates.add(row.date)
        for row in session.query(ConclusionRecord.trading_date).filter(
            ConclusionRecord.part_id == "P17",
            ConclusionRecord.verification.isnot(None),
            ConclusionRecord.verification != "",
        ).all():
            dates.add(row.trading_date.isoformat())
    finally:
        session.close()

    points: list[dict[str, Any]] = []
    cum_correct = 0
    cum_total = 0
    for date_str in sorted(dates):
        label = _hypothesis_label_for_date(date_str)
        if not label or label == "N/A":
            continue
        cum_total += 1
        if label == "对":
            cum_correct += 1
        points.append(
            {
                "date": date_str,
                "label": label,
                "cumulative_accuracy": round(cum_correct / cum_total * 100, 1),
                "n": cum_total,
            }
        )

    return {
        "points": points,
        "total_labeled": cum_total,
        "accuracy_pct": round(cum_correct / cum_total * 100, 1) if cum_total else None,
        "correct": cum_correct,
        "sufficient": cum_total >= 2,
    }


def _actual_driver_for_date(date_str: str) -> str | None:
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            try:
                data = json.loads(row.case_json)
                actual = (data.get("labels") or {}).get("actual_driver")
                if actual and actual not in ("", "N/A", "Unknown"):
                    return actual
            except Exception:
                pass
    finally:
        session.close()

    step7_path = step_json_path(7, date_str)
    if step7_path.exists():
        try:
            s7 = json.loads(step7_path.read_text(encoding="utf-8"))
            actual = s7.get("actual_driver") or (s7.get("extra") or {}).get("actual_driver")
            if actual and actual not in ("", "N/A"):
                return actual
        except Exception:
            pass
    return None


def _morning_p10_driver(date_str: str) -> tuple[str, str | None]:
    morning = load_morning(date_str)
    if not morning:
        return "", None
    p10 = (morning.get("parts") or {}).get("P10") or {}
    driver = (p10.get("driver") or p10.get("judgment") or p10.get("one_liner") or "").strip()
    driver_type = p10.get("driver_type") or morning.get("driver_type")
    return driver, driver_type


def _labeled_trading_dates() -> list[str]:
    session = get_session()
    dates: set[str] = set()
    try:
        for row in session.query(MarketCase.date).all():
            dates.add(row.date)
        for row in session.query(ConclusionRecord.trading_date).filter(
            ConclusionRecord.part_id == "S7",
        ).all():
            dates.add(row.trading_date.isoformat())
    finally:
        session.close()

    for day_dir in reports_dir().iterdir():
        if day_dir.is_dir() and (day_dir / "step7.json").exists():
            dates.add(day_dir.name)
    return sorted(dates)


def driver_accuracy_series() -> dict[str, Any]:
    """Cumulative P10 driver vs S7 actual_driver hit rate (fuzzy match) — 市场归因对比."""
    points: list[dict[str, Any]] = []
    cum_correct = 0
    cum_total = 0

    for date_str in _labeled_trading_dates():
        actual = _actual_driver_for_date(date_str)
        morning_driver, morning_driver_type = _morning_p10_driver(date_str)
        if not actual or not morning_driver:
            continue
        hit = driver_hit(morning_driver, actual, morning_driver_type=morning_driver_type)
        cum_total += 1
        if hit:
            cum_correct += 1
        points.append(
            {
                "date": date_str,
                "hit": hit,
                "morning_driver": morning_driver[:80],
                "actual_driver": actual,
                "cumulative_accuracy": round(cum_correct / cum_total * 100, 1),
                "n": cum_total,
            }
        )

    return {
        "points": points,
        "total_labeled": cum_total,
        "accuracy_pct": round(cum_correct / cum_total * 100, 1) if cum_total else None,
        "correct": cum_correct,
        "sufficient": cum_total >= 2,
    }


def _p10_verification_for_date(trading_date: date_type) -> str | None:
    session = get_session()
    try:
        row = (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == trading_date,
                ConclusionRecord.part_id == "P10",
            )
            .first()
        )
        return row.verification if row and row.verification else None
    finally:
        session.close()


def _case_labels_for_date(date_str: str) -> dict[str, Any]:
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            try:
                data = json.loads(row.case_json)
                return dict(data.get("labels") or {})
            except Exception:
                pass
    finally:
        session.close()
    return {}


def agent_driver_accuracy_series() -> dict[str, Any]:
    """
    Agent Driver 判断质量 — P10 vs verify-corrected driver when P10 is 错/部分对,
    otherwise P10 vs S7 actual_driver when verify says 对.
    """
    from src.web.verify_learning import reference_driver_for_agent_quality

    points: list[dict[str, Any]] = []
    cum_correct = 0
    cum_total = 0

    for date_str in _labeled_trading_dates():
        morning_driver, morning_driver_type = _morning_p10_driver(date_str)
        if not morning_driver:
            continue

        trading_date = date_type.fromisoformat(date_str)
        p10_ver = _p10_verification_for_date(trading_date)
        if not p10_ver:
            continue

        case_labels = _case_labels_for_date(date_str)
        s7_actual = _actual_driver_for_date(date_str)
        reference = reference_driver_for_agent_quality(
            trading_date,
            date_str,
            p10_ver,
            case_labels,
            s7_actual,
        )
        if not reference:
            continue

        hit = driver_hit(morning_driver, reference, morning_driver_type=morning_driver_type)
        cum_total += 1
        if hit:
            cum_correct += 1
        points.append(
            {
                "date": date_str,
                "hit": hit,
                "morning_driver": morning_driver[:80],
                "reference_driver": reference,
                "p10_verification": p10_ver,
                "cumulative_accuracy": round(cum_correct / cum_total * 100, 1),
                "n": cum_total,
            }
        )

    return {
        "points": points,
        "total_labeled": cum_total,
        "accuracy_pct": round(cum_correct / cum_total * 100, 1) if cum_total else None,
        "correct": cum_correct,
        "sufficient": cum_total >= 2,
    }


def _scenario_labels_for_date(date_str: str) -> dict[str, Any] | None:
    labels: dict[str, Any] | None = None
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            try:
                data = json.loads(row.case_json)
                case_labels = data.get("labels") or {}
                if case_labels.get("scenario_actual") is not None:
                    labels = dict(case_labels)
            except Exception:
                pass

        if labels is None:
            step7_path = step_json_path(7, date_str)
            if step7_path.exists():
                try:
                    s7 = json.loads(step7_path.read_text(encoding="utf-8"))
                    extra = s7.get("extra") or {}
                    if extra.get("scenario_actual") is not None or s7.get("scenario_actual") is not None:
                        labels = {
                            "scenario_primary": extra.get("scenario_primary") or s7.get("scenario_primary"),
                            "scenario_actual": extra.get("scenario_actual") or s7.get("scenario_actual"),
                            "scenario_correct": extra.get("scenario_correct", s7.get("scenario_correct")),
                        }
                except Exception:
                    pass

        day = date_type.fromisoformat(date_str)
        p15 = (
            session.query(ConclusionRecord)
            .filter(
                ConclusionRecord.trading_date == day,
                ConclusionRecord.part_id == "P15",
            )
            .first()
        )
        if p15 and p15.verification and p15.verification not in ("", "N/A"):
            if labels is None:
                labels = {}
            labels["scenario_correct"] = p15.verification == "对"
    finally:
        session.close()

    return labels


def scenario_accuracy_series() -> dict[str, Any]:
    """Cumulative P15 primary scenario vs S7 actual scenario hit rate."""
    points: list[dict[str, Any]] = []
    cum_correct = 0
    cum_total = 0

    for date_str in _labeled_trading_dates():
        labels = _scenario_labels_for_date(date_str)
        if not labels:
            continue
        primary = labels.get("scenario_primary")
        actual = labels.get("scenario_actual")
        correct = labels.get("scenario_correct")
        if primary is None or actual is None or actual == "none":
            continue
        cum_total += 1
        hit = bool(correct)
        if hit:
            cum_correct += 1
        points.append(
            {
                "date": date_str,
                "hit": hit,
                "scenario_primary": primary,
                "scenario_actual": actual,
                "cumulative_accuracy": round(cum_correct / cum_total * 100, 1),
                "n": cum_total,
            }
        )

    return {
        "points": points,
        "total_labeled": cum_total,
        "accuracy_pct": round(cum_correct / cum_total * 100, 1) if cum_total else None,
        "correct": cum_correct,
        "sufficient": cum_total >= 2,
    }


def _bias_direction(morning: dict[str, Any]) -> str | None:
    bias = morning.get("bias")
    if not bias:
        parts = morning.get("parts") or {}
        p11 = parts.get("P11") or {}
        if p11.get("judgment"):
            m = re.search(r"Bias[：:]\s*([^·]+)", p11["judgment"])
            if m:
                bias = m.group(1).strip()
    if not bias:
        return None
    b = str(bias).lower()
    if any(w in b for w in ("bull", "多")):
        return "bull"
    if any(w in b for w in ("bear", "空")):
        return "bear"
    return "neutral"


def _qqq_chg_for_date(date_str: str) -> float | None:
    raw_path = raw_data_path(date_str)
    if not raw_path.exists():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    qqq_q = _quote_from_raw(raw, "QQQ")
    return _change_pct(qqq_q)


def directional_accuracy_series() -> dict[str, Any]:
    """Morning bias direction vs QQQ day return."""
    points: list[dict[str, Any]] = []
    cum_correct = 0
    cum_total = 0

    for date_str in _labeled_trading_dates():
        morning = load_morning(date_str)
        if not morning:
            continue
        direction = _bias_direction(morning)
        qqq_chg = _qqq_chg_for_date(date_str)
        if direction is None or qqq_chg is None or direction == "neutral":
            continue

        correct = (direction == "bull" and qqq_chg > 0) or (direction == "bear" and qqq_chg < 0)
        cum_total += 1
        if correct:
            cum_correct += 1
        points.append(
            {
                "date": date_str,
                "bias": direction,
                "qqq_chg": round(qqq_chg, 2),
                "correct": correct,
                "cumulative_accuracy": round(cum_correct / cum_total * 100, 1),
                "n": cum_total,
            }
        )

    return {
        "points": points,
        "total_labeled": cum_total,
        "accuracy_pct": round(cum_correct / cum_total * 100, 1) if cum_total else None,
        "correct": cum_correct,
        "sufficient": cum_total >= 2,
    }


def _quote_from_raw(raw: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Look up a symbol across market / sector / stocks quote sections."""
    sym = symbol.upper()
    if sym == "DXY":
        sym = "DX-Y.NYB"
    if sym == "UUP":
        sym = "DX-Y.NYB"  # proxy when UUP not collected

    for section_key in ("market", "sector", "stocks"):
        section = raw.get(section_key) or {}
        quotes = section.get("quotes") or {}
        if sym in quotes:
            return quotes[sym]
        for ticker, q in quotes.items():
            if str(ticker).upper() == sym:
                return q
    return {}


def _change_pct(q: dict[str, Any]) -> float | None:
    if not q or "error" in q:
        return None
    chg = q.get("change_pct")
    if chg is not None:
        try:
            return float(chg)
        except (TypeError, ValueError):
            pass
    close = q.get("close")
    prev = q.get("prev_close")
    if close is not None and prev not in (None, 0):
        try:
            return (float(close) - float(prev)) / abs(float(prev)) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def symbols_for_driver(driver: str) -> list[str]:
    key = _normalize_driver_key(driver)
    if key == "default":
        return list(DEFAULT_SYMBOLS)
    return DRIVER_SYMBOL_MAP.get(key, DEFAULT_SYMBOLS)


def _section_for_symbol(symbol: str) -> str:
    sym = symbol.upper()
    if sym in ("QQQ", "SPY", "DIA", "TQQQ", "^VIX", "DX-Y.NYB", "ES=F"):
        return "market"
    if sym in ("SMH", "XLK", "XLF", "XLE"):
        return "sector"
    return "stocks"


def _session_chg(
    symbol: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    trading_day: date_type,
) -> float | None:
    section = _section_for_symbol(symbol)
    return session_change_pct(
        symbol if symbol != "DXY" else "DX-Y.NYB",
        raw,
        prior_raw,
        trading_day,
        section=section,
    )


def build_relative_strength(trading_date: str, driver: str) -> dict[str, Any] | None:
    """Relative strength vs QQQ using session-aware quotes (not stale Step 0 bars)."""
    raw_path = raw_data_path(trading_date)
    if not raw_path.exists():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    trading_day = date_type.fromisoformat(trading_date)
    prior_raw: dict[str, Any] = {}
    prior_path = raw_data_path(prior_trading_day(trading_day).isoformat())
    if prior_path.exists():
        try:
            prior_raw = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    qqq_chg = _session_chg("QQQ", raw, prior_raw, trading_day)
    if qqq_chg is None:
        return None

    symbols = symbols_for_driver(driver)
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        sym_chg = _session_chg(sym, raw, prior_raw, trading_day)
        if sym_chg is None:
            rows.append(
                {
                    "symbol": sym,
                    "change_pct": None,
                    "vs_qqq": None,
                    "direction": "flat",
                    "available": False,
                }
            )
            continue
        vs = round(sym_chg - qqq_chg, 2)
        direction = "up" if vs > 0.05 else "down" if vs < -0.05 else "flat"
        rows.append(
            {
                "symbol": sym,
                "change_pct": round(sym_chg, 2),
                "vs_qqq": vs,
                "vs_qqq_display": f"{vs:+.2f}%",
                "direction": direction,
                "available": True,
            }
        )

    return {
        "benchmark": "QQQ",
        "benchmark_chg": round(qqq_chg, 2),
        "driver": driver,
        "symbols": symbols,
        "rows": rows,
    }


def _format_release_value(value: Any, unit: str | None = None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        if unit in ("%", "percent"):
            return f"{v:.1f}%"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.2f}"
    except (TypeError, ValueError):
        return str(value)


def _surprise_display(surprise_pct: float | None, direction: str | None) -> dict[str, str]:
    if surprise_pct is None:
        return {"text": "—", "class": "flat", "arrow": ""}
    arrow = "⬆️" if surprise_pct > 0 else "⬇️" if surprise_pct < 0 else ""
    # inflation: higher surprise = bad (red); growth/labor: higher = good (green)
    if direction == "inflation":
        css = "surprise-bad" if surprise_pct > 0 else "surprise-good" if surprise_pct < 0 else "flat"
    else:
        css = "surprise-good" if surprise_pct > 0 else "surprise-bad" if surprise_pct < 0 else "flat"
    return {"text": f"{surprise_pct:+.1f}% {arrow}".strip(), "class": css, "arrow": arrow}


def _status_icon(status: str) -> str:
    return {"released": "✅", "waiting": "⏳", "missing": "❌"}.get(status, "⏳")


def _status_hint(status: str) -> str:
    if status == "missing":
        return "fallback to news"
    return ""


def _consensus_value(release: dict[str, Any], event_cfg: dict[str, Any]) -> float | None:
    """Consensus from snapshot, else yaml consensus_default."""
    raw = release.get("consensus")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    default = event_cfg.get("consensus_default")
    if default in (None, "", "."):
        return None
    try:
        return float(default)
    except (TypeError, ValueError):
        return None


def _measured_reaction_display(
    trading_date: str,
    scheduled_et: str,
    status: str,
    snapshot_reaction: dict[str, Any] | None,
) -> str:
    if status != "released" or scheduled_et == "—":
        return "—"
    reaction = snapshot_reaction
    if not reaction:
        try:
            reaction = compute_measured_reaction(
                date_type.fromisoformat(trading_date),
                str(scheduled_et),
            )
        except Exception:
            reaction = None
    return format_measured_reaction(reaction)


def build_catalyst_status(trading_date: str) -> dict[str, Any]:
    """今日催化剂状态表（仅经济日历当日 scheduled 的宏观发布）。"""
    cfg = load_macro_release_config()
    by_id = event_config_by_id(cfg)
    scheduled = scheduled_event_configs_for_date(trading_date, cfg=cfg)

    snap = load_releases(trading_date)
    snap_releases = list((snap or {}).get("releases") or [])
    snap_by_event = {str(r.get("event")): r for r in snap_releases if r.get("event")}

    # Historical: releases.json may exist when calendar/raw is gone
    if not scheduled and snap_releases:
        seen: set[str] = set()
        for r in snap_releases:
            eid = str(r.get("event") or "")
            if eid and eid in by_id and eid not in seen:
                seen.add(eid)
                scheduled.append(by_id[eid])

    has_scheduled = bool(scheduled)
    if not has_scheduled and not snap_releases:
        return {
            "rows": [],
            "has_data": False,
            "has_scheduled": False,
            "show_section": False,
            "has_snapshot": releases_file_path(trading_date).exists(),
            "generated_at": None,
            "counts": {},
            "qqq_chg": _qqq_chg_for_date(trading_date),
        }

    qqq_chg = _qqq_chg_for_date(trading_date)

    rows: list[dict[str, Any]] = []
    for event_cfg in scheduled:
        event_id = str(event_cfg.get("id"))
        r = snap_by_event.get(event_id, {})
        status = str(r.get("status") or "waiting")
        unit = r.get("unit") or event_cfg.get("unit")
        consensus_num = _consensus_value(r, event_cfg)
        actual_num = r.get("actual")
        surprise_pct = r.get("surprise_pct")
        if surprise_pct is None and actual_num is not None and consensus_num is not None:
            try:
                surprise_pct = round(
                    (float(actual_num) - consensus_num) / abs(consensus_num) * 100, 2
                )
            except (TypeError, ValueError, ZeroDivisionError):
                surprise_pct = None
        direction = r.get("direction") or event_cfg.get("direction")
        surprise = _surprise_display(surprise_pct, direction)
        row_class = surprise["class"] if r.get("surprise_flag") else ""
        scheduled_et = r.get("scheduled_et") or event_cfg.get("scheduled_et") or "—"
        rows.append(
            {
                "event": r.get("label") or event_cfg.get("label") or event_id,
                "scheduled_et": scheduled_et,
                "status": status,
                "status_icon": _status_icon(status),
                "status_hint": _status_hint(status),
                "actual": _format_release_value(actual_num, unit),
                "consensus": _format_release_value(consensus_num, unit),
                "prior": _format_release_value(r.get("prior"), unit),
                "surprise": surprise["text"],
                "surprise_class": surprise["class"],
                "row_class": row_class,
                "typical_impact": r.get("market_impact") or event_cfg.get("market_impact_default") or "—",
                "measured_reaction": _measured_reaction_display(
                    trading_date, str(scheduled_et), status, r.get("measured_reaction")
                ),
                "source": r.get("source"),
            }
        )

    counts = (snap or {}).get("counts") or {}
    return {
        "rows": rows,
        "has_data": bool(rows),
        "has_scheduled": has_scheduled,
        "show_section": bool(rows),
        "has_snapshot": releases_file_path(trading_date).exists(),
        "generated_at": (snap or {}).get("generated_at"),
        "counts": counts,
        "qqq_chg": qqq_chg,
    }


def build_breaking_signals(trading_date: str) -> dict[str, Any]:
    """L2 突发信号 — 从 Step 3 或现场检测。"""
    signals: list[dict[str, Any]] = []

    s3_path = step_json_path(3, trading_date)
    if s3_path.exists():
        try:
            s3 = json.loads(s3_path.read_text(encoding="utf-8"))
            signals = list(s3.get("breaking_news_signals") or [])
        except Exception:
            pass

    if not signals:
        try:
            from src.collectors.news import collect_intraday_news
            from src.utils.trading_calendar import market_open_et
            from datetime import date as date_type

            d = date_type.fromisoformat(trading_date)
            bundle = collect_intraday_news(since=market_open_et(d))
            headlines = [
                {"title": h.get("title"), "published_utc": h.get("published_utc"), "url": h.get("url")}
                for h in (bundle.get("polygon") or [])[:30]
            ]
            signals = detect_high_signal_news(headlines)
        except Exception:
            signals = []

    high = [s for s in signals if s.get("severity") == "high"]
    medium = [s for s in signals if s.get("severity") == "medium"]

    return {
        "signals": signals[:12],
        "high": high[:8],
        "medium": medium[:6],
        "has_signals": bool(signals),
        "has_high": bool(high),
    }
