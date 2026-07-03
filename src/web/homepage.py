"""Homepage helpers: decision card, hypothesis accuracy, relative strength."""

from __future__ import annotations

import json
import re
from datetime import date as date_type
from typing import Any

from src.db import ConclusionRecord, MarketCase
from src.db.session import get_session
from src.utils.paths import morning_json_path, raw_data_path, reports_dir, step_json_path
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


def _extract_driver(morning: dict[str, Any]) -> str:
    parts = morning.get("parts") or {}
    p10 = parts.get("P10") or {}
    judgment = p10.get("judgment") or p10.get("one_liner") or ""
    if judgment and judgment not in ("—", "N/A"):
        cleaned = re.sub(r"^(Driver[：:]\s*)", "", judgment, flags=re.I).strip()
        return cleaned[:120] or judgment[:120]

    hyp = morning.get("hypothesis") or {}
    stmt = (hyp.get("statement") or "").strip()
    if stmt:
        return stmt[:120]
    return "—"


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


def _trade_action(morning: dict[str, Any], step4: dict[str, Any] | None) -> str:
    if step4 is not None:
        return "Trade" if step4.get("should_trade") else "No Trade"

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
    if total >= 2 and "Edge：YES" in edge:
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
    """Build 一页纸决策卡 from morning JSON (+ Step 4 if available)."""
    morning = load_morning(trading_date)
    if not morning:
        return None

    step4 = load_step4(trading_date)
    driver = _extract_driver(morning)
    hyp = morning.get("hypothesis") or {}
    p17 = (morning.get("parts") or {}).get("P17") or {}

    hypothesis_line = (hyp.get("statement") or p17.get("one_liner") or "—").strip()
    if hyp.get("id"):
        hypothesis_line = f"{hyp['id']} · {hypothesis_line}"

    return {
        "driver": driver,
        "bias": _extract_bias(morning),
        "trade_action": _trade_action(morning, step4),
        "invalidation": _extract_invalidation(morning),
        "watch_variables": _watch_variables(morning, driver),
        "hypothesis": hypothesis_line[:200],
        "has_morning": True,
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


def _morning_p10_driver(date_str: str) -> str:
    morning = load_morning(date_str)
    if not morning:
        return ""
    p10 = (morning.get("parts") or {}).get("P10") or {}
    return (p10.get("judgment") or p10.get("one_liner") or "").strip()


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
    """Cumulative P10 driver vs S7 actual_driver hit rate (fuzzy match)."""
    points: list[dict[str, Any]] = []
    cum_correct = 0
    cum_total = 0

    for date_str in _labeled_trading_dates():
        actual = _actual_driver_for_date(date_str)
        morning_driver = _morning_p10_driver(date_str)
        if not actual or not morning_driver:
            continue
        hit = driver_hit(morning_driver, actual)
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


def _scenario_labels_for_date(date_str: str) -> dict[str, Any] | None:
    session = get_session()
    try:
        row = session.query(MarketCase).filter(MarketCase.date == date_str).first()
        if row:
            try:
                data = json.loads(row.case_json)
                labels = data.get("labels") or {}
                if labels.get("scenario_actual") is not None:
                    return labels
            except Exception:
                pass
    finally:
        session.close()

    step7_path = step_json_path(7, date_str)
    if step7_path.exists():
        try:
            s7 = json.loads(step7_path.read_text(encoding="utf-8"))
            extra = s7.get("extra") or {}
            if extra.get("scenario_actual") is not None or s7.get("scenario_actual") is not None:
                return {
                    "scenario_primary": extra.get("scenario_primary") or s7.get("scenario_primary"),
                    "scenario_actual": extra.get("scenario_actual") or s7.get("scenario_actual"),
                    "scenario_correct": extra.get("scenario_correct", s7.get("scenario_correct")),
                }
        except Exception:
            pass
    return None


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


def build_relative_strength(trading_date: str, driver: str) -> dict[str, Any] | None:
    """Relative strength vs QQQ from Step 0 raw quotes."""
    raw_path = raw_data_path(trading_date)
    if not raw_path.exists():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    qqq_q = _quote_from_raw(raw, "QQQ")
    qqq_chg = _change_pct(qqq_q)
    if qqq_chg is None:
        return None

    symbols = symbols_for_driver(driver)
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        q = _quote_from_raw(raw, sym)
        sym_chg = _change_pct(q)
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
