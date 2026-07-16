from __future__ import annotations

import json
import logging
from datetime import date, time
from typing import Any, Literal

from src.collectors.fred_client import FredClient
from src.research.format_body import normalize_body_md
from src.steps.base import save_step_result
from src.utils.data_freshness import dgs10_fred_stale, guard_fresh_raw
from src.utils.paths import morning_json_path, raw_data_path
from src.utils.pit_snapshots import (
    as_of_et_iso,
    format_as_of_display,
    pit_raw_for_step,
    save_snapshot,
    step_label,
    step_scheduled_time,
)
from src.utils.quote_resolve import intraday_session_quote, session_observation
from src.utils.trading_calendar import prior_trading_day, require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)

MAG7 = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]


def _safe_float(d: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    obj: Any = d
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)
    if obj is None:
        return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def _pct_chg(current: float | None, prev: float | None) -> float | None:
    if current is None or prev is None or prev == 0:
        return None
    return (current - prev) / abs(prev) * 100.0


def _treasury_rate(raw: dict[str, Any]) -> float | None:
    market = raw.get("market", {})
    macro = raw.get("macro", {})
    val = _safe_float(market.get("treasury_10y_fred") or {}, "value")
    if val is None:
        val = _safe_float((macro.get("series") or {}).get("DGS10") or {}, "value")
    if val is None:
        val = _safe_float(macro, "rates", "DGS10")
    return val


def _bond_from_fred() -> dict[str, Any]:
    fred = FredClient()
    data = fred.get(
        "/series/observations",
        {"series_id": "DGS10", "limit": 10, "sort_order": "desc"},
    )
    obs = [o for o in (data.get("observations") or []) if o.get("value") not in (None, ".")]
    if not obs:
        return {"ticker": "DGS10", "error": "no treasury data"}
    cur = obs[0]
    prev = obs[1] if len(obs) > 1 else None
    cur_val = float(cur["value"])
    prev_val = float(prev["value"]) if prev else None
    change_pct = _pct_chg(cur_val, prev_val) if prev_val is not None else None
    return {
        "ticker": "DGS10",
        "prev_close": round(prev_val, 4) if prev_val is not None else None,
        "last": round(cur_val, 4),
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "observation_date": cur.get("date"),
        "source": "fred",
    }


def _bond_observation(trading_date: date, raw: dict[str, Any], prior_raw: dict[str, Any]) -> dict[str, Any]:
    prior_day = prior_trading_day(trading_date)
    fred_stale = dgs10_fred_stale(raw, prior_day)

    tnx = intraday_session_quote("^TNX", trading_date)
    if "error" not in tnx:
        tnx["ticker"] = "^TNX"
        if fred_stale:
            tnx["fred_stale"] = True
        return tnx

    if not fred_stale:
        fred_bond = _bond_from_fred()
        if "error" not in fred_bond:
            return fred_bond

    cur = _treasury_rate(raw)
    prev = _treasury_rate(prior_raw)
    if cur is None:
        fred_bond = _bond_from_fred()
        if "error" not in fred_bond:
            return fred_bond
        return {"ticker": "DGS10", "error": "no treasury data"}
    change_pct = _pct_chg(cur, prev) if prev is not None else None
    return {
        "ticker": "DGS10",
        "prev_close": round(prev, 4) if prev is not None else None,
        "last": round(cur, 4),
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "source": "raw",
        "fred_stale": fred_stale,
    }


def _load_raw(trading_date: date) -> tuple[dict[str, Any], dict[str, Any]]:
    path = raw_data_path(trading_date.isoformat())
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    prior_path = raw_data_path(prior_trading_day(trading_date).isoformat())
    prior_raw: dict[str, Any] = {}
    if prior_path.exists():
        try:
            prior_raw = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return raw, prior_raw


def _morning_context(trading_date: date) -> dict[str, Any]:
    out: dict[str, Any] = {"p1_judgment": None, "bias": None, "total": None}
    path = morning_json_path(trading_date.isoformat())
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        out["bias"] = m.get("bias")
        out["total"] = m.get("total_score")
        p1 = (m.get("parts") or {}).get("P1") or {}
        out["p1_judgment"] = p1.get("judgment")
    return out


def _classify_open(
    qqq: dict[str, Any],
    smh: dict[str, Any],
    breadth: dict[str, Any],
    morning: dict[str, Any],
) -> tuple[str, float, list[str]]:
    notes: list[str] = []
    score = 0

    gap = qqq.get("gap", "Flat")
    if gap == "Up":
        score += 1
        notes.append(f"QQQ Gap {gap} ({qqq.get('gap_pct')}%)")
    elif gap == "Down":
        score -= 1
        notes.append(f"QQQ Gap {gap} ({qqq.get('gap_pct')}%)")
    else:
        notes.append("QQQ Gap Flat")

    adv = breadth.get("advancers", 0)
    dec = breadth.get("decliners", 0)
    if adv + dec > 0:
        ratio = adv / (adv + dec)
        if ratio >= 0.6:
            score += 1
            notes.append(f"Breadth 偏多 ({adv}/{adv+dec} Mag7 上涨)")
        elif ratio <= 0.4:
            score -= 1
            notes.append(f"Breadth 偏弱 ({dec}/{adv+dec} Mag7 下跌)")
        else:
            notes.append(f"Breadth 分化 ({adv}涨/{dec}跌)")

    leader = breadth.get("leader")
    if leader:
        notes.append(f"Leader: {leader}")

    qqq_chg = qqq.get("change_pct") or 0
    smh_chg = smh.get("change_pct") or 0
    if qqq_chg > 0 and smh_chg < -0.1:
        score -= 1
        notes.append("交叉验证：QQQ 涨 + SMH 跌 → Morning Risk-on 可能为假")
    elif qqq_chg > 0 and smh_chg > 0:
        score += 1
        notes.append("QQQ 与 SMH 同步走强，支持 Risk-on")

    if score >= 2:
        market = "Healthy"
        conf = 0.75
    elif score <= -1:
        market = "Weak"
        conf = 0.7
    else:
        market = "Mixed"
        conf = 0.65

    return market, conf, notes


def run_step2_open(
    trading_date: date | None = None,
    *,
    as_of_et: time | Literal["now"] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    d = require_trading_day(trading_date, job="run_step2_open")
    if d is None:
        return skipped_non_trading_day(trading_date)
    trading_date = d
    pit_as_of = as_of_et if as_of_et is not None else step_scheduled_time(2)
    logger.info("Step 2 Open report for %s (as-of %s)", trading_date, pit_as_of)

    _, stale = guard_fresh_raw(trading_date, step="run_step2_open")
    if stale:
        return stale

    raw, prior_raw = _load_raw(trading_date)
    if pit_as_of != "now":
        pit_raw = pit_raw_for_step(trading_date, 2, fallback_raw=raw)
        if pit_raw:
            raw = pit_raw

    qqq = session_observation(
        "QQQ", raw, prior_raw, trading_date, section="market", as_of_et=pit_as_of
    )
    smh = session_observation(
        "SMH", raw, prior_raw, trading_date, section="sector", as_of_et=pit_as_of
    )
    tnx = _bond_observation(trading_date, raw, prior_raw)

    mag7_moves: list[dict[str, Any]] = []
    for sym in MAG7:
        q = session_observation(
            sym, raw, prior_raw, trading_date, section="stocks", as_of_et=pit_as_of
        )
        if "error" not in q:
            mag7_moves.append(q)
    mag7_moves.sort(key=lambda x: x.get("change_pct") or 0, reverse=True)

    advancers = sum(1 for q in mag7_moves if (q.get("change_pct") or 0) > 0)
    decliners = sum(1 for q in mag7_moves if (q.get("change_pct") or 0) < 0)
    leader = mag7_moves[0]["ticker"] if mag7_moves else None

    breadth = {
        "advancers": advancers,
        "decliners": decliners,
        "leader": leader,
        "mag7": mag7_moves,
    }

    morning = _morning_context(trading_date)
    market, confidence, notes = _classify_open(qqq, smh, breadth, morning)

    bond_note = ""
    if "error" not in tnx:
        chg = tnx.get("change_pct")
        if tnx.get("ticker") == "DGS10":
            chg_txt = f"{chg:+.2f}%" if chg is not None else "—"
            stale_tag = "（FRED 滞后）" if tnx.get("fred_stale") else ""
            bond_note = f"10Y DGS10 {tnx.get('last')} ({chg_txt} vs 前日){stale_tag}"
        else:
            chg_txt = f"{chg:+.2f}%" if chg is not None else "—"
            stale_tag = "，FRED 滞后" if tnx.get("fred_stale") else ""
            bond_note = f"10Y proxy (^TNX) {chg_txt} vs 昨收{stale_tag}"
        if chg is not None and abs(chg) > 1.5:
            notes.append(f"Bond 突发：{bond_note}")

    body_lines = [
        "## Opening Report",
        "",
        f"**Market**：{market}",
        "",
        "### 观察项",
        "",
        f"- **① Gap**：QQQ {qqq.get('gap', 'N/A')}（开盘 {qqq.get('gap_pct')}% vs 昨收 {qqq.get('prev_close')}）",
        f"- **② Breadth**：Mag7 {advancers} 涨 / {decliners} 跌（代理广度）",
        f"- **③ Leader**：{leader or 'N/A'}（{mag7_moves[0].get('change_pct') if mag7_moves else '—'}%）",
        f"- **④ Bond**：{bond_note or '数据不可用'}",
        "",
        "### 交叉验证",
        "",
    ]
    body_lines.extend(f"- {n}" for n in notes if "交叉验证" in n or "Risk-on" in n or "同步走强" in n)
    if morning.get("p1_judgment"):
        body_lines.append(f"- Morning P1 判断：{morning['p1_judgment']}")

    one_liner_parts = [f"Gap {qqq.get('gap', 'Flat')}"]
    if qqq_chg := qqq.get("change_pct"):
        if smh_chg := smh.get("change_pct"):
            if qqq_chg > 0 and smh_chg > 0:
                one_liner_parts.append("SOXX/SMH 同步")
            elif qqq_chg > 0 and smh_chg < 0:
                one_liner_parts.append("半导体背离")
    one_liner = "，".join(one_liner_parts) + f" → 开盘 {market}"

    conclusion = {
        "part_id": "S2",
        "judgment": f"开盘：{market}",
        "confidence": confidence,
        "one_liner": one_liner[:256],
    }

    primary_entry_status = None
    try:
        from src.research.entry_status import enrich_primary_from_morning
        from src.utils.paths import morning_json_path as _morning_path

        morning_full: dict[str, Any] = {}
        mp = _morning_path(trading_date.isoformat())
        if mp.exists():
            morning_full = json.loads(mp.read_text(encoding="utf-8"))
        primary = (morning_full.get("best_trades") or {}).get("primary") or {}
        sym = str(primary.get("symbol") or "")
        live_px = None
        if sym:
            section = "stocks"
            if sym in ("QQQ", "SPY", "TQQQ"):
                section = "market"
            elif sym in ("SMH", "XLK", "XLF", "XLE"):
                section = "sector"
            obs = session_observation(
                sym, raw, prior_raw, trading_date, section=section, as_of_et=pit_as_of
            )
            if "error" not in obs:
                live_px = obs.get("last") or obs.get("close")
        primary_entry_status = enrich_primary_from_morning(
            morning_full,
            trading_date.isoformat(),
            session_phase="open",
            current_price=float(live_px) if live_px is not None else None,
        )
    except Exception:
        logger.exception("Step 2 entry_status enrichment failed (non-fatal)")

    payload = save_step_result(
        2,
        trading_date,
        step_id="S2",
        job_id="open_report",
        conclusion=conclusion,
        body_md=normalize_body_md("\n".join(body_lines)),
        extra={
            "decision_as_of": as_of_et_iso(trading_date, pit_as_of),
            "decision_as_of_et": format_as_of_display(pit_as_of, step_num=2),
            "observations": {
                "qqq": qqq,
                "smh": smh,
                "tnx": tnx,
                "breadth": breadth,
                "morning": morning,
            },
            "market": market,
            "primary_entry_status": primary_entry_status,
        },
    )
    if pit_as_of != "now":
        save_snapshot(
            trading_date,
            step_label(2),
            {"raw": raw, "step2": payload},
            force=force,
        )
    try:
        from src.paper.tick import maybe_paper_tick_after_step

        maybe_paper_tick_after_step(trading_date)
    except Exception:
        logger.exception("paper_tick after Step 2 failed (non-fatal)")
    return payload
