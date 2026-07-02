from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import yfinance as yf

from src.research.format_body import normalize_body_md
from src.steps.base import save_step_result
from src.utils.paths import morning_json_path, raw_data_path
from src.utils.trading_calendar import prior_trading_day, today_et

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


def _quotes(section: dict[str, Any]) -> dict[str, Any]:
    return section.get("quotes") or section.get("prices") or {}


def _quote(section: dict[str, Any], ticker: str) -> dict[str, Any]:
    return _quotes(section).get(ticker) or {}


def _gap_label(gap_pct: float | None) -> str:
    if gap_pct is None:
        return "N/A"
    if gap_pct > 0.05:
        return "Up"
    if gap_pct < -0.05:
        return "Down"
    return "Flat"


def _observation_from_quote(q: dict[str, Any]) -> dict[str, Any]:
    if "error" in q:
        return q
    prev = _safe_float(q, "prev_close")
    open_px = _safe_float(q, "open")
    last_px = _safe_float(q, "close") or _safe_float(q, "current_price") or _safe_float(q, "last")
    gap_pct = _safe_float(q, "gap_pct")
    if gap_pct is None:
        gap_pct = _pct_chg(open_px, prev)
    change_pct = _safe_float(q, "change_pct")
    if change_pct is None:
        change_pct = _pct_chg(last_px, prev)
    return {
        "ticker": q.get("ticker"),
        "prev_close": round(prev, 4) if prev is not None else None,
        "open": round(open_px, 4) if open_px is not None else None,
        "last": round(last_px, 4) if last_px is not None else None,
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "gap": _gap_label(gap_pct),
    }


def _intraday_quote(ticker: str) -> dict[str, Any]:
    t = yf.Ticker(ticker)
    hist = t.history(period="1d", interval="1m", prepost=True)
    prev = t.fast_info.get("previous_close") or t.fast_info.get("regular_market_previous_close")
    if hist.empty or not prev:
        return {"ticker": ticker, "error": "no intraday data"}

    open_px = float(hist.iloc[0]["Open"])
    last_px = float(hist.iloc[-1]["Close"])
    prev_f = float(prev)
    gap_pct = round((open_px - prev_f) / prev_f * 100, 2)
    chg_pct = round((last_px - prev_f) / prev_f * 100, 2)
    return {
        "ticker": ticker,
        "prev_close": round(prev_f, 4),
        "open": round(open_px, 4),
        "last": round(last_px, 4),
        "gap_pct": gap_pct,
        "change_pct": chg_pct,
        "gap": _gap_label(gap_pct),
        "source": "intraday",
    }


def _treasury_rate(raw: dict[str, Any]) -> float | None:
    market = raw.get("market", {})
    macro = raw.get("macro", {})
    val = _safe_float(market.get("treasury_10y_fred") or {}, "value")
    if val is None:
        val = _safe_float((macro.get("series") or {}).get("DGS10") or {}, "value")
    if val is None:
        val = _safe_float(macro, "rates", "DGS10")
    return val


def _bond_from_raw(raw: dict[str, Any], prior_raw: dict[str, Any]) -> dict[str, Any]:
    cur = _treasury_rate(raw)
    prev = _treasury_rate(prior_raw)
    if cur is None:
        return {"ticker": "DGS10", "error": "no treasury data"}
    change_pct = _pct_chg(cur, prev) if prev is not None else None
    return {
        "ticker": "DGS10",
        "prev_close": round(prev, 4) if prev is not None else None,
        "last": round(cur, 4),
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "source": "raw",
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


def _quote_with_fallback(
    ticker: str,
    raw: dict[str, Any],
    prior_raw: dict[str, Any],
    *,
    section: str,
    prior_section: str | None = None,
) -> dict[str, Any]:
    intraday = _intraday_quote(ticker)
    if "error" not in intraday:
        return intraday

    sec = raw.get(section, {})
    q = _quote(sec, ticker)
    if "error" in q or q.get("close") is None:
        return intraday

    # Enrich with prior-day close when collector omitted prev_close.
    if q.get("prev_close") is None and prior_raw:
        prior_sec = prior_raw.get(prior_section or section, {})
        prior_q = _quote(prior_sec, ticker)
        prior_close = _safe_float(prior_q, "close") or _safe_float(prior_q, "current_price")
        if prior_close is not None:
            q = {**q, "prev_close": prior_close}

    obs = _observation_from_quote(q)
    obs["source"] = "raw"
    return obs


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


def run_step2_open(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    logger.info("Step 2 Open report for %s", trading_date)

    raw, prior_raw = _load_raw(trading_date)

    qqq = _quote_with_fallback("QQQ", raw, prior_raw, section="market")
    smh = _quote_with_fallback("SMH", raw, prior_raw, section="sector")

    tnx = _intraday_quote("^TNX")
    if "error" in tnx:
        tnx = _bond_from_raw(raw, prior_raw)

    mag7_moves: list[dict[str, Any]] = []
    for sym in MAG7:
        q = _quote_with_fallback(sym, raw, prior_raw, section="stocks")
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
        if tnx.get("ticker") == "DGS10":
            bond_note = f"10Y DGS10 {tnx.get('last')} ({tnx.get('change_pct', 0):+.2f}% vs 前日)"
        else:
            bond_note = f"10Y proxy (^TNX) {tnx.get('change_pct', 0):+.2f}% vs 昨收"
        if abs(tnx.get("change_pct") or 0) > 1.5:
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

    payload = save_step_result(
        2,
        trading_date,
        step_id="S2",
        job_id="open_report",
        conclusion=conclusion,
        body_md=normalize_body_md("\n".join(body_lines)),
        extra={
            "observations": {
                "qqq": qqq,
                "smh": smh,
                "tnx": tnx,
                "breadth": breadth,
                "morning": morning,
            },
            "market": market,
        },
    )
    return payload
