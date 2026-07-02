from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import yfinance as yf

from src.research.format_body import normalize_body_md
from src.steps.base import save_step_result
from src.utils.paths import morning_json_path, raw_data_path
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)

MAG7 = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA"]


def _intraday_quote(ticker: str) -> dict[str, Any]:
    t = yf.Ticker(ticker)
    hist = t.history(period="1d", interval="1m", prepost=True)
    prev = t.fast_info.get("previous_close") or t.fast_info.get("regular_market_previous_close")
    if hist.empty or not prev:
        return {"ticker": ticker, "error": "no intraday data"}

    open_px = float(hist.iloc[0]["Open"])
    last_px = float(hist.iloc[-1]["Close"])
    prev = float(prev)
    gap_pct = round((open_px - prev) / prev * 100, 2)
    chg_pct = round((last_px - prev) / prev * 100, 2)
    return {
        "ticker": ticker,
        "prev_close": round(prev, 4),
        "open": round(open_px, 4),
        "last": round(last_px, 4),
        "gap_pct": gap_pct,
        "change_pct": chg_pct,
        "gap": "Up" if gap_pct > 0.05 else "Down" if gap_pct < -0.05 else "Flat",
    }


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

  # bond from observations passed in caller
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

    qqq = _intraday_quote("QQQ")
    smh = _intraday_quote("SMH")
    tnx = _intraday_quote("^TNX")

    mag7_moves: list[dict[str, Any]] = []
    for sym in MAG7:
        q = _intraday_quote(sym)
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
    body_lines.extend(f"- {n}" for n in notes if "交叉验证" in n or "Risk-on" in n)
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
