"""Paper trading journal + evening learning hook.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.paper.account import append_journal, load_account, save_account
from src.utils.paths import morning_json_path, step_json_path

logger = logging.getLogger(__name__)


def _load_json(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append_allocation_note(
    account: dict[str, Any],
    *,
    trading_date: str | None,
    action: str | None,
    symbol: str | None,
    allocation: dict[str, Any],
    trade: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
) -> None:
    equity = float(account.get("equity") or 0) or 1.0
    cash = float(account.get("cash") or 0)
    cash_pct = round(cash / equity * 100.0, 1)
    pos_pct = round(100.0 - cash_pct, 1)
    if trade and trade.get("position_pct") is not None:
        pos_pct = trade["position_pct"]
    reserve = allocation.get("cash_reserve_pct")
    reason = allocation.get("reason_zh") or fallback_reason or ""
    append_journal(
        account,
        {
            "type": "allocation",
            "trading_date": trading_date,
            "action": action,
            "symbol": symbol,
            "book": allocation.get("book"),
            "cash_pct": cash_pct,
            "position_pct": pos_pct,
            "cash_reserve_pct": reserve,
            "risk_pct": allocation.get("risk_pct"),
            "budget_share": allocation.get("budget_share"),
            "confidence": allocation.get("confidence"),
            "reason_zh": reason,
            "note": (
                f"为何用 {pos_pct}% 仓位 / 留 {reserve}% 现金：{reason}"
                if reserve is not None
                else reason
            ),
            "advisory_zh": "模拟分配 · 不构成投资建议",
        },
    )


def record_tick_decision(account: dict[str, Any], decision: dict[str, Any]) -> None:
    from src.paper.account import append_decision

    books = decision.get("books") or []
    if books:
        for row in books:
            allocation = row.get("allocation")
            append_decision(
                account,
                {
                    "trading_date": decision.get("trading_date"),
                    "action": row.get("action"),
                    "reason": row.get("reason"),
                    "book": row.get("book"),
                    "book_zh": row.get("book_zh"),
                    "horizon": (row.get("signal") or {}).get("horizon"),
                    "symbol": (row.get("signal") or {}).get("symbol")
                    or (row.get("trade") or {}).get("symbol"),
                    "quote": row.get("quote"),
                    "quote_source": row.get("quote_source"),
                    "entry_status": (row.get("entry_status") or {}).get("status")
                    if isinstance(row.get("entry_status"), dict)
                    else row.get("entry_status"),
                    "pnl": (row.get("trade") or {}).get("pnl"),
                    "allocation": allocation,
                },
            )
            if allocation and row.get("action") in ("ENTRY", "SKIP", "WAIT"):
                _append_allocation_note(
                    account,
                    trading_date=decision.get("trading_date"),
                    action=row.get("action"),
                    symbol=(row.get("signal") or {}).get("symbol")
                    or (row.get("trade") or {}).get("symbol"),
                    allocation=allocation,
                    trade=row.get("trade"),
                    fallback_reason=row.get("reason"),
                )
        return

    allocation = decision.get("allocation")
    append_decision(
        account,
        {
            "trading_date": decision.get("trading_date"),
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "symbol": (decision.get("signal") or {}).get("symbol")
            or (decision.get("trade") or {}).get("symbol"),
            "quote": decision.get("quote"),
            "quote_source": decision.get("quote_source"),
            "entry_status": (decision.get("entry_status") or {}).get("status")
            if isinstance(decision.get("entry_status"), dict)
            else decision.get("entry_status"),
            "pnl": (decision.get("trade") or {}).get("pnl"),
            "allocation": allocation,
        },
    )
    if allocation and decision.get("action") in ("ENTRY", "SKIP", "WAIT"):
        _append_allocation_note(
            account,
            trading_date=decision.get("trading_date"),
            action=decision.get("action"),
            symbol=(decision.get("signal") or {}).get("symbol")
            or (decision.get("trade") or {}).get("symbol"),
            allocation=allocation,
            trade=decision.get("trade"),
            fallback_reason=decision.get("reason"),
        )


def _closed_trade_pnls(account: dict[str, Any], limit: int = 30) -> list[float]:
    """Realised PnL per closing action (full EXIT + partial SCALE_OUT), newest last."""
    pnls: list[float] = []
    for t in account.get("trades") or []:
        if t.get("voided"):
            continue
        if t.get("action") in ("EXIT", "SCALE_OUT") and t.get("pnl") is not None:
            try:
                pnls.append(float(t["pnl"]))
            except (TypeError, ValueError):
                continue
    return pnls[-limit:]


def rolling_expectancy(pnls: list[float]) -> dict[str, Any]:
    """Win rate, avg win/loss, profit factor and per-trade expectancy ($)."""
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    win_rate = (len(wins) / n) if n else 0.0
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    expectancy = (sum(pnls) / n) if n else 0.0
    return {
        "n": n,
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
    }


def adaptive_risk_pct(
    current: float,
    stats: dict[str, Any],
    *,
    min_trades: int = 5,
    floor: float = 0.35,
    cap: float = 3.0,
    base: float = 1.5,
) -> tuple[float, str | None]:
    """Map realised expectancy → next risk_pct within [floor, cap].

    Ambition math (advisory, **not a guarantee**):
      monthly ≈ n_trades × E[R] × risk_pct
    When rolling edge is positive, raise risk toward ``cap`` so high-EV setups
    can deploy into the ~80% sleeve. When negative, collapse to ``floor``
    (min-risk companion). Cold start nudges gently toward ``base``.
    """
    n = int(stats.get("n") or 0)
    if n < min_trades:
        nudged = round(current + (base - current) * 0.25, 2)
        if abs(nudged - current) < 0.01:
            return current, None
        return nudged, f"样本不足({n}<{min_trades})，风险回归基准 {nudged}%"

    pf = float(stats.get("profit_factor") or 0.0)
    exp = float(stats.get("expectancy") or 0.0)
    wr = float(stats.get("win_rate") or 0.0)
    avg_win = float(stats.get("avg_win") or 0.0)
    avg_loss = abs(float(stats.get("avg_loss") or 0.0))
    # Approximate realised E[R] from $ expectancy / avg |loss| when available.
    realised_er = (exp / avg_loss) if avg_loss > 0 else 0.0

    # Positive, robust edge → scale up toward monthly-ambition risk; broken → floor.
    if exp > 0 and pf >= 1.2 and wr >= 0.4:
        # More headroom when payoff asymmetry is healthy (avg_win ≥ avg_loss).
        asymmetry_bonus = 0.35 if avg_win >= avg_loss else 0.0
        er_bonus = min(0.6, max(0.0, realised_er) * 0.5)
        target = base + min(cap - base, (pf - 1.2) * 0.8 + asymmetry_bonus + er_bonus)
    elif exp <= 0 or pf < 0.9:
        target = floor  # bleed → minimum risk until edge recovers
    else:
        target = base
    target = max(floor, min(cap, target))
    # Smooth toward the target so one window doesn't whipsaw sizing.
    nxt = round(current + (target - current) * 0.5, 2)
    nxt = max(floor, min(cap, nxt))
    if abs(nxt - current) < 0.01:
        return current, None
    return nxt, (
        f"滚动{n}笔 PF={pf:.2f} 期望={exp:+.2f} 胜率={wr:.0%} → risk_pct {current}→{nxt}%"
        f"（月≈n×E[R]×risk%，目标约10%非保证）"
    )


def record_evening_learning(trading_date: str) -> dict[str, Any]:
    """After Step 7/8: compare paper outcomes vs plan; adapt risk from expectancy."""
    account = load_account()
    morning = _load_json(morning_json_path(trading_date))
    step7 = _load_json(step_json_path(7, trading_date))
    step8 = _load_json(step_json_path(8, trading_date))

    primary = (morning.get("best_trades") or {}).get("primary") or morning.get(
        "best_opportunity"
    ) or {}
    plan_symbol = (primary.get("symbol") or "").upper() or None
    plan_er = primary.get("expected_return_pct")

    day_trades = [
        t
        for t in (account.get("trades") or [])
        if t.get("trading_date") == trading_date
    ]
    exits = [t for t in day_trades if t.get("action") in ("EXIT", "SCALE_OUT")]
    entries = [t for t in day_trades if t.get("action") == "ENTRY"]

    day_pnl = sum(float(t.get("pnl") or 0) for t in exits)
    win = None
    if exits:
        win = day_pnl > 0

    vs_plan = "n/a"
    if plan_symbol and entries:
        traded = (entries[0].get("symbol") or "").upper()
        vs_plan = "matched" if traded == plan_symbol else f"diverged ({traded} vs {plan_symbol})"
    elif plan_symbol and not entries:
        vs_plan = "no_fill"

    lesson = step7.get("lesson") or (step7.get("conclusion") or {}).get("one_liner")
    surprise = None
    try:
        from src.db.market_case_service import load_case

        case = load_case(trading_date)
        lesson = lesson or case.lesson
        surprise = case.surprise
    except Exception:
        pass

    # Rolling realised expectancy across recent closed trades (evidence).
    stats = rolling_expectancy(_closed_trade_pnls(account))

    note = {
        "type": "evening_review",
        "trading_date": trading_date,
        "plan_symbol": plan_symbol,
        "plan_expected_return_pct": plan_er,
        "vs_plan": vs_plan,
        "day_pnl": round(day_pnl, 2),
        "win": win,
        "entries": len(entries),
        "exits": len(exits),
        "rolling_stats": stats,
        "lesson": (lesson or "")[:400] if lesson else None,
        "surprise": (surprise or "")[:400] if surprise else None,
        "step8_one_liner": (step8.get("conclusion") or {}).get("one_liner"),
        "advisory_zh": "模拟复盘 · 不构成投资建议",
    }

    # Evidence-based sizing: scale risk_pct from rolling expectancy, not per-day noise.
    # Cap rises toward ~10%/mo ambition only when edge is proven (see adaptive_risk_pct).
    params = account.setdefault("params", {})
    if bool(params.get("adaptive_risk", True)):
        risk = float(params.get("risk_pct") or 1.5)
        floor = float(params.get("adaptive_risk_floor") or 0.35)
        cap = float(params.get("adaptive_risk_cap") or 3.0)
        from src.paper.account import DEFAULT_PARAMS

        adapt_base = float(DEFAULT_PARAMS.get("risk_pct") or 1.5)
        new_risk, adj_note = adaptive_risk_pct(
            risk, stats, floor=floor, cap=cap, base=adapt_base
        )
        if adj_note:
            params["risk_pct"] = new_risk
            note["param_adjust"] = adj_note
        account["learning_stats"] = {
            **stats,
            "risk_pct": params.get("risk_pct"),
            "updated_for": trading_date,
        }

    from src.paper.acceptance import (
        record_event_ls_paper_session,
        refresh_event_ls_qualification,
    )

    qualification = refresh_event_ls_qualification(account)
    if qualification.get("paper_eligible"):
        record_event_ls_paper_session(account, trading_date)
        qualification = refresh_event_ls_qualification(account)
    note["event_ls_qualification"] = {
        "status": qualification["status"],
        "backtest_passed": qualification["backtest_passed"],
        "paper_passed": qualification["paper_passed"],
        "enabled": False,
    }
    append_journal(account, note)
    save_account(account)
    logger.info(
        "Paper evening learning %s: pnl=%s win=%s vs_plan=%s",
        trading_date,
        day_pnl,
        win,
        vs_plan,
    )
    return note
