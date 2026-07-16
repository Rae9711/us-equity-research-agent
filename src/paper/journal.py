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


def record_tick_decision(account: dict[str, Any], decision: dict[str, Any]) -> None:
    from src.paper.account import append_decision

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
        },
    )


def record_evening_learning(trading_date: str) -> dict[str, Any]:
    """After Step 7/8: compare paper outcomes vs morning plan; stub param tweak."""
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
    exits = [t for t in day_trades if t.get("action") == "EXIT"]
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
        "lesson": (lesson or "")[:400] if lesson else None,
        "surprise": (surprise or "")[:400] if surprise else None,
        "step8_one_liner": (step8.get("conclusion") or {}).get("one_liner"),
        "advisory_zh": "模拟复盘 · 不构成投资建议",
    }

    # Stub calibration: nudge risk_pct slightly after losses / wins
    params = account.setdefault("params", {})
    risk = float(params.get("risk_pct") or 1.0)
    if win is True and risk < 1.5:
        params["risk_pct"] = round(min(1.5, risk + 0.05), 2)
        note["param_adjust"] = f"risk_pct {risk} → {params['risk_pct']} (win)"
    elif win is False and risk > 0.5:
        params["risk_pct"] = round(max(0.5, risk - 0.05), 2)
        note["param_adjust"] = f"risk_pct {risk} → {params['risk_pct']} (loss)"

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
