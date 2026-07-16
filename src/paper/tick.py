"""Paper trading tick orchestration.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.paper.account import (
    append_equity_point,
    ensure_positions,
    load_account,
    mark_to_market,
    open_positions,
    save_account,
)
from src.paper.allocation import portfolio_allocation_snapshot
from src.paper.execution_agent import decide_and_act
from src.paper.journal import record_tick_decision
from src.research.entry_status import infer_session_phase
from src.utils.paths import raw_data_path
from src.utils.trading_calendar import require_trading_day, skipped_non_trading_day, today_et

logger = logging.getLogger(__name__)


def _load_raw(trading_date: str) -> dict[str, Any]:
    path = raw_data_path(trading_date)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load raw for paper tick")
        return {}


def run_paper_tick(
    trading_date: date | str | None = None,
    *,
    session_phase: str | None = None,
    force_price: float | None = None,
    allow_non_trading_day: bool = False,
) -> dict[str, Any]:
    """Run one simulated trading tick and persist account state.

    When market is closed, still marks/exits using last raw snapshot price
    (demo-friendly) unless no raw data exists.
    """
    if isinstance(trading_date, str):
        d = date.fromisoformat(trading_date)
    elif trading_date is None:
        d = today_et()
    else:
        d = trading_date

    if not allow_non_trading_day:
        resolved = require_trading_day(d, job="paper_tick")
        if resolved is None:
            return skipped_non_trading_day(d)
        d = resolved

    date_str = d.isoformat()
    phase = session_phase or infer_session_phase(date_str)
    raw = _load_raw(date_str)

    account = load_account()
    ensure_positions(account)
    decision = decide_and_act(
        account,
        date_str,
        session_phase=phase,
        raw=raw or None,
        force_price=force_price,
    )
    record_tick_decision(account, decision)

    # Refresh marks from decision quotes
    px_map: dict[str, float] = {}
    for row in decision.get("books") or []:
        sig = row.get("signal") or {}
        sym = (sig.get("symbol") or (row.get("trade") or {}).get("symbol") or "").upper()
        if sym and row.get("quote") is not None:
            px_map[sym] = float(row["quote"])
    if decision.get("quote") is not None and decision.get("signal"):
        sym = str((decision.get("signal") or {}).get("symbol") or "").upper()
        if sym:
            px_map.setdefault(sym, float(decision["quote"]))
    if px_map:
        mark_to_market(account, price_by_symbol=px_map)
    elif not open_positions(account):
        mark_to_market(account, None)

    append_equity_point(account, label=f"tick:{decision.get('action')}")
    save_account(account)

    positions = account.get("positions") or {}
    alloc_snap = portfolio_allocation_snapshot(account)
    summary = {
        "ok": True,
        "trading_date": date_str,
        "session_phase": phase,
        "action": decision.get("action"),
        "reason": decision.get("reason"),
        "quote": decision.get("quote"),
        "quote_source": decision.get("quote_source"),
        "trade": decision.get("trade"),
        "signal": decision.get("signal"),
        "entry_status": decision.get("entry_status"),
        "allocation": decision.get("allocation"),
        "portfolio": alloc_snap,
        "books": decision.get("books"),
        "account": {
            "cash": account.get("cash"),
            "equity": account.get("equity"),
            "realized_pnl": account.get("realized_pnl"),
            "unrealized_pnl": account.get("unrealized_pnl"),
            "total_pnl": account.get("total_pnl"),
            "total_return_pct": account.get("total_return_pct"),
            "position": account.get("position"),
            "positions": {
                "intraday": positions.get("intraday"),
                "swing": positions.get("swing"),
            },
            "cash_pct": alloc_snap.get("cash_pct"),
            "position_pct": alloc_snap.get("position_pct"),
        },
        "advisory": True,
        "advisory_zh": "模拟交易 · 不构成投资建议",
    }
    logger.info(
        "Paper tick %s: %s — %s (equity=%s)",
        date_str,
        summary["action"],
        summary["reason"],
        account.get("equity"),
    )
    return summary


def maybe_paper_tick_after_step(trading_date: date | str | None = None) -> dict[str, Any] | None:
    """Non-fatal hook for Step 2/3/4 — swallow errors."""
    try:
        return run_paper_tick(trading_date)
    except Exception:
        logger.exception("paper_tick after step failed (non-fatal)")
        return None
