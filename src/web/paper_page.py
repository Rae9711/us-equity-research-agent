"""Paper trading page context builders.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

from typing import Any

from src.paper.account import ensure_positions, load_account
from src.paper.signals import load_candidate_signals
from src.research.entry_status import infer_session_phase
from src.utils.trading_calendar import today_et


def build_paper_page(trading_date: str | None = None) -> dict[str, Any]:
    date_str = trading_date or today_et().isoformat()
    account = load_account()
    ensure_positions(account)
    phase = infer_session_phase(date_str)
    signals = load_candidate_signals(date_str)

    decisions = list(reversed(account.get("decisions") or []))[:40]
    trades = list(reversed(account.get("trades") or []))[:40]
    journal = list(reversed(account.get("journal") or []))[:20]
    curve = account.get("equity_curve") or []

    today_decisions = [d for d in decisions if d.get("trading_date") == date_str]
    positions = account.get("positions") or {}

    primary = signals.get("morning_primary") or {}
    session_p = signals.get("session_primary") or {}
    swing = signals.get("swing") or {}

    return {
        "trading_date": date_str,
        "session_phase": phase,
        "account": account,
        "position": account.get("position"),
        "positions": {
            "intraday": positions.get("intraday"),
            "swing": positions.get("swing"),
        },
        "decisions": decisions,
        "today_decisions": today_decisions,
        "trades": trades,
        "journal": journal,
        "equity_curve": curve[-60:],
        "signal_preview": {
            "morning_primary": {
                "symbol": primary.get("symbol"),
                "direction": primary.get("direction"),
                "entry_price": primary.get("entry_price") or primary.get("entry"),
                "stop": primary.get("stop_price") or primary.get("stop"),
                "target": primary.get("target_price") or primary.get("target"),
            },
            "session_primary": {
                "symbol": session_p.get("symbol"),
                "direction": session_p.get("direction"),
            },
            "swing": {
                "symbol": swing.get("symbol"),
                "direction": swing.get("direction"),
                "entry_price": swing.get("entry_price") or swing.get("entry"),
                "stop": swing.get("stop_price") or swing.get("stop"),
                "target": swing.get("target_price") or swing.get("target"),
            },
        },
        "advisory_zh": "模拟交易 · 不构成投资建议",
    }
