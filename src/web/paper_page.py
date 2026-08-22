"""Paper trading page context builders.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

from typing import Any

from src.paper.account import ensure_positions, load_account, mark_to_market, open_positions
from src.paper.allocation import portfolio_allocation_snapshot
from src.paper.signals import load_candidate_signals
from src.paper.trade_report import build_trade_cards
from src.research.entry_status import infer_session_phase
from src.utils.trading_calendar import today_et


def build_paper_page(trading_date: str | None = None) -> dict[str, Any]:
    date_str = trading_date or today_et().isoformat()
    account = load_account()
    ensure_positions(account)
    # Refresh marks + sync closed-only cumulative metrics for the UI.
    px_map: dict[str, float] = {}
    for _book, pos in open_positions(account):
        sym = str(pos.get("symbol") or "").upper()
        try:
            px = float(pos["last_price"]) if pos.get("last_price") is not None else None
        except (TypeError, ValueError):
            px = None
        if sym and px is not None and px > 0:
            px_map[sym] = px
    mark_to_market(account, price_by_symbol=px_map or None)
    phase = infer_session_phase(date_str)
    signals = load_candidate_signals(date_str)
    portfolio = portfolio_allocation_snapshot(account)

    decisions = [
        d
        for d in reversed(account.get("decisions") or [])
        if not d.get("voided")
    ][:40]
    trades = [
        t for t in reversed(account.get("trades") or []) if not t.get("voided")
    ][:40]
    journal = list(reversed(account.get("journal") or []))[:30]
    curve = account.get("equity_curve") or []

    today_decisions = [d for d in decisions if d.get("trading_date") == date_str]
    allocation_notes = [
        j for j in journal if j.get("type") == "allocation" or j.get("reason_zh")
    ][:12]

    primary = signals.get("morning_primary") or {}
    session_p = signals.get("session_primary") or {}
    swing = signals.get("swing") or {}
    top = signals.get("top_trades") or []

    positions = account.get("positions") or {}
    open_legs = [
        {
            "book": book,
            "book_zh": "短线" if book == "intraday" else "长线",
            **pos,
            "qty_label": (
                f"{pos.get('contracts') or pos.get('shares')} 张"
                if (pos.get("asset_class") or "") == "option"
                else f"{pos.get('shares')} 股"
            ),
        }
        for book, pos in open_positions(account)
    ]

    trade_cards = build_trade_cards(account, trading_date=date_str, limit=40)

    return {
        "trading_date": date_str,
        "session_phase": phase,
        "account": account,
        "position": account.get("position"),
        "positions": {
            "intraday": positions.get("intraday"),
            "swing": positions.get("swing"),
        },
        "open_legs": open_legs,
        "portfolio": portfolio,
        "cash_pct": portfolio.get("cash_pct"),
        "position_pct": portfolio.get("position_pct"),
        "allocation_reason": portfolio.get("allocation_reason"),
        "allocation_notes": allocation_notes,
        "decisions": decisions,
        "today_decisions": today_decisions,
        "trades": trades,
        "trade_cards": trade_cards,
        "journal": journal,
        "equity_curve": curve[-60:],
        "signal_preview": {
            "morning_primary": {
                "symbol": primary.get("symbol"),
                "direction": primary.get("direction"),
                "entry_price": primary.get("entry_price") or primary.get("entry"),
                "stop": primary.get("stop_price") or primary.get("stop"),
                "target": primary.get("target_price") or primary.get("target"),
                "win_prob": primary.get("win_prob"),
            },
            "session_primary": {
                "symbol": session_p.get("symbol"),
                "direction": session_p.get("direction"),
            },
            "swing": {
                "symbol": swing.get("symbol"),
                "direction": swing.get("direction"),
                "win_prob": swing.get("win_prob") if isinstance(swing, dict) else None,
            },
            "top_trades": [
                {
                    "symbol": t.get("symbol"),
                    "direction": t.get("direction"),
                    "win_prob": t.get("win_prob"),
                    "rank": t.get("rank") or i + 1,
                }
                for i, t in enumerate(top[:5])
            ],
        },
        "advisory_zh": "模拟交易 · 不构成投资建议",
    }
