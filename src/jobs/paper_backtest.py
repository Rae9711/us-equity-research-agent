"""CLI entry for the event-driven paper-engine backtest.

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Usage:
    python -m src.jobs.paper_backtest --days 180 --end-date 2026-07-02
"""

from __future__ import annotations

from src.paper.backtest import main

if __name__ == "__main__":
    main()
