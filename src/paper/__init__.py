"""Simulated (paper) trading — ADVISORY ONLY · 不构成投资建议.

Rules-based v1: Morning / session #1 signal → ENTRY / HOLD / EXIT at last price.
No real broker.
"""

from __future__ import annotations

from src.paper.tick import run_paper_tick

__all__ = ["run_paper_tick"]
