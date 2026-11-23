"""Calibrated-expectancy entry filter (2026-11-17).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

The shadow ledger already publishes a recalibration map
(``shadow_summary.json``: ``"recalibration"`` pooled by win_prob bucket
and ``"recalibration_by_horizon"`` split by horizon — shrinkage k=10
toward the bucket's stated mean + Wilson 95% CI; see
``src/paper/shadow_ledger.py``). Until now it only lived in reports.

This module wires it into the entry path as a FILTER ONLY:

* calibrated expectancy for a candidate =
  ``p_cal * R:R - (1 - p_cal) * 1`` (R units), where ``p_cal`` is the
  bucket's calibrated win rate and ``R:R`` the candidate's planned
  reward/risk.
* ``filter == "skip"`` iff calibrated expectancy is clearly negative.
  The shrinkage prior makes this conservative: with 34 settled trades
  the >70% bucket calibrates to ~0.53 (expectancy +0.24R at R:R 1.33,
  passes) while the <50% bucket calibrates to ~0.32 (expectancy
  -0.26R, skipped) — the filter bites on the worst slice, not on
  everything.
* When a bucket has no settled trades (``n == 0``) there is no
  evidence: the answer is ``"no-data"`` and the candidate passes
  (fail-open). A missing summary file is the same.
* The filter NEVER touches sizing: it only skips an entry, before
  allocation runs, and never modifies the signal, the win_prob, or any
  allocation input.
* Planned R:R is computed on the risk the system could actually take:
  a stop closer than the live minimum stop distance (0.8% of entry,
  ``min_stop_dist_pct``) can never be entered, so the raw R:R explodes
  as the stop distance -> 0 (real case 2026-11-19 NVDA intraday:
  0.07% stop -> a recorded +16.10R verdict). The risk is floored at
  0.8% of entry, so the expectancy reflects tradeable risk.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.paper.shadow_ledger import normalize_horizon, win_prob_bucket
from src.utils.paths import data_root

logger = logging.getLogger(__name__)

# Mirrors the live ``min_stop_dist_pct`` (account params, 0.8%) and the
# shadow ledger's ``MIN_STOP_DIST_PCT``: a stop closer than this to the
# entry is rejected at entry time, so it is not tradeable risk.
MIN_STOP_DIST_PCT = 0.8


def load_calibration_map() -> dict[str, Any]:
    """Read the recalibration maps from the shadow summary (disk).

    Returns ``{"pooled": {...}, "by_horizon": {...}}`` or ``{}`` when
    the summary is missing/corrupt. Fail-open by design.
    """
    path = data_root() / "paper" / "shadow_summary.json"
    if not path.exists():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("shadow summary unreadable: %s", path, exc_info=True)
        return {}
    if not isinstance(summary, dict):
        return {}
    pooled = summary.get("recalibration")
    by_horizon = summary.get("recalibration_by_horizon")
    out: dict[str, Any] = {}
    if isinstance(pooled, dict):
        out["pooled"] = pooled
    if isinstance(by_horizon, dict):
        out["by_horizon"] = by_horizon
    return out


def planned_risk_reward(
    entry: float | None,
    stop: float | None,
    target: float | None,
    direction: str | None,
    *,
    min_stop_dist_pct: float = MIN_STOP_DIST_PCT,
) -> float | None:
    """Planned reward / planned risk from levels, or None when degenerate.

    The risk is floored at ``min_stop_dist_pct`` of the entry price:
    a stop tighter than the live minimum can never be entered, so an
    R:R computed on its raw distance overstates the reward per unit of
    risk the system would actually take (it diverges as the stop
    distance -> 0). A zero/negative risk or reward is still degenerate
    and returns None; only a positive-but-sub-floor risk is floored.
    """
    try:
        e, s, t = float(entry), float(stop), float(target)
    except (TypeError, ValueError):
        return None
    risk = (e - s) if (direction or "LONG").upper() == "LONG" else (s - e)
    reward = (t - e) if (direction or "LONG").upper() == "LONG" else (e - t)
    if risk is None or risk <= 0 or reward is None or reward <= 0:
        return None
    floor = abs(e) * float(min_stop_dist_pct) / 100.0
    if risk < floor:
        risk = floor
    return reward / risk


def _lookup_calibrated(
    bucket: str, mapping: dict[str, Any], horizon: str | None
) -> tuple[float | None, int, str]:
    """Return (calibrated_win_prob, n, source) for a bucket.

    Prefers the by-horizon map when a horizon is given and that bucket
    has settled trades; falls back to the pooled map; ``(None, 0,
    "no-data")`` when neither has evidence.
    """
    if horizon:
        h = normalize_horizon(horizon)
        by_h = (mapping.get("by_horizon") or {}).get(h) or {}
        entry = by_h.get(bucket) or {}
        value = entry.get("calibrated_win_prob")
        n = int(entry.get("n") or 0)
        if value is not None and n > 0:
            return float(value), n, "by-horizon"
    pooled = mapping.get("pooled") or {}
    entry = pooled.get(bucket) or {}
    value = entry.get("calibrated_win_prob")
    n = int(entry.get("n") or 0)
    if value is not None and n > 0:
        return float(value), n, "pooled"
    return None, 0, "no-data"


def calibrated_expectancy(
    win_prob: float | None,
    risk_reward: float | None,
    *,
    horizon: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrated expectancy (R units) for one candidate.

    ``win_prob`` is in percent (0-100); ``risk_reward`` is the planned
    R:R. Returns ``calibrated_win_prob`` (fraction | None),
    ``calibrated_expectancy_r`` (float | None), ``bucket``,
    ``horizon_used``, ``source`` and ``n``.
    """
    if win_prob is None or risk_reward is None:
        return {
            "calibrated_win_prob": None,
            "calibrated_expectancy_r": None,
            "bucket": win_prob_bucket(win_prob),
            "horizon_used": normalize_horizon(horizon) if horizon else None,
            "source": "no-data",
            "n": 0,
        }
    try:
        rr = float(risk_reward)
        wp = float(win_prob)
    except (TypeError, ValueError):
        return {
            "calibrated_win_prob": None,
            "calibrated_expectancy_r": None,
            "bucket": win_prob_bucket(win_prob),
            "horizon_used": normalize_horizon(horizon) if horizon else None,
            "source": "no-data",
            "n": 0,
        }
    if rr <= 0:
        return {
            "calibrated_win_prob": None,
            "calibrated_expectancy_r": None,
            "bucket": win_prob_bucket(wp),
            "horizon_used": normalize_horizon(horizon) if horizon else None,
            "source": "no-data",
            "n": 0,
        }
    mapping = mapping if mapping is not None else load_calibration_map()
    bucket = win_prob_bucket(wp)
    p_cal, n, source = _lookup_calibrated(bucket, mapping, horizon)
    if p_cal is None:
        return {
            "calibrated_win_prob": None,
            "calibrated_expectancy_r": None,
            "bucket": bucket,
            "horizon_used": normalize_horizon(horizon) if horizon else None,
            "source": "no-data",
            "n": n,
        }
    expectancy = p_cal * rr - (1.0 - p_cal) * 1.0
    return {
        "calibrated_win_prob": round(p_cal, 3),
        "calibrated_expectancy_r": round(expectancy, 3),
        "bucket": bucket,
        "horizon_used": normalize_horizon(horizon) if horizon else None,
        "source": source,
        "n": n,
    }


def calibration_filter(
    win_prob: float | None,
    risk_reward: float | None,
    *,
    horizon: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter verdict for one candidate: ``pass`` | ``skip`` | ``no-data``.

    ``skip`` only when calibrated expectancy is clearly negative
    (``< 0``). Everything without evidence passes.
    """
    result = calibrated_expectancy(
        win_prob, risk_reward, horizon=horizon, mapping=mapping
    )
    expectancy = result.get("calibrated_expectancy_r")
    if expectancy is None:
        result["filter"] = "no-data"
        result["reason"] = None
        return result
    if expectancy < 0:
        result["filter"] = "skip"
        result["reason"] = (
            f"校准期望值 {expectancy:+.2f}R 为负"
            f"（胜率 {win_prob:.0f}%→校准 {result['calibrated_win_prob'] * 100:.0f}%"
            f"，R:R {float(risk_reward):.2f}，{result['source']} n={result['n']}），跳过"
        )
        return result
    result["filter"] = "pass"
    result["reason"] = None
    return result


def calibration_filter_for_signal(
    signal: dict[str, Any] | None,
    *,
    horizon: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter verdict for a normalized entry signal (``normalize_slot``).

    Uses the signal's ``win_prob`` and its PLANNED R:R — computed from
    the entry/stop/target levels first (the actual trade plan), falling
    back to the ``risk_reward`` field when the levels are degenerate.
    Never mutates the signal.
    """
    sig = signal or {}
    win_prob = sig.get("win_prob")
    try:
        win_prob_f = float(win_prob) if win_prob is not None else None
    except (TypeError, ValueError):
        win_prob_f = None
    rr_f = planned_risk_reward(
        sig.get("entry_price"),
        sig.get("stop_price"),
        sig.get("target_price"),
        sig.get("direction"),
    )
    if rr_f is None:
        rr = sig.get("risk_reward")
        try:
            rr_f = float(rr) if rr is not None else None
        except (TypeError, ValueError):
            rr_f = None
    return calibration_filter(win_prob_f, rr_f, horizon=horizon, mapping=mapping)
