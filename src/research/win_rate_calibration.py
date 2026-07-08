"""Historical win-rate calibration from Market Cases and training rows.

Matches current trade setup features (RS, gap, direction, VIX, macro) to
labeled historical cases. Falls back to rules-based win_prob when sample
size is insufficient.

ADVISORY ONLY — 不构成投资建议.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

MIN_CALIBRATION_SAMPLE = 20
_MATCH_TOLERANCE = 1  # allow one bucket mismatch for partial matches


def _bucket_rs(rs: float | None, *, is_short: bool) -> str:
    if rs is None:
        return "unknown"
    if is_short:
        if rs < -1.0:
            return "weak"
        if rs > 0.5:
            return "against"
        return "neutral"
    if rs > 1.0:
        return "strong"
    if rs < -0.5:
        return "against"
    return "neutral"


def _bucket_gap(gap: float | None) -> str:
    if gap is None:
        return "unknown"
    mag = abs(gap)
    if mag < 1.5:
        return "tight"
    if mag < 3.0:
        return "moderate"
    return "extended"


def _bucket_vix(vix_chg: float | None) -> str:
    if vix_chg is None:
        return "unknown"
    if vix_chg < -2.0:
        return "falling"
    if vix_chg > 3.0:
        return "rising"
    return "stable"


def _bucket_macro(macro_calendar: dict[str, Any] | None) -> str:
    if not macro_calendar:
        return "none"
    events = macro_calendar.get("events") or macro_calendar.get("today") or []
    if not events:
        return "none"
    high = [e for e in events if str(e.get("impact", "")).lower() in ("high", "medium")]
    return "active" if high else "light"


def build_feature_signature(
    *,
    direction: str,
    rs_vs_qqq: float | None = None,
    gap_pct: float | None = None,
    vix_chg: float | None = None,
    macro_calendar: dict[str, Any] | None = None,
    symbol: str | None = None,
) -> dict[str, str]:
    is_short = direction == "SHORT"
    return {
        "direction": direction if direction in ("LONG", "SHORT") else "unknown",
        "rs_bucket": _bucket_rs(rs_vs_qqq, is_short=is_short),
        "gap_bucket": _bucket_gap(gap_pct),
        "vix_bucket": _bucket_vix(vix_chg),
        "macro_bucket": _bucket_macro(macro_calendar),
        "symbol": (symbol or "").upper(),
    }


def _signature_match_score(a: dict[str, str], b: dict[str, str]) -> int:
    keys = ("direction", "rs_bucket", "gap_bucket", "vix_bucket", "macro_bucket")
    return sum(1 for k in keys if a.get(k) == b.get(k))


def _trade_outcome_from_case(case: dict[str, Any]) -> tuple[bool | None, float | None]:
    """Return (won, return_pct) for the morning primary trade."""
    labels = case.get("labels") or {}
    morning = case.get("morning") or {}
    features = case.get("features") or {}
    actual = case.get("actual") or {}

    direction = labels.get("trade_direction") or ""
    symbol = (labels.get("primary_symbol") or "").upper()
    if not symbol:
        primary = (morning.get("best_trades") or {}).get("primary") or {}
        symbol = (primary.get("symbol") or "").upper()
        direction = direction or primary.get("direction") or ""

    if direction not in ("LONG", "SHORT") or not symbol:
        return None, None

    sym_key = f"{symbol.lower()}_chg"
    ret = actual.get("return_pct")
    if ret is None:
        ret = features.get(sym_key)
    if ret is None:
        ret = features.get(f"{symbol}_chg")
    if ret is None:
        return None, None

    try:
        ret_f = float(ret)
    except (TypeError, ValueError):
        return None, None

    if direction == "LONG":
        won = ret_f > 0
    else:
        won = ret_f < 0
    return won, round(ret_f, 2)


def _load_cases_from_db(exclude_date: str | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    try:
        from src.db import MarketCase, get_session

        session = get_session()
        try:
            rows = session.query(MarketCase).order_by(MarketCase.date.desc()).limit(500).all()
            for row in rows:
                if exclude_date and row.date == exclude_date:
                    continue
                try:
                    cases.append(json.loads(row.case_json))
                except json.JSONDecodeError:
                    continue
        finally:
            session.close()
    except Exception:
        logger.debug("MarketCase DB unavailable for calibration", exc_info=True)
    return cases


def _load_cases_from_training_rows(exclude_date: str | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    try:
        from src.db import TrainingRow, get_session

        session = get_session()
        try:
            rows = session.query(TrainingRow).order_by(TrainingRow.date.desc()).limit(500).all()
            for row in rows:
                if exclude_date and row.date == exclude_date:
                    continue
                try:
                    features = json.loads(row.features_json)
                    labels = json.loads(row.labels_json)
                except json.JSONDecodeError:
                    continue
                cases.append(
                    {
                        "date": row.date,
                        "features": features,
                        "labels": labels,
                        "morning": features.get("morning") or {},
                        "actual": labels.get("actual") or {},
                    }
                )
        finally:
            session.close()
    except Exception:
        logger.debug("TrainingRow DB unavailable for calibration", exc_info=True)
    return cases


def load_historical_cases(exclude_date: str | None = None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for case in _load_cases_from_db(exclude_date) + _load_cases_from_training_rows(exclude_date):
        d = case.get("date") or ""
        if d and d in seen:
            continue
        if d:
            seen.add(d)
        out.append(case)
    return out


def _case_signature(case: dict[str, Any]) -> dict[str, str] | None:
    labels = case.get("labels") or {}
    features = case.get("features") or {}
    morning = case.get("morning") or {}
    primary = (morning.get("best_trades") or {}).get("primary") or {}

    direction = labels.get("trade_direction") or primary.get("direction") or ""
    symbol = labels.get("primary_symbol") or primary.get("symbol")
    rs = primary.get("relative_strength")
    if rs is None:
        rs = features.get("rs_vs_qqq")
    gap = primary.get("gap_pct")
    if gap is None:
        gap = features.get("qqq_gap")
    vix_chg = features.get("vix_chg")

    if direction not in ("LONG", "SHORT"):
        return None
    return build_feature_signature(
        direction=direction,
        rs_vs_qqq=_safe_float(rs),
        gap_pct=_safe_float(gap),
        vix_chg=_safe_float(vix_chg),
        macro_calendar=case.get("macro_calendar"),
        symbol=symbol,
    )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 100.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    low = max(0.0, (centre - margin) / denom) * 100
    high = min(1.0, (centre + margin) / denom) * 100
    return round(low, 1), round(high, 1)


def calibrate_win_prob(
    features: dict[str, str],
    *,
    rules_win_prob: float,
    exclude_date: str | None = None,
    min_sample: int = MIN_CALIBRATION_SAMPLE,
) -> dict[str, Any]:
    """Match features to historical cases; prefer calibrated when n >= min_sample."""
    cases = load_historical_cases(exclude_date=exclude_date)
    exact_wins = 0
    exact_n = 0
    partial_wins = 0
    partial_n = 0

    for case in cases:
        sig = _case_signature(case)
        if not sig:
            continue
        won, _ = _trade_outcome_from_case(case)
        if won is None:
            continue
        score = _signature_match_score(features, sig)
        if score == 5:
            exact_n += 1
            exact_wins += int(won)
        elif score >= 5 - _MATCH_TOLERANCE:
            partial_n += 1
            partial_wins += int(won)

    use_exact = exact_n >= min_sample
    n = exact_n if use_exact else partial_n
    wins = exact_wins if use_exact else partial_wins

    if n >= min_sample:
        rate = wins / n * 100.0
        ci_low, ci_high = _wilson_ci(wins, n)
        return {
            "calibrated_win_prob": round(rate, 1),
            "sample_size": n,
            "confidence_interval": [ci_low, ci_high],
            "source": "historical",
            "match_quality": "exact" if use_exact else "partial",
            "win_prob_source": "historical",
            "disclaimer": None,
        }

    return {
        "calibrated_win_prob": rules_win_prob,
        "sample_size": n,
        "confidence_interval": None,
        "source": "rules_fallback",
        "match_quality": "insufficient",
        "win_prob_source": "rules",
        "disclaimer": (
            f"Historical sample n={n} < {min_sample}; using rules breakdown"
            if n > 0
            else "No historical matches; using rules breakdown"
        ),
    }


def find_similar_days(
    features: dict[str, str],
    *,
    exclude_date: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return closest historical days by feature signature match."""
    cases = load_historical_cases(exclude_date=exclude_date)
    scored: list[tuple[int, dict[str, Any]]] = []

    for case in cases:
        sig = _case_signature(case)
        if not sig:
            continue
        won, ret = _trade_outcome_from_case(case)
        if won is None:
            continue
        match = _signature_match_score(features, sig)
        if match < 3:
            continue
        scored.append(
            (
                match,
                {
                    "date": case.get("date"),
                    "outcome": "win" if won else "loss",
                    "return_pct": ret,
                    "match_score": match,
                    "direction": sig.get("direction"),
                },
            )
        )

    scored.sort(key=lambda x: (-x[0], x[1].get("date") or ""), reverse=False)
    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:limit]]


def return_distribution_bins(returns: list[float], bins: int = 5) -> list[dict[str, Any]]:
    """Simple histogram of historical returns for similar days."""
    if len(returns) < 3:
        return []
    lo, hi = min(returns), max(returns)
    if lo == hi:
        return [{"bin": f"{lo:.1f}%", "count": len(returns), "pct": 100.0}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for r in returns:
        idx = min(bins - 1, int((r - lo) / width)) if width > 0 else 0
        counts[idx] += 1
    total = len(returns)
    out: list[dict[str, Any]] = []
    for i, c in enumerate(counts):
        b_lo = lo + i * width
        b_hi = b_lo + width
        out.append(
            {
                "bin": f"{b_lo:+.1f}% to {b_hi:+.1f}%",
                "count": c,
                "pct": round(c / total * 100, 1),
            }
        )
    return out
