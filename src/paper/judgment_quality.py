"""Judgment-quality panel for Daily Trading OS.

Scores EVERY shadow-ledger candidate (not just settled trades) on four
angles, so the reliability of the *judgement itself* can be verified on a
much bigger sample than the ~27 settled trades:

* Direction: did price move the predicted way N trading days after the
  candidate date? Reference is the last close BEFORE the candidate date
  (the judgement was made that morning); forward closes use bars strictly
  after it. No lookahead. Reported per horizon (1/3/5 days) and per
  win_prob bucket.
* Levels race: lenient entry -- touch the planned entry (or entry zone)
  any time inside the hold window -- then did the target hit before the
  stop? Uses the live module's ``evaluate_day`` for the race, so exit
  semantics match the shadow ledger. Outcomes: target_first / stop_first /
  never-entered / expired / censored (bars ran out).
* Consistency: direction flips for the same symbol across consecutive
  predictions (sorted by date, origin). A flip is counted whenever the
  direction changes vs the previous prediction for that symbol.
* Realism: planned stop/target distances in units of the symbol's recent
  average daily range (trailing 10 bars before the candidate date).
  A stop at 0.34 days of range is noise-scale: ordinary daily movement
  reaches it.

Pure functions only: no network, no file I/O. The job
(``src/jobs/shadow_ledger_update.py``) owns fetching bars and state.
"""

from __future__ import annotations

import math
from typing import Any

from src.paper.shadow_ledger import MAX_HOLD_DAYS, evaluate_day, win_prob_bucket

HORIZONS = (1, 3, 5)
RANGE_LOOKBACK = 10
MIN_RANGE_BARS = 5


def _f(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sorted_bars(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((b for b in bars if b.get("date")), key=lambda b: b["date"])


def _ref_close(bars: list[dict[str, Any]], cdate: str) -> float | None:
    """Last close strictly before the candidate date (known at judgement time)."""
    prev = [b for b in bars if (b.get("date") or "") < cdate]
    if not prev:
        return None
    return _f(prev[-1].get("close"))


def _fwd_close(bars: list[dict[str, Any]], cdate: str, n: int) -> float | None:
    """Close of the n-th trading day strictly after the candidate date."""
    after = [b for b in bars if (b.get("date") or "") > cdate]
    if len(after) < n:
        return None
    return _f(after[n - 1].get("close"))


def directional_scores(
    records: list[dict[str, Any]],
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """Forward-direction accuracy per horizon, overall and per win_prob bucket."""
    out: dict[str, Any] = {}
    for h in horizons:
        tot = hit = 0
        by_bucket: dict[str, list[int]] = {}
        for rec in records:
            bars = _sorted_bars(bars_by_symbol.get(rec["symbol"], []))
            ref = _ref_close(bars, rec["candidate_date"])
            fwd = _fwd_close(bars, rec["candidate_date"], h)
            if ref is None or fwd is None or ref <= 0:
                continue
            ret = fwd / ref - 1
            good = (ret > 0) if rec["direction"] == "LONG" else (ret < 0)
            tot += 1
            hit += 1 if good else 0
            bk = win_prob_bucket(rec.get("win_prob"))
            cell = by_bucket.setdefault(bk, [0, 0])
            cell[0] += 1 if good else 0
            cell[1] += 1
        out[f"{h}d"] = {
            "n": tot,
            "accuracy": (hit / tot) if tot else None,
            "by_bucket": {bk: {"n": n, "accuracy": (c / n) if n else None}
                          for bk, (c, n) in sorted(by_bucket.items())},
        }
    return out


def _touched_entry(record: dict[str, Any], bar: dict[str, Any]) -> bool:
    lo = _f(bar.get("low"))
    hi = _f(bar.get("high"))
    if lo is None or hi is None:
        return False
    zone = record.get("entry_zone") or {}
    zlow = _f(zone.get("low"))
    zhigh = _f(zone.get("high"))
    if zlow is not None and zhigh is not None:
        return zlow <= hi and zhigh >= lo
    entry = _f(record.get("entry"))
    return entry is not None and lo <= entry <= hi


def levels_race(
    records: list[dict[str, Any]],
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    as_of_date: str,
) -> dict[str, Any]:
    """Lenient-entry target-vs-stop race for every candidate.

    Entry: first bar after candidate_date (within the hold window) that
    touches the planned entry or zone. Then the race uses ``evaluate_day``
    from the entry bar on. Expiry exits at the close and counts as
    ``expired`` (no level hit within the window).
    """
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for rec in records:
        bars = [b for b in _sorted_bars(bars_by_symbol.get(rec["symbol"], []))
                if (b.get("date") or "") <= as_of_date]
        after = [b for b in bars if (b.get("date") or "") > rec["candidate_date"]]
        max_hold = MAX_HOLD_DAYS.get(rec.get("horizon") or "unknown", 3)
        outcome: str
        entry_date = None
        exit_date = None
        entry_idx = None
        for i, b in enumerate(after):
            if i >= max_hold:
                break
            if _touched_entry(rec, b):
                entry_idx = i
                entry_date = b.get("date")
                break
        if entry_idx is None:
            outcome = "censored" if len(after) < max_hold else "never-entered"
        else:
            probe = dict(rec)
            probe["entry_price"] = rec["entry"]
            outcome = "expired"
            for j in range(entry_idx, min(len(after), entry_idx + max_hold)):
                ex = evaluate_day(probe, after[j])
                if ex is not None:
                    outcome = ("target_first" if ex["exit_reason"] == "target"
                               else "stop_first")
                    exit_date = ex["exit_date"]
                    break
            else:
                if len(after) < entry_idx + max_hold:
                    outcome = "censored"
        counts[outcome] = counts.get(outcome, 0) + 1
        rows.append({"id": rec["id"], "symbol": rec["symbol"],
                     "direction": rec["direction"], "horizon": rec["horizon"],
                     "win_prob": rec.get("win_prob"), "outcome": outcome,
                     "entry_date": entry_date, "exit_date": exit_date})
    resolved = [r for r in rows if r["outcome"] in ("target_first", "stop_first")]
    tf = sum(1 for r in resolved if r["outcome"] == "target_first")
    by_bucket: dict[str, dict[str, Any]] = {}
    for bk in ("<50", "50-60", "60-70", ">70"):
        sub = [r for r in resolved if win_prob_bucket(r.get("win_prob")) == bk]
        if sub:
            n_tf = sum(1 for r in sub if r["outcome"] == "target_first")
            by_bucket[bk] = {"n": len(sub), "target_first_rate": n_tf / len(sub)}
    by_horizon: dict[str, dict[str, Any]] = {}
    for h in ("intraday", "swing", "unknown"):
        sub = [r for r in resolved if (r.get("horizon") or "unknown") == h]
        if sub:
            n_tf = sum(1 for r in sub if r["outcome"] == "target_first")
            by_horizon[h] = {"n": len(sub), "target_first_rate": n_tf / len(sub)}
    return {"counts": counts,
            "resolved_n": len(resolved),
            "target_first_rate": (tf / len(resolved)) if resolved else None,
            "by_bucket": by_bucket,
            "by_horizon": by_horizon,
            "rows": rows}


def consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Direction flips per symbol across consecutive predictions."""
    by_sym: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_sym.setdefault(rec["symbol"], []).append(rec)
    flips = 0
    flip_detail: list[str] = []
    flips_by_symbol: dict[str, int] = {}
    flip_days: dict[str, int] = {}
    for sym, recs in sorted(by_sym.items()):
        recs.sort(key=lambda r: (r["candidate_date"], r.get("origin") or ""))
        prev = None
        for rec in recs:
            if prev is not None and rec["direction"] != prev["direction"]:
                flips += 1
                flips_by_symbol[sym] = flips_by_symbol.get(sym, 0) + 1
                flip_days[rec["candidate_date"]] = flip_days.get(rec["candidate_date"], 0) + 1
                flip_detail.append(
                    f"{sym} {prev['candidate_date']} {prev['direction']} -> "
                    f"{rec['candidate_date']} {rec['direction']} ({rec.get('origin')})")
            prev = rec
    return {"predictions": len(records), "flips": flips,
            "flips_by_symbol": flips_by_symbol,
            "flip_days": dict(sorted(flip_days.items())),
            "flip_detail": flip_detail}


def _avg_range(bars: list[dict[str, Any]], cdate: str) -> float | None:
    prev = [b for b in bars if (b.get("date") or "") < cdate][-RANGE_LOOKBACK:]
    if len(prev) < MIN_RANGE_BARS:
        return None
    vals = []
    for b in prev:
        hi, lo, cl = _f(b.get("high")), _f(b.get("low")), _f(b.get("close"))
        if hi is None or lo is None or cl is None or cl <= 0:
            continue
        vals.append((hi - lo) / cl)
    if len(vals) < MIN_RANGE_BARS:
        return None
    return sum(vals) / len(vals)


def _quantiles(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "median": xs[n // 2], "p25": xs[n // 4], "p75": xs[3 * n // 4]}


def realism(
    records: list[dict[str, Any]],
    bars_by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Stop/target distances in units of the trailing average daily range."""
    stop_ratios: list[float] = []
    target_ratios: list[float] = []
    by_horizon: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        bars = _sorted_bars(bars_by_symbol.get(rec["symbol"], []))
        ar = _avg_range(bars, rec["candidate_date"])
        if ar is None or ar <= 0:
            continue
        entry = _f(rec.get("entry"))
        stop = _f(rec.get("stop"))
        target = _f(rec.get("target"))
        if entry is None or stop is None or target is None or entry <= 0:
            continue
        sr = abs(entry - stop) / entry / ar
        tr = abs(target - entry) / entry / ar
        stop_ratios.append(sr)
        target_ratios.append(tr)
        cell = by_horizon.setdefault(rec.get("horizon") or "unknown",
                                     {"stop": [], "target": []})
        cell["stop"].append(sr)
        cell["target"].append(tr)
    return {
        "stop_days_of_range": _quantiles(stop_ratios),
        "target_days_of_range": _quantiles(target_ratios),
        "stop_under_half_day_pct": (
            sum(1 for x in stop_ratios if x < 0.5) / len(stop_ratios)
            if stop_ratios else None),
        "by_horizon": {h: {"stop": _quantiles(v["stop"]),
                            "target": _quantiles(v["target"])}
                       for h, v in sorted(by_horizon.items())},
    }


def compute_panel(
    records: list[dict[str, Any]],
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    as_of_date: str,
) -> dict[str, Any]:
    """Full judgement-quality panel for one as-of date."""
    return {
        "as_of_date": as_of_date,
        "candidates": len(records),
        "directional": directional_scores(records, bars_by_symbol),
        "levels_race": levels_race(records, bars_by_symbol, as_of_date),
        "consistency": consistency(records),
        "realism": realism(records, bars_by_symbol),
    }
