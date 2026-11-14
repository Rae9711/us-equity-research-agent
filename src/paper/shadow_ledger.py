"""Shadow ledger for Daily Trading OS.

Tracks EVERY daily candidate (not just filled paper trades) under uniform,
deterministic rules so the quality of the *judgement itself* can be
calibrated: per win_prob bucket win rates, Brier score, expectancy in R,
profit factor, split by horizon and month.

Design notes
------------
* Pure functions only: no network, no file I/O. The job
  (``src/jobs/shadow_ledger_update.py``) owns fetching bars and state.
* No lookahead: entry is evaluated on the NEXT trading day's open; exits
  are evaluated on daily OHLC bars. Only fully closed bars (date < as-of
  date) are ever used.
* Every exit records ``r_multiple`` (reward / planned risk), including
  target exits -- the live paper ledger's failure to do so is a known
  defect this module deliberately does not repeat.
* Candidate sources (morning.json):
    - full plans: ``best_opportunity``, ``swing_trade``,
      ``best_trades.primary`` / ``best_trades.swing``, ``top_trades``.
      These carry entry_price / stop_price / target_price / horizon.
    - ``trade_candidates``: symbol / direction / win_prob / current_price /
      expected_high / expected_low / risk_reward only. Levels are DERIVED
      (target = expected extreme, stop = entry -/+ move / risk_reward) and
      marked ``levels_source="derived"``. A candidate that matches a full
      plan on (symbol, direction) merges into the plan row (one idea = one
      prediction) and keeps the plan's levels.
    - ``best_trades.stock_trades`` / ``watchlist_items`` carry no levels
      and no direction; they are not candidates and are ignored.
* step3.json: ``trade_reeval.top_trades`` (== ``session_trade_update`` /
  ``session_update`` lists) are fresh intraday-session plans. They are
  separate predictions from the morning ones (new as-of, new win_prob).
  ``primary_entry_status`` is a status check, not a new prediction.
* Horizon for plain candidates is "unknown". They come from the morning
  candidate pool that feeds the intraday primary pick, so their max hold
  is the intraday 3 trading days (see MAX_HOLD_DAYS).

Rules (deterministic, see task spec)
-----------------------------------
* Entry, next trading day open: LONG enters iff open <= entry_price or
  open inside entry_zone; SHORT enters iff open >= entry_price or inside
  the zone. Fill at the open. Otherwise skipped ("open-missed").
* Exit, daily bars from the entry day on: stop hit (low <= stop for LONG)
  exits at the stop; target hit (high >= target) exits at the target.
  A gap through a level fills at the open. If both levels are touched on
  the same day without a gap, order is decided by the open vs the entry
  price (open towards the target -> target first, else stop first) and
  the trade is flagged ``ambiguous_same_day_touch``.
* Max hold: intraday/unknown 3 trading days, swing 20 trading days
  (entry day counts as day 1); expiry exit at that day's close.
* ``r_multiple`` = signed exit move / planned risk (entry-to-stop
  distance). Win is r > 0.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = 1

MAX_HOLD_DAYS = {"intraday": 3, "swing": 20, "unknown": 3}

# Minimum stop distance (% of entry) for a candidate to be a trade at all.
# Mirrors the live ``min_stop_dist_pct`` (account params, 0.8%): the live
# system refuses to open a leg whose stop is closer than this to the
# entry, so the shadow ledger must not count such plans as trades either.
MIN_STOP_DIST_PCT = 0.8

WIN_PROB_BUCKETS = ["<50", "50-60", "60-70", ">70"]

BACKFILL_FROM = "2026-10-20"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(x: Any) -> float | None:
    """Best-effort float, None when missing/invalid/non-finite."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normalize_direction(raw: Any) -> str | None:
    if isinstance(raw, str):
        d = raw.strip().upper()
        if d in ("LONG", "SHORT"):
            return d
    return None


def normalize_horizon(raw: Any) -> str:
    if isinstance(raw, str):
        h = raw.strip().lower()
        if h in ("intraday", "short", "day", "daytrade", "day-trade"):
            return "intraday"
        if h in ("swing", "long", "position"):
            return "swing"
    return "unknown"


def win_prob_bucket(win_prob: float | None) -> str:
    """Bucket label for a win_prob given in percent (0-100)."""
    if win_prob is None:
        return "unknown"
    if win_prob < 50:
        return "<50"
    if win_prob < 60:
        return "50-60"
    if win_prob < 70:
        return "60-70"
    return ">70"


def r_multiple(direction: str, entry: float, stop: float, exit_price: float) -> float | None:
    """Reward / planned risk. Positive = win. None when risk is degenerate."""
    risk = (entry - stop) if direction == "LONG" else (stop - entry)
    if risk is None or risk <= 0:
        return None
    move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return move / risk


def _valid_levels(direction: str, entry: float, stop: float, target: float) -> bool:
    if direction == "LONG":
        return stop < entry < target
    return target < entry < stop


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def derive_levels(
    direction: str,
    current_price: float | None,
    expected_high: float | None,
    expected_low: float | None,
    risk_reward: float | None,
) -> tuple[float, float, float] | None:
    """Derive (entry, stop, target) for a plain trade_candidate.

    entry = current_price; target = the expected extreme in the trade
    direction; stop distance = expected move / risk_reward (falls back to
    the opposite expected extreme when risk_reward is unusable).
    """
    entry = _f(current_price)
    hi = _f(expected_high)
    lo = _f(expected_low)
    if entry is None or hi is None or lo is None:
        return None
    rr = _f(risk_reward)
    if direction == "LONG":
        target = hi
        move = target - entry
        if move <= 0:
            return None
        stop = entry - (move / rr) if rr and rr > 0 else lo
    else:
        target = lo
        move = entry - target
        if move <= 0:
            return None
        stop = entry + (move / rr) if rr and rr > 0 else hi
    if not _valid_levels(direction, entry, stop, target):
        return None
    return (entry, stop, target)


def _plan_to_candidate(plan: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Convert one full-plan dict to a candidate row, or None if unusable."""
    symbol = str(plan.get("symbol") or "").strip().upper()
    direction = normalize_direction(plan.get("direction"))
    entry = _f(plan.get("entry_price"))
    stop = _f(plan.get("stop_price"))
    target = _f(plan.get("target_price"))
    win_prob = _f(plan.get("win_prob"))
    if not symbol or direction is None or entry is None or stop is None or target is None:
        return None
    if not _valid_levels(direction, entry, stop, target):
        return None
    zone = plan.get("entry_zone") if isinstance(plan.get("entry_zone"), dict) else None
    zone_low = _f(zone.get("low")) if zone else None
    zone_high = _f(zone.get("high")) if zone else None
    if zone_low is None or zone_high is None:
        zone_low = zone_high = None
    return {
        "symbol": symbol,
        "direction": direction,
        "horizon": normalize_horizon(plan.get("horizon")),
        "horizon_source": "plan",
        "win_prob": win_prob,
        "entry": entry,
        "stop": stop,
        "target": target,
        "entry_zone": {"low": zone_low, "high": zone_high} if zone_low is not None else None,
        "levels_source": "plan",
        "sources": [source],
        "trade_action": plan.get("trade_action"),
    }


def _iter_plan_dicts(morning: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Yield (source_name, plan_dict) for every full-plan slot in morning.json."""
    out: list[tuple[str, dict[str, Any]]] = []
    for key in ("best_opportunity", "swing_trade"):
        v = morning.get(key)
        if isinstance(v, dict):
            out.append((key, v))
    bt = morning.get("best_trades")
    if isinstance(bt, dict):
        for key in ("primary", "swing"):
            v = bt.get(key)
            if isinstance(v, dict):
                out.append((f"best_trades.{key}", v))
        tt = bt.get("top_trades")
        if isinstance(tt, list):
            for i, v in enumerate(tt):
                if isinstance(v, dict):
                    out.append((f"best_trades.top_trades[{i}]", v))
    tt = morning.get("top_trades")
    if isinstance(tt, list):
        for i, v in enumerate(tt):
            if isinstance(v, dict):
                out.append((f"top_trades[{i}]", v))
    return out


def extract_morning_candidates(morning: dict[str, Any]) -> list[dict[str, Any]]:
    """All morning candidates for one date, merged to one row per idea.

    Full plans win over plain candidates on (symbol, direction); plans
    that are the same trade (same symbol/direction/levels) merge too.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_levels: dict[tuple[str, str, float, float, float], dict[str, Any]] = {}

    # 1) Full plans first (they carry real levels).
    for source, plan in _iter_plan_dicts(morning):
        cand = _plan_to_candidate(plan, source)
        if cand is None:
            continue
        lvl_key = (cand["symbol"], cand["direction"], cand["entry"], cand["stop"], cand["target"])
        if lvl_key in by_levels:
            prev = by_levels[lvl_key]
            for s in cand["sources"]:
                if s not in prev["sources"]:
                    prev["sources"].append(s)
            continue
        by_levels[lvl_key] = cand
        by_key[(cand["symbol"], cand["direction"])] = cand

    # 2) Plain trade_candidates: merge into a matching plan, else derive.
    cands = morning.get("trade_candidates")
    if isinstance(cands, list):
        for c in cands:
            if not isinstance(c, dict):
                continue
            symbol = str(c.get("symbol") or "").strip().upper()
            direction = normalize_direction(c.get("direction"))
            if not symbol or direction is None:
                continue
            key = (symbol, direction)
            win_prob = _f(c.get("win_prob"))
            if key in by_key:
                prev = by_key[key]
                if "trade_candidates" not in prev["sources"]:
                    prev["sources"].append("trade_candidates")
                if prev.get("win_prob") is None and win_prob is not None:
                    prev["win_prob"] = win_prob
                continue
            levels = derive_levels(
                direction,
                _f(c.get("current_price")),
                _f(c.get("expected_high")),
                _f(c.get("expected_low")),
                _f(c.get("risk_reward")),
            )
            if levels is None:
                continue
            entry, stop, target = levels
            cand = {
                "symbol": symbol,
                "direction": direction,
                "horizon": "unknown",
                "horizon_source": "derived",
                "win_prob": win_prob,
                "entry": entry,
                "stop": stop,
                "target": target,
                "entry_zone": None,
                "levels_source": "derived",
                "sources": ["trade_candidates"],
                "trade_action": c.get("trade_action"),
            }
            by_key[key] = cand
    return list(by_key.values())


def extract_step3_candidates(step3: dict[str, Any]) -> list[dict[str, Any]]:
    """Fresh intraday-session plans from step3.json (trade_reeval only).

    ``primary_entry_status`` is a status re-check of the morning plan, not
    a new prediction, and is deliberately excluded.
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float, float]] = set()
    reeval = step3.get("trade_reeval")
    plans: list[dict[str, Any]] = []
    if isinstance(reeval, dict):
        tt = reeval.get("top_trades")
        if isinstance(tt, list):
            plans = [p for p in tt if isinstance(p, dict)]
    for plan in plans:
        cand = _plan_to_candidate(plan, "step3.trade_reeval")
        if cand is None:
            continue
        lvl_key = (cand["symbol"], cand["direction"], cand["entry"], cand["stop"], cand["target"])
        if lvl_key in seen:
            continue
        seen.add(lvl_key)
        # step3 re-evaluates the intraday session trade; its plans carry no
        # horizon field, so mark the inference explicitly.
        cand["horizon"] = "intraday"
        cand["horizon_source"] = "inferred-step3-intraday"
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def make_record(candidate_date: str, origin: str, cand: dict[str, Any]) -> dict[str, Any]:
    """Build a ledger record for one candidate prediction."""
    return {
        "id": f"{candidate_date}|{origin}|{cand['symbol']}|{cand['direction']}",
        "candidate_date": candidate_date,
        "origin": origin,
        "symbol": cand["symbol"],
        "direction": cand["direction"],
        "horizon": cand["horizon"],
        "horizon_source": cand.get("horizon_source"),
        "win_prob": cand.get("win_prob"),
        "entry": cand["entry"],
        "stop": cand["stop"],
        "target": cand["target"],
        "entry_zone": cand.get("entry_zone"),
        "levels_source": cand.get("levels_source"),
        "sources": list(cand.get("sources") or []),
        "trade_action": cand.get("trade_action"),
        "status": "awaiting-entry",
        "entry_date": None,
        "entry_price": None,
        "skip_reason": None,
        "exit_date": None,
        "exit_price": None,
        "exit_reason": None,
        "ambiguous_same_day_touch": False,
        "gap_through_stop": False,
        "gap_through_target": False,
        "r_multiple": None,
        "hold_days": 0,
        "last_evaluated_date": None,
        "exit_bar": None,
    }


# ---------------------------------------------------------------------------
# Entry / exit evaluation (pure; bars = [{date, open, high, low, close}])
# ---------------------------------------------------------------------------

def _bar_date(bar: dict[str, Any]) -> str | None:
    d = bar.get("date")
    return str(d)[:10] if d else None


def _touched(direction: str, bar: dict[str, Any], stop: float, target: float) -> tuple[bool, bool]:
    low = _f(bar.get("low"))
    high = _f(bar.get("high"))
    if low is None or high is None:
        return (False, False)
    if direction == "LONG":
        return (low <= stop, high >= target)
    return (high >= stop, low <= target)


def _gap_exit(direction: str, bar: dict[str, Any], stop: float, target: float) -> str | None:
    """'stop' / 'target' if the open gapped through that level, else None."""
    op = _f(bar.get("open"))
    if op is None:
        return None
    if direction == "LONG":
        if op <= stop:
            return "stop"
        if op >= target:
            return "target"
    else:
        if op >= stop:
            return "stop"
        if op <= target:
            return "target"
    return None


def evaluate_entry(
    record: dict[str, Any], bars: list[dict[str, Any]]
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Evaluate entry on the first bar after candidate_date.

    Returns (outcome, bar, reason): ("entered", bar, None) /
    ("skipped", None, reason) / ("pending", None, None). Only bars
    strictly after candidate_date are considered; bars must be sorted by
    date ascending.

    A fill already beyond the stop (LONG open <= stop, SHORT open >= stop)
    or beyond the target is not a trade at all -- the stop would sit on
    the wrong side of the fill and the planned risk is meaningless -- so
    it is skipped, not entered-and-instantly-stopped.

    Live parity: the live system rejects entries whose stop sits closer
    than MIN_STOP_DIST_PCT to the entry (``min_stop_dist_pct`` in the
    account params, 0.8%). Such plans are not trades at all -- the stop
    is noise-scale -- so they are skipped here as "stop-too-tight"
    instead of letting normal noise stop them out.
    """
    entry = _f(record.get("entry"))
    stop = _f(record.get("stop"))
    if entry is not None and stop is not None and entry > 0:
        if abs(entry - stop) / entry * 100.0 < MIN_STOP_DIST_PCT:
            return ("skipped", None, "stop-too-tight")
    cdate = record["candidate_date"]
    for bar in bars:
        d = _bar_date(bar)
        if d is None or d <= cdate:
            continue
        op = _f(bar.get("open"))
        if op is None:
            continue
        direction = record["direction"]
        entry = record["entry"]
        stop = record["stop"]
        target = record["target"]
        zone = record.get("entry_zone") or {}
        zlow = _f(zone.get("low"))
        zhigh = _f(zone.get("high"))
        in_zone = zlow is not None and zhigh is not None and zlow <= op <= zhigh
        # Beyond-level checks first: they are more specific than open-missed.
        if direction == "LONG":
            if op <= stop:
                return ("skipped", None, "open-below-stop")
            if op >= target:
                return ("skipped", None, "open-above-target")
            ok = op <= entry or in_zone
        else:
            if op >= stop:
                return ("skipped", None, "open-above-stop")
            if op <= target:
                return ("skipped", None, "open-below-target")
            ok = op >= entry or in_zone
        if not ok:
            return ("skipped", None, "open-missed")
        return ("entered", bar, None)
    return ("pending", None, None)


def evaluate_day(
    record: dict[str, Any], bar: dict[str, Any]
) -> dict[str, Any] | None:
    """Evaluate one holding day. Returns an exit dict or None (still open).

    The exit dict carries exit_date/price/reason and the ambiguity/gap
    flags. Callers own hold-day counting and expiry.
    """
    direction = record["direction"]
    entry = record["entry_price"]
    stop = record["stop"]
    target = record["target"]
    if entry is None:
        return None
    d = _bar_date(bar)
    gap = _gap_exit(direction, bar, stop, target)
    if gap == "stop":
        return {"exit_date": d, "exit_price": _f(bar.get("open")),
                "exit_reason": "stop", "gap_through_stop": True,
                "gap_through_target": False, "ambiguous_same_day_touch": False}
    if gap == "target":
        return {"exit_date": d, "exit_price": _f(bar.get("open")),
                "exit_reason": "target", "gap_through_stop": False,
                "gap_through_target": True, "ambiguous_same_day_touch": False}
    stop_hit, target_hit = _touched(direction, bar, stop, target)
    if stop_hit and target_hit:
        op = _f(bar.get("open"))
        # Order by the open's direction vs the entry price; flag ambiguity.
        target_first = (op is not None and entry is not None
                        and ((direction == "LONG" and op >= entry)
                             or (direction == "SHORT" and op <= entry)))
        if target_first:
            return {"exit_date": d, "exit_price": target,
                    "exit_reason": "target", "gap_through_stop": False,
                    "gap_through_target": False, "ambiguous_same_day_touch": True}
        return {"exit_date": d, "exit_price": stop,
                "exit_reason": "stop", "gap_through_stop": False,
                "gap_through_target": False, "ambiguous_same_day_touch": True}
    if stop_hit:
        return {"exit_date": d, "exit_price": stop,
                "exit_reason": "stop", "gap_through_stop": False,
                "gap_through_target": False, "ambiguous_same_day_touch": False}
    if target_hit:
        return {"exit_date": d, "exit_price": target,
                "exit_reason": "target", "gap_through_stop": False,
                "gap_through_target": False, "ambiguous_same_day_touch": False}
    return None


def update_record(
    record: dict[str, Any],
    bars: list[dict[str, Any]],
    as_of_date: str,
) -> dict[str, Any]:
    """Advance one record through bars up to as_of_date (inclusive).

    Bars must be sorted by date ascending and contain only closed bars
    (date <= as_of_date). Mutates and returns the record. Idempotent:
    ``last_evaluated_date`` makes re-running over the same bars a no-op,
    so hold-day counting and exits reproduce exactly.
    """
    usable = sorted(
        (b for b in bars if (_bar_date(b) or "") <= as_of_date),
        key=lambda b: _bar_date(b) or "",
    )
    if record["status"] == "awaiting-entry":
        outcome, bar, reason = evaluate_entry(record, usable)
        if outcome == "entered" and bar is not None:
            record["status"] = "open"
            record["entry_date"] = _bar_date(bar)
            record["entry_price"] = _f(bar.get("open"))
            record["hold_days"] = 1
            record["last_evaluated_date"] = None
        elif outcome == "skipped":
            record["status"] = "skipped"
            record["skip_reason"] = reason or "open-missed"
            return record
        else:
            return record
    if record["status"] != "open":
        return record

    max_hold = MAX_HOLD_DAYS.get(record.get("horizon") or "unknown", 3)
    last_eval = record.get("last_evaluated_date")
    for bar in usable:
        d = _bar_date(bar)
        if d is None or d < (record["entry_date"] or ""):
            continue
        if last_eval is not None and d <= last_eval:
            continue
        # hold_days counts trading days from the entry day (day 1).
        if d != record["entry_date"]:
            record["hold_days"] = (record.get("hold_days") or 1) + 1
        record["last_evaluated_date"] = d
        exit_info = evaluate_day(record, bar)
        if exit_info is not None:
            _apply_exit(record, exit_info, bar)
            return record
        if (record.get("hold_days") or 0) >= max_hold:
            close = _f(bar.get("close"))
            _apply_exit(record, {
                "exit_date": d, "exit_price": close,
                "exit_reason": "expiry", "gap_through_stop": False,
                "gap_through_target": False, "ambiguous_same_day_touch": False,
            }, bar)
            return record
    return record


def _apply_exit(record: dict[str, Any], exit_info: dict[str, Any], bar: dict[str, Any]) -> None:
    record["status"] = "exited"
    record["exit_date"] = exit_info["exit_date"]
    record["exit_price"] = exit_info["exit_price"]
    record["exit_reason"] = exit_info["exit_reason"]
    record["ambiguous_same_day_touch"] = bool(exit_info.get("ambiguous_same_day_touch"))
    record["gap_through_stop"] = bool(exit_info.get("gap_through_stop"))
    record["gap_through_target"] = bool(exit_info.get("gap_through_target"))
    record["exit_bar"] = {
        "date": _bar_date(bar),
        "open": _f(bar.get("open")),
        "high": _f(bar.get("high")),
        "low": _f(bar.get("low")),
        "close": _f(bar.get("close")),
    }
    ep = record.get("entry_price")
    if ep is not None and exit_info["exit_price"] is not None:
        record["r_multiple"] = r_multiple(
            record["direction"], ep, record["stop"], exit_info["exit_price"])


# ---------------------------------------------------------------------------
# Calibration summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Recalibration: stated win_prob -> observed win rate
# ---------------------------------------------------------------------------
# The shadow ledger's buckets report the empirical win rate, but with only
# ~26 settled trades a bucket's raw rate is mostly noise (the >70% bucket
# sits at 3/7 = 43% against a stated ~75%). A recalibration map shrinks
# each bucket's empirical rate toward the bucket's mean *stated* win_prob
# with a fixed pseudo-observation strength, and reports a Wilson 95% CI,
# so the daily watch can say what a stated confidence is actually worth
# instead of quoting either the noisy raw rate or the untested claim.
# This is reporting only: nothing here feeds live sizing or entry rules
# (changing those needs Rae's decision).
# ---------------------------------------------------------------------------

# Pseudo-observation strength of the prior (mean stated win_prob).
RECALIBRATION_PRIOR_STRENGTH = 10.0

# Settled trades needed before a bucket's calibrated rate counts as
# evidence rather than a shrunk placeholder.
RECALIBRATION_MIN_SAMPLE = 10

# Fallback prior (fraction) when a bucket has settled trades but no stated
# win_prob at all: the bucket's nominal centre.
BUCKET_PRIOR_CENTRE = {"<50": 0.40, "50-60": 0.55, "60-70": 0.65, ">70": 0.80}


def _wilson_ci_frac(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a win rate, as fractions in [0, 1]."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def recalibration_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-bucket stated-confidence -> observed-win-rate mapping.

    For each win_prob bucket over settled records:
    - ``empirical_win_rate``: wins / n (None when n == 0).
    - ``stated_mean_win_prob``: mean of the stated win_probs (fraction,
      None when no stated probs).
    - ``calibrated_win_prob``: shrinkage estimate
      (n * empirical + k * prior) / (n + k) with k =
      RECALIBRATION_PRIOR_STRENGTH and prior = stated mean (or the bucket
      centre when no stated probs). None when n == 0 and no prior exists
      (cannot happen for a bucket with settled trades, but kept explicit).
    - ``wilson_ci_95``: [low, high] fractions; [0, 1] when n == 0.
    - ``sample_sufficient``: n >= RECALIBRATION_MIN_SAMPLE.
    """
    settled = [r for r in records if r.get("status") == "exited" and r.get("r_multiple") is not None]
    out: dict[str, dict[str, Any]] = {}
    for bucket in WIN_PROB_BUCKETS:
        sub = [r for r in settled if win_prob_bucket(r.get("win_prob")) == bucket]
        n = len(sub)
        wins = sum(1 for r in sub if r["r_multiple"] > 0)
        stated = [r["win_prob"] / 100.0 for r in sub if r.get("win_prob") is not None]
        stated_mean = sum(stated) / len(stated) if stated else None
        prior = stated_mean if stated_mean is not None else BUCKET_PRIOR_CENTRE[bucket]
        if n == 0:
            calibrated = None
            empirical = None
        else:
            empirical = wins / n
            calibrated = (n * empirical + RECALIBRATION_PRIOR_STRENGTH * prior) / (
                n + RECALIBRATION_PRIOR_STRENGTH
            )
        lo, hi = _wilson_ci_frac(wins, n)
        out[bucket] = {
            "n": n,
            "wins": wins,
            "empirical_win_rate": empirical,
            "stated_mean_win_prob": stated_mean,
            "calibrated_win_prob": calibrated,
            "wilson_ci_95": [round(lo, 3), round(hi, 3)],
            "sample_sufficient": n >= RECALIBRATION_MIN_SAMPLE,
        }
    return out


def calibrated_win_prob(
    win_prob: float | None, mapping: dict[str, dict[str, Any]] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map one stated win_prob (percent) to its calibrated rate.

    Returns {"calibrated_win_prob": fraction|None, "bucket": str,
    "source": "calibrated"|"prior"|"unstated"}. With no settled trades in
    the bucket the answer is the bucket prior (the stated mean, or the
    nominal centre) and the source says so -- it is a placeholder, not
    evidence. Pass a precomputed ``mapping`` (from recalibration_map) or
    the raw ``records``; either is enough.
    """
    if win_prob is None:
        return {"calibrated_win_prob": None, "bucket": "unknown", "source": "unstated"}
    bucket = win_prob_bucket(win_prob)
    if mapping is None:
        mapping = recalibration_map(records or [])
    entry = mapping.get(bucket) or {}
    value = entry.get("calibrated_win_prob")
    if value is None:
        prior = entry.get("stated_mean_win_prob")
        if prior is None:
            prior = BUCKET_PRIOR_CENTRE.get(bucket)
        return {"calibrated_win_prob": prior, "bucket": bucket, "source": "prior"}
    source = "calibrated" if entry.get("sample_sufficient") else "calibrated-shrunk"
    return {"calibrated_win_prob": value, "bucket": bucket, "source": source}


def _stats(rs: list[float], win_probs: list[float | None]) -> dict[str, Any]:
    n = len(rs)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": None,
                "expectancy_r": None, "profit_factor": None, "brier": None}
    wins = sum(1 for r in rs if r > 0)
    losses = n - wins
    pos = sum(r for r in rs if r > 0)
    neg = -sum(r for r in rs if r < 0)
    brier = None
    pairs = [(p, 1.0 if r > 0 else 0.0) for p, r in zip(win_probs, rs) if p is not None]
    if pairs:
        brier = sum((p / 100.0 - y) ** 2 for p, y in pairs) / len(pairs)
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n,
        "expectancy_r": sum(rs) / n,
        "profit_factor": (pos / neg) if neg > 0 else (None if pos == 0 else float("inf")),
        "brier": brier,
    }


def compute_summary(records: list[dict[str, Any]], as_of_date: str | None = None) -> dict[str, Any]:
    """Calibration summary over all records. Pure and reproducible."""
    settled = [r for r in records if r.get("status") == "exited" and r.get("r_multiple") is not None]
    rs = [r["r_multiple"] for r in settled]
    probs = [r.get("win_prob") for r in settled]
    totals = _stats(rs, probs)
    totals["candidates"] = len(records)
    totals["entered"] = sum(1 for r in records if r.get("status") in ("open", "exited"))
    totals["skipped"] = sum(1 for r in records if r.get("status") == "skipped")
    totals["awaiting_entry"] = sum(1 for r in records if r.get("status") == "awaiting-entry")
    totals["open"] = sum(1 for r in records if r.get("status") == "open")
    totals["settled"] = len(settled)

    buckets: dict[str, dict[str, Any]] = {}
    for b in WIN_PROB_BUCKETS:
        sub = [r for r in settled if win_prob_bucket(r.get("win_prob")) == b]
        buckets[b] = _stats([r["r_multiple"] for r in sub], [r.get("win_prob") for r in sub])

    by_horizon: dict[str, dict[str, Any]] = {}
    for h in ("intraday", "swing", "unknown"):
        sub = [r for r in settled if (r.get("horizon") or "unknown") == h]
        by_horizon[h] = _stats([r["r_multiple"] for r in sub], [r.get("win_prob") for r in sub])

    by_month: dict[str, dict[str, Any]] = {}
    months = sorted({(r.get("candidate_date") or "")[:7] for r in settled if r.get("candidate_date")})
    for m in months:
        sub = [r for r in settled if (r.get("candidate_date") or "").startswith(m)]
        by_month[m] = _stats([r["r_multiple"] for r in sub], [r.get("win_prob") for r in sub])

    return {
        "as_of_date": as_of_date,
        "totals": totals,
        "buckets": buckets,
        "recalibration": recalibration_map(settled),
        "by_horizon": by_horizon,
        "by_month": by_month,
        "settled_trades": [
            {
                "id": r["id"],
                "candidate_date": r.get("candidate_date"),
                "symbol": r.get("symbol"),
                "direction": r.get("direction"),
                "horizon": r.get("horizon"),
                "win_prob": r.get("win_prob"),
                "entry": r.get("entry_price"),
                "exit": r.get("exit_price"),
                "exit_reason": r.get("exit_reason"),
                "r_multiple": r.get("r_multiple"),
                "ambiguous_same_day_touch": r.get("ambiguous_same_day_touch"),
            }
            for r in settled
        ],
    }
