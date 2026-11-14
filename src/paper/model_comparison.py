"""Anthropic-vs-Gemini shadow model comparison (report-only).

Why this module exists
----------------------
Since 2026-11-13 the morning job also asks Gemini for a shadow opinion and
recomputes the trade candidates from Gemini's parts. Comparing the two
models' *candidates* looks like the obvious comparison, but it is vacuous:

* ``compute_trade_decision`` derives win_prob / trade_action / final_score
  purely from rule inputs (raw quotes, rule_bundle); the LLM parts never
  enter those numbers.
* The only parts the decision reads are P9 (instrument / horizon), P13
  (edges) and P16 (gate / trade plan) -- all in RULE_PART_IDS, i.e. rule
  filled, and ``_merge_parts`` lets rule keys override LLM keys. So the
  merged P9/P13/P16 are (up to float noise) identical for both models.

Verified 2026-11-13: Gemini's 11 trade candidates match the primary's on
win_prob and trade_action for 11/11 symbols (8/11 byte-identical rows).
That 100% is a property of the pipeline, not evidence the models agree.

The comparison that CAN discriminate the models is at the level each model
actually authors: the narrative parts no rule fills -- P3 (macro calendar)
and P12 (focus stocks). This module compares those, and records the
candidate comparison with an explicit ``candidates_rule_determined`` flag
so nobody mistakes 100% candidate agreement for model agreement.

Report-only: nothing here feeds entries, sizing, or any gate.
"""

from __future__ import annotations

from typing import Any

# Parts the LLM authors outright: not in morning.RULE_PART_IDS and not
# engine-built (R0 regime, P17 hypothesis, P18 trade candidates).
MODEL_AUTHORED_PARTS: tuple[str, ...] = ("P3", "P12")

CANDIDATE_RULE_NOTE = (
    "Candidate win_prob / trade_action / final_score are computed from rule "
    "inputs only (raw quotes + rule_bundle); the only parts the decision "
    "reads (P9/P13/P16) are rule-filled and rule keys override LLM keys in "
    "the merge. Candidate agreement is therefore NOT evidence the models "
    "agree -- compare the model-authored parts (P3/P12) instead."
)


def _f(x: Any) -> float | None:
    try:
        if x is None or isinstance(x, bool):
            return None
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # reject NaN


def _round4(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def compare_shadow_candidates(
    primary: list[dict[str, Any]] | None,
    gemini: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Per-symbol agreement between the primary and Gemini candidate lists.

    ``candidates_rule_determined`` is always True: see the module docstring.
    A row counts as identical only when both sides present the symbol and
    the full row dicts are equal.
    """
    prim = {c.get("symbol"): c for c in (primary or [])
            if isinstance(c, dict) and c.get("symbol")}
    gem = {c.get("symbol"): c for c in (gemini or [])
           if isinstance(c, dict) and c.get("symbol")}
    rows: list[dict[str, Any]] = []
    for sym in sorted(set(prim) | set(gem)):
        p, g = prim.get(sym), gem.get(sym)
        if p is None or g is None:
            rows.append({
                "symbol": sym, "present": {"primary": p is not None,
                                           "gemini": g is not None},
                "win_prob_primary": _f(p.get("win_prob")) if p else None,
                "win_prob_gemini": _f(g.get("win_prob")) if g else None,
                "win_prob_equal": None, "trade_action_equal": None,
                "identical": False,
            })
            continue
        pw, gw = _f(p.get("win_prob")), _f(g.get("win_prob"))
        rows.append({
            "symbol": sym,
            "present": {"primary": True, "gemini": True},
            "win_prob_primary": pw,
            "win_prob_gemini": gw,
            "win_prob_equal": (pw == gw) if pw is not None and gw is not None else None,
            "trade_action_equal": p.get("trade_action") == g.get("trade_action"),
            "identical": p == g,
        })
    common = [r for r in rows if r["present"]["primary"] and r["present"]["gemini"]]
    n_ident = sum(1 for r in common if r["identical"])
    return {
        "candidates_rule_determined": True,
        "note": CANDIDATE_RULE_NOTE,
        "n_primary": len(prim),
        "n_gemini": len(gem),
        "n_common": len(common),
        "n_win_prob_equal": sum(1 for r in common if r["win_prob_equal"]),
        "n_trade_action_equal": sum(1 for r in common if r["trade_action_equal"]),
        "n_identical": n_ident,
        "agreement_rate": (n_ident / len(common)) if common else None,
        "rows": rows,
    }


def compare_model_parts(
    primary_parts: dict[str, Any] | None,
    gemini_parts: dict[str, Any] | None,
    part_ids: tuple[str, ...] = MODEL_AUTHORED_PARTS,
) -> dict[str, Any]:
    """Compare the parts each model actually authors (P3/P12).

    ``judgment_equal`` is None when either side is missing the part;
    ``confidence_delta`` is None when either confidence is missing.
    """
    prim, gem = primary_parts or {}, gemini_parts or {}
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for pid in part_ids:
        p, g = prim.get(pid) or {}, gem.get(pid) or {}
        p, g = (p if isinstance(p, dict) else {}), (g if isinstance(g, dict) else {})
        pj, gj = p.get("judgment"), g.get("judgment")
        pc, gc = _f(p.get("confidence")), _f(g.get("confidence"))
        delta = abs(pc - gc) if pc is not None and gc is not None else None
        if delta is not None:
            deltas.append(delta)
        rows.append({
            "part": pid,
            "primary_present": bool(p),
            "gemini_present": bool(g),
            "judgment_equal": (
                (str(pj).strip() == str(gj).strip())
                if pj is not None and gj is not None else None),
            "confidence_primary": pc,
            "confidence_gemini": gc,
            "confidence_delta": _round4(delta),
        })
    compared = [r for r in rows if r["primary_present"] and r["gemini_present"]]
    return {
        "parts": list(part_ids),
        "n_compared": len(compared),
        "n_judgment_equal": sum(1 for r in compared if r["judgment_equal"]),
        "mean_abs_confidence_delta": (
            _round4(sum(deltas) / len(deltas)) if deltas else None),
        "rows": rows,
    }


def summarize_model_agreement(morning_doc: dict[str, Any] | None) -> dict[str, Any]:
    """One-day Anthropic-vs-Gemini agreement entry from a morning.json doc."""
    doc = morning_doc or {}
    shadow = doc.get("gemini_shadow") or {}
    shadow_ok = bool(shadow.get("ok")) and bool(shadow.get("parts"))
    entry: dict[str, Any] = {
        "date": doc.get("trading_date"),
        "primary_model": None,  # morning.json records no primary model name
        "gemini_model": shadow.get("model"),
        "shadow_ok": shadow_ok,
        "shadow_error": shadow.get("error"),
    }
    if not shadow_ok:
        entry["candidates"] = None
        entry["parts"] = None
        entry["note"] = (
            "No usable Gemini shadow this day (shadow missing, failed, or "
            "empty); nothing to compare.")
        return entry
    entry["candidates"] = compare_shadow_candidates(
        doc.get("trade_candidates"), shadow.get("trade_candidates"))
    entry["parts"] = compare_model_parts(
        doc.get("parts"), shadow.get("parts"))
    return entry


def aggregate_model_agreement(
    entries_by_date: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Roll per-day agreement entries into the shadow-summary section."""
    entries = [e for _, e in sorted((entries_by_date or {}).items())
               if isinstance(e, dict)]
    with_shadow = [e for e in entries if e.get("shadow_ok")]
    cand_rows = cand_ident = 0
    parts_compared = parts_equal = 0
    deltas: list[float] = []
    for e in with_shadow:
        cand = e.get("candidates") or {}
        cand_rows += int(cand.get("n_common") or 0)
        cand_ident += int(cand.get("n_identical") or 0)
        parts = e.get("parts") or {}
        parts_compared += int(parts.get("n_compared") or 0)
        parts_equal += int(parts.get("n_judgment_equal") or 0)
        for r in (parts.get("rows") or []):
            if r.get("confidence_delta") is not None:
                deltas.append(float(r["confidence_delta"]))
    return {
        "days": entries,
        "totals": {
            "n_days": len(entries),
            "n_days_with_shadow": len(with_shadow),
            "candidate_rows": cand_rows,
            "candidate_identical": cand_ident,
            "candidate_agreement_rate": (
                cand_ident / cand_rows if cand_rows else None),
            "parts_compared": parts_compared,
            "parts_judgment_equal": parts_equal,
            "mean_abs_confidence_delta": (
                _round4(sum(deltas) / len(deltas)) if deltas else None),
        },
    }
