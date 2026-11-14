"""Tests for the Anthropic-vs-Gemini shadow comparison (src/paper/model_comparison.py).

All fixtures are synthetic -- no live quotes, no state files, except the
job-level test which uses a tmp state dir and no records (so no bar fetch).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.jobs.shadow_ledger_update import run as shadow_job_run
from src.paper.model_comparison import (
    aggregate_model_agreement,
    compare_model_parts,
    compare_shadow_candidates,
    summarize_model_agreement,
)


def _cand(symbol, win_prob, action="Pass"):
    return {"symbol": symbol, "win_prob": float(win_prob),
            "trade_action": action, "final_score": 0.36}


def _morning_doc(*, shadow_ok=True, trading_date="2026-11-13"):
    doc = {
        "trading_date": trading_date,
        "trade_candidates": [_cand("NVDA", 51), _cand("AMD", 59)],
        "parts": {
            "P3": {"judgment": "无重大 Tier 1 数据发布", "confidence": 0.8},
            "P12": {"judgment": "N/A", "confidence": None,
                    "one_liner": "LLM 未返回该 Part，已回填占位"},
        },
    }
    if shadow_ok:
        doc["gemini_shadow"] = {
            "ok": True,
            "model": "gemini-3.5-flash",
            "error": None,
            "parts": {
                "P3": {"judgment": "无重大 Tier 1 数据发布", "confidence": 0.8},
                "P12": {"judgment": "焦点股票：NVDA / MSFT / AMZN",
                        "confidence": 0.7},
            },
            "trade_candidates": [_cand("NVDA", 51), _cand("AMD", 59)],
        }
    return doc


def test_candidates_identical_flagged_rule_determined():
    cands = [_cand("NVDA", 51), _cand("AMD", 59), _cand("QQQ", 54)]
    out = compare_shadow_candidates(cands, [dict(c) for c in cands])
    assert out["candidates_rule_determined"] is True
    assert "rule" in out["note"].lower()
    assert out["n_common"] == 3
    assert out["n_identical"] == 3
    assert out["agreement_rate"] == 1.0
    # 100% candidate agreement must not be mistaken for model agreement.
    assert out["n_win_prob_equal"] == 3


def test_candidates_differ_detected():
    prim = [_cand("NVDA", 51), _cand("AMD", 59, "Small"), _cand("QQQ", 54)]
    gem = [_cand("NVDA", 62), _cand("AMD", 59, "Pass"), _cand("TSLA", 42)]
    out = compare_shadow_candidates(prim, gem)
    assert out["n_common"] == 2
    assert out["n_win_prob_equal"] == 1  # AMD only
    assert out["n_trade_action_equal"] == 1  # NVDA only
    assert out["n_identical"] == 0
    assert out["agreement_rate"] == 0.0
    by_sym = {r["symbol"]: r for r in out["rows"]}
    assert by_sym["QQQ"]["present"] == {"primary": True, "gemini": False}
    assert by_sym["QQQ"]["win_prob_equal"] is None
    assert by_sym["TSLA"]["present"] == {"primary": False, "gemini": True}


def test_candidates_empty_sides_do_not_crash():
    out = compare_shadow_candidates([], None)
    assert out["n_common"] == 0
    assert out["agreement_rate"] is None
    assert out["candidates_rule_determined"] is True


def test_parts_compare_judgment_and_confidence():
    prim = {
        "P3": {"judgment": "平静", "confidence": 0.8},
        "P12": {"judgment": "聚焦 NVDA", "confidence": 0.7},
    }
    gem = {
        "P3": {"judgment": "平静", "confidence": 0.6},
        "P12": {"judgment": "聚焦 MSFT", "confidence": 0.7},
    }
    out = compare_model_parts(prim, gem)
    assert out["n_compared"] == 2
    assert out["n_judgment_equal"] == 1  # P3 only
    assert out["mean_abs_confidence_delta"] == 0.1
    by_part = {r["part"]: r for r in out["rows"]}
    assert by_part["P3"]["confidence_delta"] == 0.2
    assert by_part["P12"]["judgment_equal"] is False


def test_parts_compare_missing_part_is_none_not_false():
    out = compare_model_parts({"P3": {"judgment": "x", "confidence": 0.5}}, {})
    assert out["n_compared"] == 0
    by_part = {r["part"]: r for r in out["rows"]}
    assert by_part["P3"]["judgment_equal"] is None
    assert by_part["P3"]["confidence_delta"] is None
    assert by_part["P12"]["primary_present"] is False


def test_summarize_real_shape_2026_11_13():
    # Mirrors the real 11-13 morning.json: identical candidates, but the
    # primary never returned P12 (fallback placeholder) while Gemini did.
    entry = summarize_model_agreement(_morning_doc())
    assert entry["date"] == "2026-11-13"
    assert entry["gemini_model"] == "gemini-3.5-flash"
    assert entry["shadow_ok"] is True
    assert entry["candidates"]["n_identical"] == 2
    assert entry["candidates"]["candidates_rule_determined"] is True
    parts = entry["parts"]
    assert parts["n_compared"] == 2
    assert parts["n_judgment_equal"] == 1  # P3 agrees; P12 differs
    by_part = {r["part"]: r for r in parts["rows"]}
    assert by_part["P12"]["judgment_equal"] is False
    assert by_part["P12"]["confidence_delta"] is None  # primary has none


def test_summarize_no_shadow():
    entry = summarize_model_agreement(_morning_doc(shadow_ok=False))
    assert entry["shadow_ok"] is False
    assert entry["candidates"] is None
    assert entry["parts"] is None


def test_summarize_empty_doc_does_not_crash():
    entry = summarize_model_agreement({})
    assert entry["shadow_ok"] is False
    assert entry["date"] is None


def test_aggregate_totals():
    entries = {
        "2026-11-13": summarize_model_agreement(_morning_doc()),
        "2026-11-12": summarize_model_agreement(
            _morning_doc(shadow_ok=False, trading_date="2026-11-12")),
    }
    out = aggregate_model_agreement(entries)
    assert out["totals"]["n_days"] == 2
    assert out["totals"]["n_days_with_shadow"] == 1
    assert out["totals"]["candidate_rows"] == 2
    assert out["totals"]["candidate_identical"] == 2
    assert out["totals"]["candidate_agreement_rate"] == 1.0
    assert out["totals"]["parts_compared"] == 2
    assert out["totals"]["parts_judgment_equal"] == 1
    assert [d["date"] for d in out["days"]] == ["2026-11-12", "2026-11-13"]


def test_aggregate_empty():
    out = aggregate_model_agreement({})
    assert out["days"] == []
    assert out["totals"]["n_days"] == 0
    assert out["totals"]["candidate_agreement_rate"] is None


def test_job_records_model_agreement_and_is_idempotent(tmp_path):
    state_dir = Path(tmp_path) / "state"
    reports = state_dir / "reports" / "2026-11-13"
    reports.mkdir(parents=True)
    (reports / "morning.json").write_text(
        json.dumps(_morning_doc()), encoding="utf-8")
    result = shadow_job_run(state_dir, date(2026, 11, 13), date(2026, 11, 13))
    assert result["dates_ingested"] == ["2026-11-13"]
    summary = json.loads(
        (state_dir / "paper" / "shadow_summary.json").read_text())
    section = summary["model_agreement"]
    assert section["totals"]["n_days"] == 1
    assert section["totals"]["n_days_with_shadow"] == 1
    assert section["totals"]["candidate_identical"] == 2
    assert section["days"][0]["gemini_model"] == "gemini-3.5-flash"
    ledger = json.loads(
        (state_dir / "paper" / "shadow_ledger.json").read_text())
    assert "2026-11-13" in ledger["model_agreement"]
    # Re-running must not duplicate the entry.
    shadow_job_run(state_dir, date(2026, 11, 13), date(2026, 11, 13))
    summary2 = json.loads(
        (state_dir / "paper" / "shadow_summary.json").read_text())
    assert summary2["model_agreement"]["totals"]["n_days"] == 1
