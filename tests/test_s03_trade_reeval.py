"""Tests for Step 3 trade_reeval persistence."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

from src.steps.s03_update import run_step3_market_update


@patch("src.steps.s03_update.save_snapshot")
@patch("src.steps.s03_update.save_step_result")
@patch("src.steps.s03_update.guard_fresh_raw")
@patch("src.steps.s03_update.require_trading_day")
@patch("src.steps.s03_update._headlines_since_open", return_value=([], []))
@patch("src.steps.s03_update._macro_releases_for_step3", return_value=[])
@patch("src.steps.s03_update.detect_high_signal_news", return_value=[])
@patch("src.steps.s03_update.summarize_signals", return_value={"has_high": False, "high": 0})
def test_step3_includes_trade_reeval_when_morning_exists(
    _sum,
    _det,
    _macro,
    _head,
    mock_require,
    mock_guard,
    mock_save,
    _snap,
    tmp_path,
    monkeypatch,
):
    trading_date = date(2026, 7, 13)
    mock_require.return_value = trading_date
    mock_guard.return_value = (True, None)

    morning = {
        "bias": "Bearish",
        "total_score": -1,
        "parts": {"P16": {"judgment": "No Trade"}},
        "best_trades": {
            "primary": {
                "symbol": "MU",
                "direction": "SHORT",
                "entry_zone": {
                    "low": 910.0,
                    "mid": 913.0,
                    "high": 916.0,
                    "display": "910–916",
                },
                "entry_price": 913.0,
                "stop_price": 954.0,
                "target_price": 899.29,
                "level_anchors": {"vwap": 928.0},
            }
        },
        "top_trades": [],
        "edges": {},
    }

    data_root = tmp_path / "data"
    reports = data_root / "reports" / "2026-07-13"
    raw_dir = data_root / "raw"
    reports.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    (reports / "morning.json").write_text(json.dumps(morning), encoding="utf-8")
    (raw_dir / "2026-07-13.json").write_text(
        '{"trading_date":"2026-07-13","stocks":{"quotes":{}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    captured: dict = {}

    def _save(step_num, td, **kwargs):
        captured["extra"] = kwargs.get("extra") or {}
        return {
            "step_num": step_num,
            "trading_date": td.isoformat(),
            "conclusion": kwargs.get("conclusion"),
            "body_md": kwargs.get("body_md"),
            **captured["extra"],
        }

    mock_save.side_effect = _save

    monkeypatch.setattr(
        "src.research.entry_status.resolve_symbol_last",
        lambda *a, **k: 933.0,
    )
    monkeypatch.setattr(
        "src.research.entry_status.build_session_trade_update",
        lambda morning, trading_date, raw=None, session_phase="open": {
            "as_of": "10:00",
            "changed": False,
            "primary_changed": False,
            "why_changed": "#1 unchanged vs Morning: MU",
            "note": "#1 unchanged vs Morning: MU",
            "primary": {"symbol": "MU", "direction": "SHORT", "remaining_er_pct": 3.64},
            "top_trades": [{"symbol": "MU", "rank": 1, "remaining_er_pct": 3.64}],
            "morning_primary_live": {
                "symbol": "MU",
                "remaining_er_pct": 3.64,
                "live_expected_return_pct": 3.64,
                "entry_status": {"status": "MISSED"},
            },
            "compared_to_morning_primary": {"symbol": "MU", "entry_status": "MISSED"},
            "advisory": True,
        },
    )

    with patch("src.steps.s03_update.AnthropicClient") as mock_client:
        mock_client.return_value.complete_json.return_value = {
            "judgment": "Driver 变了吗：NO · 新 Total：-1",
            "confidence": 0.7,
            "one_liner": "NO CHANGE",
            "body_md": "- Driver NO CHANGE",
        }
        payload = run_step3_market_update(trading_date)

    assert payload.get("trade_reeval") is not None or payload.get("session_trade_update") is not None
    reeval = payload.get("trade_reeval") or payload.get("session_trade_update")
    assert reeval.get("note") or reeval.get("why_changed")
    body = payload.get("body_md") or ""
    assert "10:00" in body and ("Re-eval" in body or "Session Update" in body)
