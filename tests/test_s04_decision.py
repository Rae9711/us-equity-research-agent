from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.steps.s04_decision import run_step4_trade_decision
from src.utils.paths import morning_json_path, report_dir, step_json_path


def _write_morning(trading_date: str, payload: dict) -> None:
    report_dir(trading_date).mkdir(parents=True, exist_ok=True)
    morning_json_path(trading_date).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


@patch("src.steps.s04_decision.guard_fresh_raw", return_value=(None, None))
@patch("src.steps.s04_decision._load_ev_inputs", return_value={"expected_return": 0.02, "expected_loss": -0.01, "risk_reward": 2.0})
@patch("src.steps.s04_decision.save_step_result")
def test_step4_stock_primary_when_index_no_trade(mock_save, _ev, _fresh, data_root):
    trading_date = "2099-01-02"
    _write_morning(
        trading_date,
        {
            "total_score": -3,
            "bias": "Neutral",
            "index_trade": "NO TRADE",
            "best_trades": {
                "index_trade": "NO TRADE",
                "primary": {
                    "symbol": "TSLA",
                    "direction": "SHORT",
                    "instrument": "TSLA 0DTE Put",
                    "confidence": 86,
                    "expected_move": "+2.52%",
                    "entry": "Below 409.8",
                    "stop": "420.1",
                    "target": "399.5",
                    "why_chain": "RS vs QQQ +0.54%",
                },
                "stock_trades": [
                    {
                        "rank": 1,
                        "symbol": "TSLA",
                        "direction": "SHORT",
                        "confidence": 86,
                        "expected_move": "+2.52%",
                    }
                ],
            },
        },
    )

    def _capture(_step, _d, **kwargs):
        return {"conclusion": kwargs["conclusion"], **kwargs.get("extra", {})}

    mock_save.side_effect = _capture

    payload = run_step4_trade_decision(date.fromisoformat(trading_date))

    assert payload["should_trade"] is True
    assert payload["index_trade"] == "NO TRADE"
    assert payload["stock_trade"] == "SHORT TSLA"
    assert "Index：NO TRADE" in payload["conclusion"]["judgment"]
    assert "Stock：SHORT TSLA" in payload["conclusion"]["judgment"]


@patch("src.steps.s04_decision.guard_fresh_raw", return_value=(None, None))
@patch("src.steps.s04_decision._load_ev_inputs", return_value={"expected_return": 0.02, "expected_loss": -0.01, "risk_reward": 2.0})
@patch("src.steps.s04_decision.save_step_result")
def test_step4_no_trade_without_primary(mock_save, _ev, _fresh, data_root):
    trading_date = "2099-01-05"
    _write_morning(
        trading_date,
        {
            "total_score": -1,
            "best_trades": {
                "index_trade": "NO TRADE",
                "primary": None,
                "threshold_message": "今日无任何标的达到交易阈值",
            },
            "parts": {"P13": {"judgment": "Edge：0/4 YES"}},
        },
    )
    step_json_path(2, trading_date).write_text(
        json.dumps({"market": "Mixed"}),
        encoding="utf-8",
    )

    def _capture(_step, _d, **kwargs):
        return {"conclusion": kwargs["conclusion"], **kwargs.get("extra", {})}

    mock_save.side_effect = _capture

    payload = run_step4_trade_decision(date.fromisoformat(trading_date))

    assert payload["should_trade"] is False
    assert payload["conclusion"]["judgment"] == "Should trade：NO"
