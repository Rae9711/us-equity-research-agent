from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "/data"))


def raw_dir() -> Path:
    d = data_root() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = data_root() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_data_path(trading_date: str) -> Path:
    return raw_dir() / f"{trading_date}.json"


def report_dir(trading_date: str) -> Path:
    d = reports_dir() / trading_date
    d.mkdir(parents=True, exist_ok=True)
    return d


def morning_report_path(trading_date: str) -> Path:
    return report_dir(trading_date) / "morning.md"


def morning_json_path(trading_date: str) -> Path:
    return report_dir(trading_date) / "morning.json"


def step_json_path(step_num: int, trading_date: str) -> Path:
    return report_dir(trading_date) / f"step{step_num}.json"


def step_report_path(step_num: int, trading_date: str) -> Path:
    return report_dir(trading_date) / f"step{step_num}.md"


def case_json_path(trading_date: str) -> Path:
    return report_dir(trading_date) / "case.json"
