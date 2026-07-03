"""L1 · 宏观数据实时抓取（CPI / PCE / NFP / ISM / JOLTS ...）。

- FRED 为主，BLS 兜底（NFP / 失业率）。
- 提供 `fetch_release(...)` 单点抓取，`check_todays_releases(...)` 遍历今日日历。
- 三次轮询节奏由 `config/macro_releases.yaml` 的 `poll_offsets_min` 控制。
- 无 API key、网络失败、数据尚未发布时全部优雅降级。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import yaml
from pytz import timezone

from src.collectors.fred_client import FredClient
from src.utils.paths import data_root, raw_dir
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)

ET = timezone("America/New_York")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_releases.yaml"


@lru_cache(maxsize=1)
def load_macro_release_config() -> dict[str, Any]:
    """加载今日发布日历配置；文件缺失返回空事件列表。"""
    if not _CONFIG_PATH.exists():
        return {"events": [], "poll_offsets_min": [2, 5, 10], "observation_history": 6}
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    data.setdefault("events", [])
    data.setdefault("poll_offsets_min", [2, 5, 10])
    data.setdefault("observation_history", 6)
    return data


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "."):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fred_observations(series_id: str, limit: int = 6) -> list[dict[str, Any]]:
    """最近 N 条观测（desc）。无 key / 失败返回空列表。"""
    client = FredClient()
    if not client.api_key:
        logger.debug("FRED_API_KEY 未设置，跳过 %s", series_id)
        return []
    try:
        data = client.get(
            "/series/observations",
            {
                "series_id": series_id,
                "limit": limit,
                "sort_order": "desc",
            },
        )
        return list(data.get("observations") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED %s 抓取失败: %s", series_id, exc)
        return []


def _bls_latest(series_id: str) -> dict[str, Any] | None:
    """BLS Public API v2 · 抓最新一期。BLS 无 key 也能用，但有配额。"""
    url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
    payload: dict[str, Any] = {"registrationkey": os.environ.get("BLS_API_KEY", "") or None}
    payload = {k: v for k, v in payload.items() if v}
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, params=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BLS %s 抓取失败: %s", series_id, exc)
        return None

    series = (data.get("Results") or {}).get("series") or []
    if not series:
        return None
    obs = series[0].get("data") or []
    if not obs:
        return None
    top = obs[0]
    return {
        "value": _safe_float(top.get("value")),
        "year": top.get("year"),
        "period": top.get("period"),
        "period_name": top.get("periodName"),
    }


def _surprise_pct(actual: float | None, consensus: float | None) -> float | None:
    if actual is None or consensus is None or consensus == 0:
        return None
    try:
        return round((actual - consensus) / abs(consensus) * 100, 2)
    except (TypeError, ZeroDivisionError):
        return None


def _observation_is_fresh(obs_date_str: str, target: date) -> bool:
    """观测归属期是否 ≥ 目标日所在月/季。粗略判断：观测日期年月 >= target 上月。"""
    try:
        obs_date = date.fromisoformat(obs_date_str)
    except (TypeError, ValueError):
        return False
    # 大部分宏观数据发布时归属 target 的上一自然月；给一个宽窗口
    return (obs_date.year, obs_date.month) >= (target.year, max(1, target.month - 2))


def fetch_release(
    event: dict[str, Any],
    target_date: date | None = None,
    *,
    history: int = 6,
) -> dict[str, Any] | None:
    """抓取单条宏观事件；返回统一结构或 None（获取不到任何值）。

    Returns:
        {
          event, label, fred_id, scheduled_et,
          actual, consensus, prior, surprise_pct,
          observation_date, released_at, direction, source,
          market_impact
        }
    """
    target_date = target_date or today_et()
    result: dict[str, Any] = {
        "event": event.get("id"),
        "label": event.get("label"),
        "fred_id": event.get("fred_id"),
        "bls_id": event.get("bls_id"),
        "scheduled_et": event.get("scheduled_et"),
        "unit": event.get("unit"),
        "direction": event.get("direction"),
        "actual": None,
        "consensus": _safe_float(event.get("consensus_default")),
        "prior": None,
        "surprise_pct": None,
        "observation_date": None,
        "released_at": None,
        "source": None,
        "market_impact": event.get("market_impact_default"),
    }

    obs_list = _fred_observations(event["fred_id"], limit=history) if event.get("fred_id") else []
    if obs_list:
        latest = obs_list[0]
        result["actual"] = _safe_float(latest.get("value"))
        result["observation_date"] = latest.get("date")
        result["released_at"] = latest.get("realtime_start")
        result["source"] = "fred"
        if len(obs_list) > 1:
            result["prior"] = _safe_float(obs_list[1].get("value"))

    if result["actual"] is None and event.get("bls_id"):
        bls = _bls_latest(event["bls_id"])
        if bls and bls.get("value") is not None:
            result["actual"] = bls["value"]
            period = bls.get("period_name") or ""
            year = bls.get("year") or ""
            result["observation_date"] = f"{year}-{period}" if year else None
            result["source"] = "bls"

    if result["actual"] is not None and result["consensus"] is not None:
        result["surprise_pct"] = _surprise_pct(result["actual"], result["consensus"])

    if result["actual"] is None:
        return None
    return result


def _status_for(release: dict[str, Any] | None, scheduled_et: str, target_date: date) -> str:
    """waiting / released / missing —— 完全基于观察日期新鲜度。"""
    if release is None or release.get("actual") is None:
        try:
            hh, mm = scheduled_et.split(":")
            sched = ET.localize(datetime(target_date.year, target_date.month, target_date.day,
                                         int(hh), int(mm)))
            now = datetime.now(ET)
            if now < sched:
                return "waiting"
            return "missing"
        except Exception:  # noqa: BLE001
            return "missing"

    obs_date = release.get("observation_date")
    if obs_date and isinstance(obs_date, str) and _observation_is_fresh(obs_date, target_date):
        return "released"
    return "waiting"


def check_todays_releases(target_date: date | None = None) -> list[dict[str, Any]]:
    """遍历今日日历，返回每个事件的状态。"""
    target_date = target_date or today_et()
    cfg = load_macro_release_config()
    history = int(cfg.get("observation_history", 6))
    results: list[dict[str, Any]] = []

    for event in cfg.get("events", []):
        release: dict[str, Any] | None = None
        try:
            release = fetch_release(event, target_date=target_date, history=history)
        except Exception:
            logger.exception("宏观事件 %s 抓取抛错", event.get("id"))

        status = _status_for(release, event.get("scheduled_et", ""), target_date)
        threshold = float(event.get("surprise_threshold_pct") or 0)
        surprise = (release or {}).get("surprise_pct")
        surprise_flag = bool(
            surprise is not None and abs(surprise) >= threshold and threshold > 0
        )

        item = {
            "event": event.get("id"),
            "label": event.get("label"),
            "scheduled_et": event.get("scheduled_et"),
            "status": status,
            "actual": (release or {}).get("actual"),
            "consensus": (release or {}).get("consensus"),
            "prior": (release or {}).get("prior"),
            "surprise_pct": surprise,
            "observation_date": (release or {}).get("observation_date"),
            "released_at": (release or {}).get("released_at"),
            "source": (release or {}).get("source"),
            "unit": event.get("unit"),
            "direction": event.get("direction"),
            "surprise_flag": surprise_flag,
            "surprise_threshold_pct": threshold,
            "market_impact": (release or {}).get("market_impact") or event.get("market_impact_default"),
        }
        results.append(item)

    return results


def releases_file_path(trading_date: date | str | None = None) -> Path:
    """`data/raw/YYYY-MM-DD-releases.json` — 与原有 raw 数据同目录。"""
    if trading_date is None:
        trading_date = today_et()
    if isinstance(trading_date, date):
        trading_date = trading_date.isoformat()
    return data_root() / "raw" / f"{trading_date}-releases.json"


def persist_releases(
    trading_date: date | None = None,
    *,
    releases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """把最新状态写入 `data/raw/YYYY-MM-DD-releases.json`。

    Idempotent：多次调用只会覆盖当日快照。
    """
    trading_date = trading_date or today_et()
    releases = releases if releases is not None else check_todays_releases(trading_date)

    payload = {
        "trading_date": trading_date.isoformat(),
        "generated_at": datetime.now(ET).isoformat(),
        "releases": releases,
        "counts": {
            "total": len(releases),
            "released": sum(1 for r in releases if r.get("status") == "released"),
            "waiting": sum(1 for r in releases if r.get("status") == "waiting"),
            "missing": sum(1 for r in releases if r.get("status") == "missing"),
            "surprise": sum(1 for r in releases if r.get("surprise_flag")),
        },
    }
    path = releases_file_path(trading_date)
    raw_dir()  # ensure directory exists before write
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_releases(trading_date: date | str | None = None) -> dict[str, Any] | None:
    """读取当日快照；缺失返回 None。"""
    path = releases_file_path(trading_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("解析 %s 失败", path)
        return None


def scheduled_release_slots(cfg: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    """去重 (hour, minute) 列表，用于 scheduler 派发轮询。"""
    cfg = cfg or load_macro_release_config()
    slots: set[tuple[int, int]] = set()
    for event in cfg.get("events", []):
        raw = str(event.get("scheduled_et") or "").strip()
        if not raw:
            continue
        try:
            hh, mm = raw.split(":")
            slots.add((int(hh), int(mm)))
        except (ValueError, AttributeError):
            continue
    return sorted(slots)
