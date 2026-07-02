from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.llm.anthropic_client import AnthropicClient
from src.research.format_body import normalize_body_md
from src.steps.base import save_step_result
from src.utils.paths import morning_json_path, step_json_path
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)

SYSTEM = """你是 Daily Trading OS 的 Step 3 Market Update Agent（10:00 ET）。
根据 Morning Research、Step 2 Opening Report 与最新新闻，判断盘中 Driver 是否切换，并给出更新后的 Total 与置信度。
输出 JSON：{"judgment":"...", "confidence":0.8, "one_liner":"...", "body_md":"..."}
judgment 格式：Driver 变了吗：YES/NO · 若变，新 Driver：{词} · 新 Total：{±N}
body_md 用 markdown 列表，含昨日→更新后 Score 表（若 Driver 未变可写 NO CHANGE）。
"""


def run_step3_market_update(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    logger.info("Step 3 market update for %s", date_str)

    morning = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    s2 = {}
    s2_path = step_json_path(2, date_str)
    if s2_path.exists():
        s2 = json.loads(s2_path.read_text(encoding="utf-8"))

    context = {
        "morning_bias": morning.get("bias"),
        "morning_total": morning.get("total_score"),
        "p10": (morning.get("parts") or {}).get("P10"),
        "s2_market": s2.get("market"),
        "s2_conclusion": s2.get("conclusion"),
    }

    conclusion: dict[str, Any]
    body_md: str
    try:
        client = AnthropicClient()
        result = client.complete_json(
            SYSTEM,
            f"上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}",
            max_tokens=4096,
        )
        conclusion = {
            "part_id": "S3",
            "judgment": result.get("judgment", "Driver 变了吗：NO"),
            "confidence": result.get("confidence", 0.7),
            "one_liner": result.get("one_liner", "盘中无重大 Driver 切换"),
        }
        body_md = normalize_body_md(result.get("body_md") or "")
    except Exception:
        logger.exception("Step 3 LLM failed, using rules fallback")
        total = morning.get("total_score") or 0
        s2_market = s2.get("market", "Mixed")
        adj = 1 if s2_market == "Healthy" else -1 if s2_market == "Weak" else 0
        new_total = total + adj
        conclusion = {
            "part_id": "S3",
            "judgment": f"Driver 变了吗：NO · 新 Total：{new_total:+d}",
            "confidence": 0.7,
            "one_liner": f"开盘 {s2_market}，Total 微调至 {new_total:+d}",
        }
        body_md = normalize_body_md(
            f"- 开盘状态：{s2_market}\n- Morning Total：{total:+d}\n- 更新后 Total：{new_total:+d}"
        )

    return save_step_result(
        3,
        trading_date,
        step_id="S3",
        job_id="market_update",
        conclusion=conclusion,
        body_md=body_md,
        extra={"context": context},
    )
