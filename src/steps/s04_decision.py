from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.steps.base import save_step_result
from src.research.format_body import normalize_body_md
from src.utils.paths import morning_json_path, step_json_path
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)

DISCLAIMER = "仅供参考，非交易指令 · ADVISORY_ONLY"


def run_step4_trade_decision(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    logger.info("Step 4 trade decision for %s", date_str)

    morning = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    s2 = json.loads(step_json_path(2, date_str).read_text()) if step_json_path(2, date_str).exists() else {}
    s3 = json.loads(step_json_path(3, date_str).read_text()) if step_json_path(3, date_str).exists() else {}

    total = morning.get("total_score") or 0
    s3_j = (s3.get("conclusion") or {}).get("judgment", "")
    if "新 Total：" in s3_j:
        try:
            total = int(s3_j.split("新 Total：")[-1].strip().split()[0])
        except ValueError:
            pass

    market = s2.get("market", "Mixed")
    edge = ((morning.get("parts") or {}).get("P13") or {}).get("judgment", "")
    should_trade = total >= 2 and market in ("Healthy", "Mixed") and "Edge：YES" in edge

    if should_trade:
        judgment = "Should trade：YES · QQQ Call"
        confidence = min(0.85, 0.55 + total * 0.03)
        one_liner = f"Risk 环境尚可，Total {total:+d}，建议 QQQ Call（观察用）"
        body = [
            "**Should we trade?** YES",
            "",
            "**Why?**",
            f"- Total {total:+d}",
            f"- 开盘 {market}",
            f"- Morning Bias {morning.get('bias', 'N/A')}",
            "",
            "**Trade**：QQQ Call",
            f"**Confidence**：{confidence:.0%}",
        ]
    else:
        judgment = "Should trade：NO"
        confidence = 0.65
        reasons = []
        if total < 2:
            reasons.append(f"Total {total:+d} 不足")
        if market == "Weak":
            reasons.append("开盘 Weak")
        if "Edge：YES" not in edge:
            reasons.append("无明显 Edge")
        one_liner = "；".join(reasons) or "条件未满足，不交易"
        body = [
            "**Should we trade?** NO",
            "",
            "**Why not?**",
            *[f"- {r}" for r in reasons],
        ]

    conclusion = {
        "part_id": "S4",
        "judgment": judgment,
        "confidence": confidence,
        "one_liner": one_liner[:256],
    }

    payload = save_step_result(
        4,
        trading_date,
        step_id="S4",
        job_id="trade_decision",
        conclusion=conclusion,
        body_md=normalize_body_md("\n".join(body)),
        extra={
            "should_trade": should_trade,
            "disclaimer": DISCLAIMER,
            "instrument": "QQQ Call" if should_trade else None,
        },
    )
    payload["disclaimer"] = DISCLAIMER
    return payload
