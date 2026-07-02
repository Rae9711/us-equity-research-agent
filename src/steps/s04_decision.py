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


def _load_ev_inputs(morning: dict, date_str: str) -> dict:
    """Load EV computation inputs from morning and context data."""
    from src.engines.bayesian import load_weights
    from src.engines.calibration import compute_ev

    total = morning.get("total_score") or 0
    confidence = 0.60
    regime_label = "Range"

    # Try to get regime from case
    try:
        from src.db.market_case_service import load_case
        case = load_case(date_str)
        regime_label = case.regime.label if case.regime else "Range"
    except Exception:
        pass

    weights = load_weights()
    ai_weight = weights.get("AI", 0.35)

    ev = compute_ev(
        morning_total=total,
        confidence=confidence,
        regime_label=regime_label,
        bayesian_ai_weight=ai_weight,
    )
    return ev


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

    ev_data = _load_ev_inputs(morning, date_str)

    if should_trade:
        confidence = min(0.85, 0.55 + total * 0.03)
        ev_rr = ev_data.get("risk_reward", 1.0)
        ev_ret = ev_data.get("expected_return", 0.0)
        ev_loss = ev_data.get("expected_loss", 0.0)
        judgment = (
            f"Should trade：YES · QQQ Call · "
            f"EV R:R {ev_rr:.1f} · Expected {ev_ret:.1%}"
        )
        one_liner = f"Risk 环境尚可，Total {total:+d}，R:R {ev_rr:.1f}，建议 QQQ Call（ADVISORY_ONLY）"
        body = [
            "**Should we trade?** YES",
            "",
            "**Why?**",
            f"- Total {total:+d}",
            f"- 开盘 {market}",
            f"- Morning Bias {morning.get('bias', 'N/A')}",
            "",
            "**Trade**：QQQ Call",
            f"**Expected Return**：{ev_ret:.1%}",
            f"**Expected Loss**：{ev_loss:.1%}",
            f"**Risk/Reward**：{ev_rr:.1f}",
            f"**Confidence**：{confidence:.0%}",
            f"**Win Rate Estimate**：{ev_data.get('win_rate_estimate', 0):.0%}",
            f"**Calibration**：{ev_data.get('calibration_source', 'heuristic')}",
            "",
            f"> {DISCLAIMER}",
        ]
    else:
        confidence = 0.65
        reasons = []
        if total < 2:
            reasons.append(f"Total {total:+d} 不足")
        if market == "Weak":
            reasons.append("开盘 Weak")
        if "Edge：YES" not in edge:
            reasons.append("无明显 Edge")
        one_liner = "；".join(reasons) or "条件未满足，不交易"
        judgment = "Should trade：NO"
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
            "expected_return": ev_data.get("expected_return"),
            "expected_loss": ev_data.get("expected_loss"),
            "risk_reward": ev_data.get("risk_reward"),
        },
    )
    payload["disclaimer"] = DISCLAIMER
    return payload
