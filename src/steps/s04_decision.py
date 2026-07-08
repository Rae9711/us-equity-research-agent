from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.steps.base import save_step_result
from src.research.format_body import normalize_body_md
from src.utils.data_freshness import guard_fresh_raw
from src.utils.paths import morning_json_path, step_json_path
from src.utils.pit_snapshots import save_snapshot, step_label
from src.utils.trading_calendar import require_trading_day, skipped_non_trading_day

logger = logging.getLogger(__name__)

DISCLAIMER = "仅供参考，非交易指令 · ADVISORY_ONLY"


def _load_ev_inputs(morning: dict, date_str: str) -> dict:
    """Load EV computation inputs from morning and context data."""
    from src.engines.bayesian import load_weights
    from src.engines.calibration import compute_ev

    total = morning.get("total_score") or 0
    confidence = 0.60
    regime_label = "Range"

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


def _stock_trade_label(primary: dict[str, Any] | None) -> str | None:
    if not primary:
        return None
    direction = primary.get("direction")
    symbol = primary.get("symbol")
    if direction in ("LONG", "SHORT") and symbol:
        return f"{direction} {symbol}"
    return None


def _resolve_trades(morning: dict[str, Any]) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    """Read Decision Agent v2 lanes from morning.json."""
    best_trades = morning.get("best_trades") or {}
    index_trade = morning.get("index_trade") or best_trades.get("index_trade") or "NO TRADE"
    primary = best_trades.get("primary")
    stock_trades = morning.get("stock_trades") or best_trades.get("stock_trades") or []
    return index_trade, primary, stock_trades


def run_step4_trade_decision(trading_date: date | None = None) -> dict[str, Any]:
    d = require_trading_day(trading_date, job="run_step4_trade_decision")
    if d is None:
        return skipped_non_trading_day(trading_date)
    trading_date = d
    date_str = trading_date.isoformat()
    logger.info("Step 4 trade decision for %s", date_str)

    _, stale = guard_fresh_raw(trading_date, step="run_step4_trade_decision")
    if stale:
        return stale

    morning: dict[str, Any] = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    index_trade, primary, stock_trades = _resolve_trades(morning)
    stock_label = _stock_trade_label(primary)
    has_index_trade = index_trade not in (None, "", "NO TRADE")
    has_stock_trade = stock_label is not None

    ev_data = _load_ev_inputs(morning, date_str)
    total = morning.get("total_score") or 0

    if has_index_trade or has_stock_trade:
        parts: list[str] = []
        if has_index_trade:
            parts.append(f"Index：{index_trade}")
        else:
            parts.append("Index：NO TRADE")
        if has_stock_trade:
            parts.append(f"Stock：{stock_label}")
        judgment = " · ".join(parts)

        confidence = 0.65
        if primary and primary.get("confidence") is not None:
            try:
                confidence = min(0.95, float(primary["confidence"]) / 100.0)
            except (TypeError, ValueError):
                pass
        elif has_index_trade:
            confidence = min(0.85, 0.55 + max(total, 0) * 0.03)

        one_liner_bits: list[str] = []
        if has_index_trade:
            one_liner_bits.append(f"指数 {index_trade}")
        elif has_stock_trade:
            one_liner_bits.append("指数观望")
        if has_stock_trade:
            one_liner_bits.append(f"个股 {stock_label}")
        one_liner = "；".join(one_liner_bits) + f"（{DISCLAIMER}）"

        body = [
            f"**Index Trade**：{index_trade if has_index_trade else 'NO TRADE'}",
            "",
        ]
        if has_stock_trade and primary:
            body.extend(
                [
                    f"**Stock Trade**：{stock_label}",
                    f"**Instrument**：{primary.get('instrument', '—')}",
                    f"**Confidence**：{primary.get('confidence', '—')}%",
                    f"**Expected Move**：{primary.get('expected_move', '—')}",
                    f"**Entry**：{primary.get('entry', '—')}",
                    f"**Stop**：{primary.get('stop', '—')}",
                    f"**Target**：{primary.get('target', '—')}",
                    "",
                    "**Why**",
                    f"- {primary.get('why_chain') or '；'.join(primary.get('why_factors') or []) or '—'}",
                    "",
                ]
            )
        if has_index_trade:
            ev_rr = ev_data.get("risk_reward", 1.0)
            ev_ret = ev_data.get("expected_return", 0.0)
            ev_loss = ev_data.get("expected_loss", 0.0)
            body.extend(
                [
                    "**Index context**",
                    f"- Total {total:+d}",
                    f"- Morning Bias {morning.get('bias', 'N/A')}",
                    f"- Expected Return {ev_ret:.1%}",
                    f"- Expected Loss {ev_loss:.1%}",
                    f"- Risk/Reward {ev_rr:.1f}",
                    "",
                ]
            )
        body.append(f"> {DISCLAIMER}")

        should_trade = True
        instrument = primary.get("instrument") if primary else index_trade
    else:
        s2 = (
            json.loads(step_json_path(2, date_str).read_text())
            if step_json_path(2, date_str).exists()
            else {}
        )
        s3 = (
            json.loads(step_json_path(3, date_str).read_text())
            if step_json_path(3, date_str).exists()
            else {}
        )

        if "新 Total：" in ((s3.get("conclusion") or {}).get("judgment", "")):
            try:
                total = int(
                    (s3.get("conclusion") or {})["judgment"]
                    .split("新 Total：")[-1]
                    .strip()
                    .split()[0]
                )
            except ValueError:
                pass

        market = s2.get("market", "Mixed")
        edge = ((morning.get("parts") or {}).get("P13") or {}).get("judgment", "")
        threshold_msg = (morning.get("best_trades") or {}).get("threshold_message")
        should_trade = False
        confidence = 0.65
        reasons: list[str] = []
        if threshold_msg:
            reasons.append(threshold_msg)
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
            "",
            f"> {DISCLAIMER}",
        ]
        instrument = None
        index_trade = "NO TRADE"

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
            "index_trade": index_trade,
            "stock_trade": stock_label,
            "stock_trades": stock_trades,
            "primary": primary,
            "instrument": instrument,
            "expected_return": ev_data.get("expected_return"),
            "expected_loss": ev_data.get("expected_loss"),
            "risk_reward": ev_data.get("risk_reward"),
        },
    )
    payload["disclaimer"] = DISCLAIMER
    save_snapshot(
        trading_date,
        step_label(4),
        {"step4": payload, "morning": morning},
    )
    return payload
