"""Job: step8_learning — Step 8 at 20:00 ET.

Bayesian update + Playbook case storage + TrainingRow write.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.db import TrainingRow, get_session
from src.db.market_case_service import load_case, save_case, update_case
from src.engines.bayesian import load_weights, max_delta, update_weights
from src.engines.playbook import store_case
from src.steps.base import save_step_result
from src.utils.paths import step_json_path
from src.utils.trading_calendar import today_et

logger = logging.getLogger(__name__)


def run_step8_learning(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    logger.info("Step 8 Learning for %s", date_str)

    case = load_case(date_str)

    # 1. Bayesian Driver Update (verify overrides S7 driver when user corrected S7)
    attr_dict = case.attribution.model_dump() if case.attribution else {}
    from src.web.verify_learning import bayesian_driver_for_date

    actual_driver, run_bayesian = bayesian_driver_for_date(
        trading_date,
        (case.labels.actual_driver or "Unknown") if case.labels else "Unknown",
    )
    if run_bayesian:
        prior_weights, posterior_weights = update_weights(attr_dict, actual_driver)
    else:
        prior_weights = load_weights()
        posterior_weights = prior_weights
    big_driver, big_delta = max_delta(prior_weights, posterior_weights)

    # 2. Playbook Case Storage (includes human verify lessons in case.lesson)
    playbook_case_id = store_case(case) if (case.lesson or case.surprise) else None

    # 3. Training Row
    if case.features:
        features_dict = {
            k: v for k, v in case.features.model_dump().items() if v is not None
        }
        labels_dict = case.labels.model_dump() if case.labels else {}
        labels_dict["bayesian_prior_ai"] = prior_weights.get("AI")
        labels_dict["bayesian_post_ai"] = posterior_weights.get("AI")

        _write_training_row(date_str, features_dict, labels_dict)
    else:
        logger.info("No features available for training row on %s", date_str)

    # 4. Update Market Case with bayesian drivers + playbook refs
    playbook_refs = list(case.playbook_refs or [])
    if playbook_case_id and playbook_case_id not in playbook_refs:
        playbook_refs.append(playbook_case_id)

    update_case(date_str, {
        "bayesian_drivers": posterior_weights,
        "playbook_refs": playbook_refs,
    })

    # Build output
    bayesian_summary = (
        f"{big_driver} {'+' if big_delta >= 0 else ''}{big_delta * 100:.1f}%"
    )
    playbook_summary = (
        f"新增 {playbook_case_id}" if playbook_case_id else "无新 Case"
    )

    judgment = (
        f"Bayesian 最大调整：{bayesian_summary} · Playbook：{playbook_summary}"
    )
    one_liner = f"{big_driver} 后验调整 {big_delta * 100:.1f}%；{playbook_summary}"

    body_lines = [
        "## Step 8 — Learning",
        "",
        "### 8.1 Bayesian Driver Update",
        "",
        "| Driver | Prior | Posterior | Δ |",
        "|--------|------:|----------:|---|",
    ]
    for driver in prior_weights:
        prior_v = prior_weights.get(driver, 0.0)
        post_v = posterior_weights.get(driver, 0.0)
        delta = post_v - prior_v
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        body_lines.append(
            f"| {driver} | {prior_v:.1%} | **{post_v:.1%}** | {arrow} {abs(delta):.1%} |"
        )

    body_lines += [
        "",
        "### 8.2 Playbook",
        "",
        f"- {playbook_summary}",
        "",
        "### 8.3 Training Row",
        "",
        f"- Features + Labels written to `training_rows` for {date_str}",
    ]

    conclusion = {
        "part_id": "S8",
        "judgment": judgment,
        "confidence": None,
        "one_liner": one_liner[:256],
    }

    payload = save_step_result(
        8,
        trading_date,
        step_id="S8",
        job_id="learning",
        conclusion=conclusion,
        body_md="\n".join(body_lines),
        extra={
            "prior_weights": prior_weights,
            "posterior_weights": posterior_weights,
            "playbook_case_id": playbook_case_id,
            "actual_driver": actual_driver,
        },
    )
    return payload


def _write_training_row(date_str: str, features: dict, labels: dict) -> None:
    session = get_session()
    try:
        existing = (
            session.query(TrainingRow).filter(TrainingRow.date == date_str).first()
        )
        if existing:
            existing.features_json = json.dumps(features, ensure_ascii=False)
            existing.labels_json = json.dumps(labels, ensure_ascii=False)
        else:
            session.add(TrainingRow(
                date=date_str,
                features_json=json.dumps(features, ensure_ascii=False),
                labels_json=json.dumps(labels, ensure_ascii=False),
            ))
        session.commit()
        logger.info("Training row written for %s", date_str)
    except Exception as exc:
        session.rollback()
        logger.error("Failed to write training row for %s: %s", date_str, exc)
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_step8_learning()
    print(result.get("conclusion", {}).get("one_liner", ""))
