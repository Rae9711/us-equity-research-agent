from __future__ import annotations

from typing import Any

from src.steps.meta import step_about, step_label


def render_step_markdown(payload: dict[str, Any]) -> str:
    step_num = payload.get("step_num", "?")
    step_id = payload.get("step_id", "")
    trading_date = payload.get("trading_date", "")
    conclusion = payload.get("conclusion") or {}
    body = payload.get("body_md") or ""

    conf = conclusion.get("confidence")
    conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "N/A"

    lines = [
        f"# Step {step_num} — {step_label(step_id)}",
        "",
        f"> {step_about(step_id)}",
        "",
        "| 字段 | 内容 |",
        "|------|------|",
        f"| **判断** | {conclusion.get('judgment', '—')} |",
        f"| **置信度** | {conf_s} |",
        f"| **一句话** | {conclusion.get('one_liner', '—')} |",
        "",
        body,
        "",
    ]
    if payload.get("disclaimer"):
        lines.extend(["---", "", f"*{payload['disclaimer']}*", ""])
    return "\n".join(lines)
