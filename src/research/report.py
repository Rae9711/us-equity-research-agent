from __future__ import annotations

from typing import Any

from src.research.format_body import normalize_body_md
from src.research.format_scenarios import format_scenario_body
from src.research.parts_meta import PART_ORDER, part_about, part_label


def _conf_str(conf: Any) -> str:
    return f"{conf:.0%}" if isinstance(conf, (int, float)) else "N/A"


def render_morning_report(
    trading_date: str,
    parts: dict[str, dict[str, Any]],
    rule_meta: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"# Morning Research — {trading_date}",
        "",
        "> Step 1 · 8:00 AM ET · ADVISORY ONLY",
        "",
    ]
    if rule_meta:
        lines += [
            f"**综合 Bias**：{rule_meta.get('bias', 'N/A')} · **Total**：{rule_meta.get('total', 0):+d}",
            "",
        ]

    lines.append("## 结论总表")
    lines.append("")
    lines.append("| Part | 判断 | 置信度 | 一句话 |")
    lines.append("|------|------|--------|--------|")
    for pid in PART_ORDER:
        p = parts.get(pid) or {}
        lines.append(
            f"| {part_label(pid)} | {p.get('judgment', '—')} | {_conf_str(p.get('confidence'))} | {p.get('one_liner', '—')} |"
        )
    lines.append("")

    lines += ["## 详细分析", ""]
    lines.append("以下为各 Part 展开说明；结论摘要见上表。")
    lines.append("")

    for pid in PART_ORDER:
        p = parts.get(pid) or {}
        if pid in ("P15", "P16"):
            body = format_scenario_body(p.get("body_md") or "", part_id=pid)
        else:
            body = normalize_body_md(p.get("body_md") or "")

        lines += [
            f"### {part_label(pid)}",
            "",
            f"> {part_about(pid)}",
            "",
            "| 字段 | 内容 |",
            "|------|------|",
            f"| **判断** | {p.get('judgment', '—')} |",
            f"| **置信度** | {_conf_str(p.get('confidence'))} |",
            f"| **一句话** | {p.get('one_liner', '—')} |",
            "",
            body,
            "",
        ]

    return "\n".join(lines)
