from __future__ import annotations

from typing import Any

from src.research.format_body import normalize_body_md
from src.research.format_scenarios import format_scenario_body
from src.research.parts_meta import MORNING_REPORT_ORDER, part_about, part_label


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
        "> Step 1 · 8:00 AM ET · ADVISORY ONLY — 不构成投资建议",
        "",
    ]
    if rule_meta:
        lines += [
            f"**综合 Bias**：{rule_meta.get('bias', 'N/A')} · **Total**：{rule_meta.get('total', 0):+d}",
            "",
        ]

    best = (rule_meta or {}).get("best_opportunity") or {}
    candidates = (rule_meta or {}).get("trade_candidates") or []
    if best:
        lines += _render_best_opportunity_section(best)
    if candidates:
        lines += _render_candidates_section(candidates)

    lines.append("## 结论总表")
    lines.append("")
    lines.append("| ID | Part | 判断 | 置信度 | 一句话 |")
    lines.append("|----|------|------|--------|--------|")
    for pid in MORNING_REPORT_ORDER:
        p = parts.get(pid) or {}
        lines.append(
            f"| {pid} | {part_label(pid)} | {p.get('judgment', '—')} | {_conf_str(p.get('confidence'))} | {p.get('one_liner', '—')} |"
        )
    lines.append("")

    lines += ["## 详细分析", ""]
    lines.append("以下为各 Part 展开说明；结论摘要见上表。")
    lines.append("")

    for pid in MORNING_REPORT_ORDER:
        p = parts.get(pid) or {}
        if pid in ("P15", "P16"):
            body = format_scenario_body(p.get("body_md") or "", part_id=pid)
        elif pid == "P17":
            body = _format_hypothesis_body(p.get("hypothesis") or {})
        elif pid == "P18":
            body = p.get("body_md") or ""
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


def _render_best_opportunity_section(best: dict[str, Any]) -> list[str]:
    why = best.get("why_chain") or " · ".join(best.get("why") or [])
    avoid = ", ".join(best.get("avoid") or [])
    return [
        "## 🥇 Best Opportunity Today",
        "",
        "> ADVISORY — 不构成投资建议",
        "",
        f"- **Direction**：{best.get('direction', '—')}",
        f"- **Symbol**：{best.get('symbol', '—')}",
        f"- **Instrument**：{best.get('instrument', '—')}",
        f"- **Confidence**：{best.get('confidence', '—')}%",
        f"- **Entry**：{best.get('entry', '—')}",
        f"- **Stop**：{best.get('stop', '—')}",
        f"- **Target**：{best.get('target', '—')}",
        f"- **Duration**：{best.get('duration', '—')}",
        f"- **Why**：{why}",
        f"- **Avoid**：{avoid}",
        f"- **One-liner**：{best.get('one_liner', '—')}",
        "",
    ]


def _render_candidates_section(candidates: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## P18 Trade Candidates",
        "",
        "> ADVISORY — 不构成投资建议",
        "",
        "| Rank | Symbol | Win% | ER% | R:R | Score | Trade |",
        "|------|--------|------|-----|-----|-------|-------|",
    ]
    for row in candidates:
        lines.append(
            f"| {row.get('rank', '—')} | {row.get('symbol', '—')} | "
            f"{row.get('win_prob', row.get('score', '—'))} | "
            f"{row.get('expected_return_pct', '—')} | "
            f"{row.get('risk_reward', '—')} | "
            f"{row.get('final_score', row.get('score', '—'))} | "
            f"{row.get('trade_action', row.get('trade', '—'))} |"
        )
    lines.append("")
    return lines


def _format_hypothesis_body(hypothesis: dict[str, Any]) -> str:
    if not hypothesis:
        return "_Hypothesis 未生成_"
    lines = [
        f"**Hypothesis ID**：`{hypothesis.get('id', '—')}`",
        "",
        f"**陈述**：{hypothesis.get('statement', '—')}",
        "",
        f"**状态**：{hypothesis.get('status', '待验证')}",
        "",
        "**Evidence**",
        "",
    ]
    for item in hypothesis.get("evidence") or []:
        lines.append(f"- {item}")
    if not hypothesis.get("evidence"):
        lines.append("- —")
    lines += ["", "**Counter-evidence**", ""]
    for item in hypothesis.get("counter_evidence") or []:
        lines.append(f"- {item}")
    if not hypothesis.get("counter_evidence"):
        lines.append("- —")
    return "\n".join(lines)
