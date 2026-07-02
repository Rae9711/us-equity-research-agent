from __future__ import annotations

import re

_SCENARIO_HEAD = re.compile(
    r"^[\s\-•*]*Scenario\s*([ABC])\b[（(]?[^）):]*[）)]?\s*[：:]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_scenario_line(letter: str, rest: str) -> str:
    rest = rest.strip()
    rest = re.sub(r"^若\s*", "", rest)
    # Collapse "触发条件 → 反应 → 行动" chains into single 若…→…
    if "→" not in rest and "->" not in rest:
        if "：" in rest or ":" in rest:
            parts = re.split(r"[：:]", rest, maxsplit=1)
            if len(parts) == 2:
                rest = f"若 {parts[0].strip()} → {parts[1].strip()}"
        else:
            rest = f"若 {rest} → （待补充行动）"
    else:
        rest = rest.replace("->", "→")
        if not rest.startswith("若"):
            before_arrow, _, after = rest.partition("→")
            rest = f"若 {before_arrow.strip()} → {after.strip()}"
    return f"Scenario {letter}：{rest}"


def format_scenario_body(text: str, *, part_id: str = "P15") -> str:
    """
    Normalize P15/P16 body to WORKFLOW one-liner-per-scenario format, e.g.:
    Scenario A：若 NVDA 领涨 + QQQ 突破昨日高点 → 买 QQQ Call
    """
    if not text or not text.strip():
        return _empty_template(part_id)

    found: dict[str, str] = {}
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        m = _SCENARIO_HEAD.match(line)
        if m:
            found[m.group(1).upper()] = m.group(2).strip()
            continue
        # Inline "Scenario A ..." without colon on same line
        m2 = re.search(
            r"Scenario\s*([ABC])\b[（(]?[^）):]*[）)]?\s*[：:]\s*(.+)",
            line,
            re.I,
        )
        if m2:
            found[m2.group(1).upper()] = m2.group(2).strip()

    if found:
        lines = [
            _normalize_scenario_line(letter, found[letter])
            for letter in "ABC"
            if letter in found
        ]
        if len(lines) >= 2:
            return "\n\n".join(lines)

    # Fallback: split long prose into three bullets if A/B/C mentioned
    chunks = re.split(r"(?=Scenario\s*[ABC])", text, flags=re.I)
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) >= 2:
        lines = []
        for chunk in chunks[:3]:
            m = _SCENARIO_HEAD.match(chunk.split("\n")[0].strip()) or re.match(
                r"Scenario\s*([ABC]).*?[：:]\s*(.+)", chunk, re.I | re.S
            )
            if m:
                letter = m.group(1).upper()
                rest = m.group(2).split("\n")[0].strip()
                lines.append(_normalize_scenario_line(letter, rest))
        if lines:
            return "\n\n".join(lines)

    return text.strip()


def _empty_template(part_id: str) -> str:
    action_hint = "具体交易行动" if part_id == "P16" else "市场反应"
    return "\n\n".join(
        f"Scenario {letter}：若 （触发条件） → （{action_hint}）" for letter in "ABC"
    )
