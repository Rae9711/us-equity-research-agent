from __future__ import annotations

import re


def normalize_body_md(text: str) -> str:
    """Turn LLM body text into readable markdown (lists + paragraph breaks)."""
    if not text or not text.strip():
        return "_（无补充分析）_"

    raw = text.strip().replace("\r\n", "\n")

    # Already structured markdown (headings, tables, bullets)
    if re.search(r"^#{1,3}\s", raw, re.M) or re.search(r"^\|.+\|", raw, re.M):
        return _ensure_blank_lines(raw)

    lines: list[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # Bullet-like prefixes
        if re.match(r"^[-*•·]\s+", line):
            lines.append(re.sub(r"^[•·]\s*", "- ", line))
            continue

        # Key-value rows (2Y：4.14% / Direction：Flat)
        kv = re.match(r"^(.+?)[：:]\s*(.+)$", line)
        if kv:
            lines.append(f"- **{kv.group(1).strip()}**：{kv.group(2).strip()}")
            continue

        # Score table rows (Macro | +1)
        if "|" in line and not line.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2:
                lines.append(f"- **{parts[0]}**：{parts[1]}")
                continue

        # Long run-on sentence — split on Chinese / English clause breaks
        if len(line) > 80 and ("，" in line or "。" in line or "；" in line):
            chunks = re.split(r"(?<=[，。；])", line)
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    lines.append(f"- {chunk}")
            continue

        lines.append(f"- {line}")

    return _ensure_blank_lines("\n".join(lines))


def _ensure_blank_lines(text: str) -> str:
    """Ensure markdown paragraphs/lists get proper spacing."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    # Single newline between non-list lines → double for paragraph break
    text = re.sub(r"(?<![\n|])\n(?![\n\-|*#])", "\n\n", text)
    return text
