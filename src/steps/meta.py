from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepDef:
    num: int
    step_id: str
    time_et: str
    title: str
    subtitle: str
    job_id: str


STEPS: list[StepDef] = [
    StepDef(0, "Step0", "7:45", "数据更新", "Raw Data 采集", "collect_raw"),
    StepDef(1, "Step1", "8:00", "Morning Research", "R0 → P1–P16 → P17 → P18 Decision", "morning_research"),
    StepDef(2, "S2", "9:30", "Open", "开盘观察 · 不交易", "open_report"),
    StepDef(3, "S3", "10:00", "Market Update", "Driver 切换 · 重算 Score", "market_update"),
    StepDef(4, "S4", "10:15", "Trading Decision", "建议 only · ADVISORY", "trade_decision"),
    StepDef(5, "S5", "12:00", "Midday Review", "Driver 是否仍成立", "midday_review"),
    StepDef(6, "S6", "14:00", "Afternoon Review", "Trade still valid?", "afternoon_review"),
    StepDef(7, "S7", "16:10", "Attribution", "涨跌归因 · Surprise · Decision Log", "evening_review"),
    StepDef(8, "S8", "20:00", "Learning", "Bayesian · Playbook · 训练行", "learning"),
]

STEP_BY_NUM = {s.num: s for s in STEPS}
STEP_BY_ID = {s.step_id: s for s in STEPS}
INTRADAY_STEPS = {2, 3, 4, 5, 6, 7, 8}

STEP_LABELS: dict[str, dict[str, str]] = {
    "S2": {
        "title": "9:30 Open",
        "about": "不交易只观察：Gap / Breadth / Leader / Bond，验证 Morning 叙事",
    },
    "S3": {
        "title": "10:00 Market Update",
        "about": "盘中 Driver 是否切换，重新计算 Score 与置信度",
    },
    "S4": {
        "title": "10:15 Trading Decision",
        "about": "Should we trade? 仅供参考，非交易指令",
    },
    "S5": {
        "title": "12:00 Midday Review",
        "about": "Midday Driver 是否仍等于 Morning Driver",
    },
    "S6": {
        "title": "14:00 Afternoon Review",
        "about": "交易逻辑是否仍成立，Bond 突发监控",
    },
    "S7": {
        "title": "16:10 Evening Review",
        "about": "今日真正 Driver、Morning 对错、Decision Log",
    },
    "S8": {
        "title": "20:00 Learning",
        "about": "Driver 权重调整与 Playbook 更新",
    },
}


def step_label(step_id: str) -> str:
    if step_id == "Step0":
        return "Step 0 数据更新"
    if step_id == "Step1":
        return "Step 1 Morning Research"
    meta = STEP_LABELS.get(step_id, {})
    return f"{step_id} {meta.get('title', step_id)}"


def step_about(step_id: str) -> str:
    if step_id == "Step0":
        from src.research.parts_meta import part_about as pa

        return pa("Step0")
    if step_id == "Step1":
        return "Morning Research P1–P16 完整晨会流程"
    return STEP_LABELS.get(step_id, {}).get("about", "")
