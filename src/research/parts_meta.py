from __future__ import annotations

PART_ORDER = [f"P{i}" for i in range(1, 17)]
MORNING_REPORT_ORDER = ["R0"] + PART_ORDER + ["P17"]

# title: 表格与章节标题；about: 这个 Part 是干什么的
PART_META: dict[str, dict[str, str]] = {
    "R0": {
        "title": "Regime Engine",
        "about": "决策级市场时代（AI Expansion / Macro Fear / …），先于 P1–P16",
    },
    "Step0": {
        "title": "数据更新",
        "about": "Step 0 Raw Data 是否就绪，缺失项有哪些",
    },
    "P1": {
        "title": "昨天发生了什么？",
        "about": "汇总上一交易日指数、宏观与新闻事实，给出昨日定性（Risk-on/off/Neutral/分化）",
    },
    "P2": {
        "title": "昨天为什么涨/跌？",
        "about": "找出昨日最大 Driver，写清因果链（事实 vs 解释的分界）",
    },
    "P3": {
        "title": "今天市场最关心什么？",
        "about": "列出今日/本周重要事件与重要性，确定本周首要关注",
    },
    "P4": {
        "title": "债券 Bond",
        "about": "2Y/10Y 利率方向 → Growth 判断，输出 Morning Score",
    },
    "P5": {
        "title": "美元 Dollar",
        "about": "美元指数强弱 → 对 Risk/QQQ 的影响，输出 Morning Score",
    },
    "P6": {
        "title": "波动率 VIX",
        "about": "VIX 升降与恐慌情绪 → Risk-on 是否可信，输出 Morning Score",
    },
    "P7": {
        "title": "板块轮动 Sector",
        "about": "昨日各板块相对强弱，判断最强板块与轮动类型（普涨/普跌/分化）",
    },
    "P8": {
        "title": "AI 主题",
        "about": "AI/Mag7/半导体叙事：偏多、中性还是偏空，及主要理由与风险",
    },
    "P9": {
        "title": "期权 Options",
        "about": "四问：适合买期权/0DTE/Call/Put（YES/NO），事件日禁止 0DTE",
    },
    "P10": {
        "title": "今日 Driver",
        "about": "Driver Type（分类）+ Driver（具体标签）；无宏观催化剂日禁止写 Macro",
    },
    "P11": {
        "title": "综合打分 Scoring",
        "about": "宏观/债券/美元/VIX/AI 等因子加权 → Total 与 Bias（牛/熊/中性）",
    },
    "P12": {
        "title": "市场状态 Regime",
        "about": "趋势、波动、风险、流动性 → 今天是不是趋势确认日",
    },
    "P13": {
        "title": "Edge 优势",
        "about": "今天有没有信息优势（重大数据/CPI/PCE/FOMC 等催化剂）",
    },
    "P14": {
        "title": "市场偏好 Preference",
        "about": "结构化偏好（AI > Defensive、Growth > Value、Momentum > Mean Reversion），非新闻罗列",
    },
    "P15": {
        "title": "情景 Scenario",
        "about": "三种 Scenario，每行：Scenario X：若 触发条件 → 市场反应",
    },
    "P16": {
        "title": "交易计划 Trading Plan",
        "about": "IF-THEN 执行计划，每行：Scenario X：若 条件 → 行动（买 Call / 放弃 / 不追）",
    },
    "P17": {
        "title": "Hypothesis",
        "about": "可验证假设：Evidence / Counter / 状态（待验证 → 盘中更新）",
    },
}

# 向后兼容
PART_TITLES = {pid: m["title"] for pid, m in PART_META.items()}


def part_label(part_id: str) -> str:
    meta = PART_META.get(part_id, {})
    title = meta.get("title", part_id)
    return f"{part_id} {title}" if meta else part_id


def part_about(part_id: str) -> str:
    return PART_META.get(part_id, {}).get("about", "")
