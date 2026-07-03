"""P15 Scenario assessment — compare morning scenarios vs end-of-day market."""

from __future__ import annotations

import re
from typing import Any

from src.schemas.market_case import FeaturesModel

_SCENARIO_HEAD = re.compile(
    r"Scenario\s*([ABC])\b[（(]?[^）):]*[）)]?\s*[：:]\s*(.+)$",
    re.IGNORECASE,
)
_PRIMARY_RE = re.compile(
    r"最可能\s*Scenario[：:]\s*([ABC])\b",
    re.IGNORECASE,
)

_BULL_KW = ("涨", "多", "突破", "bull", "risk-on", "领涨", "上行")
_BEAR_KW = ("跌", "空", "放弃", "bear", "risk-off", "回落", "下行", "减仓")
_CHOP_KW = ("震荡", "range", "横盘", "不交易", "观望", "chop", "中性")


def parse_primary_scenario(judgment: str) -> str | None:
    """Extract morning's primary scenario letter from P15 judgment."""
    if not judgment:
        return None
    m = _PRIMARY_RE.search(judgment)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([ABC])\b", judgment)
    if m2 and "scenario" in judgment.lower():
        return m2.group(1).upper()
    return None


def parse_scenarios(body_md: str) -> dict[str, str]:
    """Parse P15 body into {A: trigger_text, B: ..., C: ...}."""
    found: dict[str, str] = {}
    for line in (body_md or "").replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        m = _SCENARIO_HEAD.match(line)
        if m:
            found[m.group(1).upper()] = m.group(2).strip()
    return found


def _archetype(text: str) -> str:
    """Classify scenario text as bullish / bearish / chop."""
    t = text.lower()
    bull = sum(1 for kw in _BULL_KW if kw in t)
    bear = sum(1 for kw in _BEAR_KW if kw in t)
    chop = sum(1 for kw in _CHOP_KW if kw in t)
    if chop >= bull and chop >= bear and chop > 0:
        return "chop"
    if bull > bear:
        return "bull"
    if bear > bull:
        return "bear"
    return "neutral"


def _day_archetype(features: FeaturesModel) -> str:
    qqq = features.qqq_chg
    if qqq is None:
        return "neutral"
    if qqq >= 0.35:
        return "bull"
    if qqq <= -0.35:
        return "bear"
    if abs(qqq) <= 0.25:
        return "chop"
    return "neutral"


def _trigger_score(trigger: str, features: FeaturesModel, raw: dict[str, Any] | None) -> float:
    """Heuristic 0–1 score for how well trigger conditions matched the day."""
    if not trigger:
        return 0.0

    t = trigger.lower()
    score_parts: list[float] = []
    clauses = re.split(r"\s*[+＋]\s*|且|并且", trigger)
    if len(clauses) <= 1:
        clauses = [trigger]

    for clause in clauses:
        c = clause.strip().lower()
        if not c:
            continue
        matched: float | None = None

        if re.search(r"nfp|非农|employment|payroll", c):
            qqq = features.qqq_chg
            smh = features.smh_chg
            if "弱" in c or "低于" in c or "下行" in c:
                matched = 1.0 if (qqq is not None and qqq > 0) or (smh is not None and smh < 0) else 0.0
            elif "强" in c or "高于" in c:
                matched = 1.0 if qqq is not None and qqq < 0 else 0.0

        elif re.search(r"nvda|英伟达", c):
            nvda = features.nvda_chg
            smh = features.smh_chg
            if nvda is not None:
                if "领涨" in c or "强" in c:
                    matched = 1.0 if nvda > 0 and (smh is None or nvda >= smh) else 0.0
                elif "跌" in c or "回落" in c:
                    matched = 1.0 if nvda < 0 else 0.0
                else:
                    matched = 1.0 if nvda > 0 else 0.0

        elif re.search(r"qqq|纳指|ndx", c):
            qqq = features.qqq_chg
            if qqq is not None:
                if "突破" in c or "破高" in c or "新高" in c:
                    matched = 1.0 if _qqq_broke_high(raw, features) else 0.0
                elif "跌" in c or "回落" in c:
                    matched = 1.0 if qqq < 0 else 0.0
                else:
                    matched = 1.0 if qqq > 0 else 0.0

        elif re.search(r"smh|semiconductor|半导体", c):
            smh = features.smh_chg
            if smh is not None:
                if "回落" in c or "跌" in c:
                    matched = 1.0 if smh < 0 else 0.0
                else:
                    matched = 1.0 if smh > 0 else 0.0

        elif re.search(r"10y|利率|bond|美债", c):
            dgs10_chg = _dgs10_chg(raw)
            if dgs10_chg is not None:
                if "涨" in c or "升" in c or "上行" in c:
                    matched = 1.0 if dgs10_chg > 0 else 0.0
                elif "跌" in c or "降" in c:
                    matched = 1.0 if dgs10_chg < 0 else 0.0

        elif re.search(r"vix|波动", c):
            vix = features.vix_chg
            if vix is not None:
                if "升" in c or "涨" in c or "恐慌" in c:
                    matched = 1.0 if vix > 0 else 0.0
                elif "降" in c or "跌" in c:
                    matched = 1.0 if vix < 0 else 0.0

        elif re.search(r"震荡|range|横盘", c):
            matched = 1.0 if _day_archetype(features) == "chop" else 0.0

        elif re.search(r"涨|多|突破", c):
            qqq = features.qqq_chg
            matched = 1.0 if qqq is not None and qqq > 0 else 0.0

        elif re.search(r"跌|空|放弃", c):
            qqq = features.qqq_chg
            matched = 1.0 if qqq is not None and qqq < 0 else 0.0

        if matched is not None:
            score_parts.append(matched)

    if not score_parts:
        return 0.0
    return sum(score_parts) / len(score_parts)


def _qqq_broke_high(raw: dict[str, Any] | None, features: FeaturesModel) -> bool:
    if not raw:
        return features.qqq_chg is not None and features.qqq_chg > 0.5
    market = raw.get("market") or {}
    qqq = (market.get("quotes") or market.get("prices") or {}).get("QQQ") or {}
    close = qqq.get("close") or qqq.get("current_price")
    high = qqq.get("high")
    prior = qqq.get("prev_close")
    if close and high and prior:
        try:
            return float(close) >= float(prior) and float(high) > float(prior)
        except (TypeError, ValueError):
            pass
    return features.qqq_chg is not None and features.qqq_chg > 0.5


def _dgs10_chg(raw: dict[str, Any] | None) -> float | None:
    if not raw:
        return None
    macro = raw.get("macro") or {}
    cur = (macro.get("rates") or {}).get("DGS10")
    if cur is None:
        cur = (raw.get("market") or {}).get("treasury_10y_fred", {}).get("value")
    prior_raw = raw.get("_prior_macro") or {}
    prior = (prior_raw.get("rates") or {}).get("DGS10")
    try:
        if cur is not None and prior is not None:
            return float(cur) - float(prior)
    except (TypeError, ValueError):
        pass
    return None


def _reaction_text(scenario_line: str) -> str:
    if "→" in scenario_line:
        return scenario_line.split("→", 1)[1].strip()
    if "->" in scenario_line:
        return scenario_line.split("->", 1)[1].strip()
    return scenario_line


def assess_scenarios(
    *,
    p15_judgment: str,
    p15_body: str,
    features: FeaturesModel,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare morning P15 vs end-of-day outcome.

    Returns:
        scenario_primary: morning's pick (A/B/C or None)
        scenario_actual: best-matching scenario (A/B/C/none)
        scenario_correct: bool — primary matched actual
        scenario_scores: per-letter trigger scores
        detection_method: auto
    """
    primary = parse_primary_scenario(p15_judgment)
    scenarios = parse_scenarios(p15_body)
    day_arch = _day_archetype(features)

    scores: dict[str, float] = {}
    for letter, text in scenarios.items():
        trigger = text.split("→")[0].strip() if "→" in text else text
        trigger = re.sub(r"^若\s*", "", trigger)
        trigger_score = _trigger_score(trigger, features, raw)
        reaction_arch = _archetype(_reaction_text(text))
        arch_bonus = 0.0
        if reaction_arch == day_arch:
            arch_bonus = 0.35
        elif reaction_arch == "neutral" or day_arch == "neutral":
            arch_bonus = 0.1
        scores[letter] = round(min(1.0, trigger_score * 0.65 + arch_bonus), 3)

    actual: str = "none"
    if scores:
        best_letter, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score >= 0.45:
            actual = best_letter
        elif day_arch != "neutral":
            for letter, text in scenarios.items():
                if _archetype(_reaction_text(text)) == day_arch:
                    actual = letter
                    break

    correct = primary is not None and actual != "none" and primary == actual

    return {
        "scenario_primary": primary,
        "scenario_actual": actual,
        "scenario_correct": correct,
        "scenario_scores": scores,
        "day_archetype": day_arch,
        "detection_method": "auto",
    }
