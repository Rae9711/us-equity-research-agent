from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from src.collectors.macro_releases import check_todays_releases, load_releases, persist_releases
from src.collectors.news import collect_intraday_news
from src.llm.anthropic_client import AnthropicClient
from src.research.format_body import normalize_body_md
from src.steps.base import save_step_result
from src.utils.news_signals import detect_high_signal_news, summarize_signals
from src.utils.paths import morning_json_path, step_json_path
from src.utils.trading_calendar import market_open_et, today_et

logger = logging.getLogger(__name__)

SYSTEM = """你是 Daily Trading OS 的 Step 3 Market Update Agent（10:00 ET）。
根据 Morning Research、Step 2 Opening Report、开盘以来新闻（headlines_since_open）、
今日宏观数据发布（macro_releases）、突发信号（breaking_news_signals），
判断盘中 Driver 是否切换，并给出更新后的 Total 与置信度。

输出 JSON：{"judgment":"...", "confidence":0.8, "one_liner":"...", "body_md":"..."}
judgment 格式：Driver 变了吗：YES/NO · 若变，新 Driver：{词} · 新 Total：{±N}
body_md 用 markdown 列表，含昨日→更新后 Score 表（若 Driver 未变可写 NO CHANGE）。

若 macro_releases 中任一 surprise_flag=true（实际 vs consensus 超阈值），
或 breaking_news_signals 含 severity=high 且与当前 Driver 主题相关，
必须重新评估 Driver，不得默认 NO CHANGE；若仍判断 Driver 未变，须在 body_md 中明确说明理由。
"""


def _headlines_since_open(trading_date: date) -> list[dict[str, Any]]:
    try:
        bundle = collect_intraday_news(since=market_open_et(trading_date))
        return [
            {
                "title": h.get("title"),
                "ticker": h.get("ticker"),
                "published_utc": h.get("published_utc"),
                "sentiment": h.get("sentiment"),
                "url": h.get("url"),
            }
            for h in (bundle.get("polygon") or [])[:20]
        ]
    except Exception:
        logger.exception("Intraday news collection failed")
        return []


def _macro_releases_for_step3(trading_date: date) -> list[dict[str, Any]]:
    """加载今日宏观发布；优先读快照，失败则现场抓取（不抛错）。"""
    try:
        snap = load_releases(trading_date)
        if snap and snap.get("releases"):
            return list(snap["releases"])
        payload = persist_releases(trading_date)
        return list(payload.get("releases") or [])
    except Exception:
        logger.exception("macro releases load failed; trying live check")
        try:
            return check_todays_releases(trading_date)
        except Exception:
            logger.exception("check_todays_releases failed")
            return []


def _surprise_attention(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛选需要 Step 3 特别关注的 surprise 事件。"""
    out: list[dict[str, Any]] = []
    for r in releases:
        if not r.get("surprise_flag"):
            continue
        out.append(
            {
                "event": r.get("event"),
                "label": r.get("label"),
                "actual": r.get("actual"),
                "consensus": r.get("consensus"),
                "surprise_pct": r.get("surprise_pct"),
                "direction": r.get("direction"),
                "market_impact": r.get("market_impact"),
            }
        )
    return out


def run_step3_market_update(trading_date: date | None = None) -> dict[str, Any]:
    trading_date = trading_date or today_et()
    date_str = trading_date.isoformat()
    logger.info("Step 3 market update for %s", date_str)

    morning = {}
    if morning_json_path(date_str).exists():
        morning = json.loads(morning_json_path(date_str).read_text(encoding="utf-8"))

    s2 = {}
    s2_path = step_json_path(2, date_str)
    if s2_path.exists():
        s2 = json.loads(s2_path.read_text(encoding="utf-8"))

    headlines_since_open = _headlines_since_open(trading_date)
    breaking_news_signals = detect_high_signal_news(headlines_since_open)
    breaking_summary = summarize_signals(breaking_news_signals)

    macro_releases = _macro_releases_for_step3(trading_date)
    surprise_attention = _surprise_attention(macro_releases)

    context = {
        "morning_bias": morning.get("bias"),
        "morning_total": morning.get("total_score"),
        "p10": (morning.get("parts") or {}).get("P10"),
        "s2_market": s2.get("market"),
        "s2_conclusion": s2.get("conclusion"),
        "headlines_since_open": headlines_since_open,
        "macro_releases": macro_releases,
        "surprise_attention": surprise_attention,
        "breaking_news_signals": breaking_news_signals,
        "breaking_summary": breaking_summary,
    }

    user_extra = ""
    if surprise_attention:
        user_extra += (
            "\n\n⚠️ 宏观 surprise 超阈值，必须重新评估 Driver：\n"
            + json.dumps(surprise_attention, ensure_ascii=False, indent=2)
        )
    if breaking_summary.get("has_high"):
        user_extra += (
            "\n\n⚠️ 检测到 high 突发信号，若判断 Driver 未变须给出充分理由：\n"
            + json.dumps(breaking_news_signals[:5], ensure_ascii=False, indent=2)
        )

    conclusion: dict[str, Any]
    body_md: str
    try:
        client = AnthropicClient()
        result = client.complete_json(
            SYSTEM,
            f"上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}{user_extra}",
            max_tokens=4096,
        )
        conclusion = {
            "part_id": "S3",
            "judgment": result.get("judgment", "Driver 变了吗：NO"),
            "confidence": result.get("confidence", 0.7),
            "one_liner": result.get("one_liner", "盘中无重大 Driver 切换"),
        }
        body_md = normalize_body_md(result.get("body_md") or "")
    except Exception:
        logger.exception("Step 3 LLM failed, using rules fallback")
        total = morning.get("total_score") or 0
        s2_market = s2.get("market", "Mixed")
        adj = 1 if s2_market == "Healthy" else -1 if s2_market == "Weak" else 0
        if surprise_attention or breaking_summary.get("has_high"):
            adj += 1 if adj >= 0 else -1
        new_total = total + adj
        driver_note = ""
        if surprise_attention:
            driver_note = f" · 宏观 surprise: {surprise_attention[0].get('label')}"
        elif breaking_summary.get("has_high"):
            driver_note = " · 突发高信号"
        conclusion = {
            "part_id": "S3",
            "judgment": f"Driver 变了吗：NO · 新 Total：{new_total:+d}{driver_note}",
            "confidence": 0.7,
            "one_liner": f"开盘 {s2_market}，Total 微调至 {new_total:+d}",
        }
        body_md = normalize_body_md(
            f"- 开盘状态：{s2_market}\n- Morning Total：{total:+d}\n- 更新后 Total：{new_total:+d}"
            + (f"\n- 宏观发布：{len(macro_releases)} 条" if macro_releases else "")
            + (f"\n- 突发信号：{breaking_summary.get('high', 0)} high" if breaking_summary.get("has_high") else "")
        )

    return save_step_result(
        3,
        trading_date,
        step_id="S3",
        job_id="market_update",
        conclusion=conclusion,
        body_md=body_md,
        extra={
            "context": context,
            "headlines_since_open": headlines_since_open,
            "intraday_news_count": len(headlines_since_open),
            "macro_releases": macro_releases,
            "surprise_attention": surprise_attention,
            "breaking_news_signals": breaking_news_signals,
            "breaking_summary": breaking_summary,
        },
    )
