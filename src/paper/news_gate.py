"""News-availability gate for new entries (fix 6).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Polygon's intraday news endpoint 429s frequently on the free tier
(especially Mag7 tickers). When a symbol's news fetch failed, the
morning/step-3 research for that symbol was built without its news — so the
tick must not open a *new* position in it. Position management (stop /
target exits) is unaffected and keeps running.

Evidence comes from the news-collection error lists, freshest first:
  1. Step 3's ``intraday_news_errors`` (10:00 ET intraday collection).
  2. Step 0 raw ``news.errors`` (07:40 morning collection) — only when no
     Step 3 file exists yet for the day.

Error entries look like ``"AAPL:Client error '429 Too Many Requests' ..."``
(per-ticker) or ``"broad:Client error '429 ...'"`` (market-wide query).
Only *positive* evidence blocks an entry: no evidence (no Step 3 file, no
errors) means the gate is open — the tick must not halt trading just
because a collection file is missing.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fallback universe when config/symbols.yaml cannot be read.
FALLBACK_NEWS_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
)

WHOLE_COLLECTION_MARKER = "collect_intraday_news raised"


def news_universe() -> frozenset[str]:
    """Tickers whose per-ticker Polygon news fetch is tracked."""
    try:
        from src.collectors.config import load_symbols

        tickers = (load_symbols().get("news") or {}).get("polygon_tickers") or []
        out = {str(t).upper() for t in tickers if str(t).strip()}
        if out:
            return frozenset(out)
    except Exception:
        logger.debug("news universe config load failed", exc_info=True)
    return frozenset(FALLBACK_NEWS_TICKERS)


def parse_news_error_entries(
    errors: list[Any] | None,
    *,
    universe: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Parse a news-collection error list into a gate status.

    Returns ``blocked_symbols`` (set), ``global_block`` (bool),
    ``broad_failed`` (bool), ``whole_collection_failed`` (bool) and a
    human-readable ``reasons`` list. ``global_block`` is True only when the
    whole collection failed, or the broad market-wide query failed *and*
    every universe ticker's fetch failed too — a broad-only failure with
    working per-ticker fetches blocks nothing by itself.
    """
    universe = universe if universe is not None else news_universe()
    blocked: set[str] = set()
    broad_failed = False
    whole_failed = False
    for entry in errors or []:
        text = str(entry or "")
        if not text:
            continue
        if text.startswith("collect_intraday_news"):
            whole_failed = True
            continue
        prefix = text.split(":", 1)[0].strip()
        if prefix == "broad":
            broad_failed = True
        elif prefix.upper() in universe:
            blocked.add(prefix.upper())
    global_block = bool(whole_failed) or bool(
        broad_failed and universe and universe <= blocked
    )
    reasons: list[str] = []
    if global_block:
        reasons.append("全局新闻数据缺失（Polygon 新闻接口失败）")
    for sym in sorted(blocked):
        reasons.append(f"{sym} 新闻数据缺失（Polygon 429/失败）")
    return {
        "blocked_symbols": blocked,
        "global_block": global_block,
        "broad_failed": broad_failed,
        "whole_collection_failed": whole_failed,
        "reasons": reasons,
    }


def _load_step3_errors(trading_date: str) -> tuple[list[Any] | None, bool]:
    """Return (errors, file_existed) for Step 3's intraday news errors."""
    from src.utils.paths import step_json_path

    path = step_json_path(3, trading_date)
    if not path.exists():
        return None, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("step3 json unreadable for %s", trading_date, exc_info=True)
        return None, True
    errors = data.get("intraday_news_errors")
    if not isinstance(errors, list):
        return None, True
    return errors, True


def _load_raw_news_errors(
    trading_date: str, raw: dict[str, Any] | None
) -> list[Any] | None:
    """Step 0 morning news errors from the raw snapshot (fallback source)."""
    if raw is None:
        from src.utils.paths import raw_data_path

        path = raw_data_path(trading_date)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("raw snapshot unreadable for %s", trading_date, exc_info=True)
            return None
    news = (raw or {}).get("news") or {}
    errors = news.get("errors")
    return errors if isinstance(errors, list) else None


def news_gate_for_date(
    trading_date: str,
    *,
    step3: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """News gate status for ``trading_date`` (freshest evidence wins).

    ``step3`` / ``raw`` inject the Step 3 payload / raw snapshot (tests);
    otherwise they are loaded from disk. The returned dict carries
    ``blocked_symbols`` (sorted list), ``global_block``, ``source``
    (``"step3"`` | ``"step0"`` | ``"none"`` | ``"error"``) and ``reasons``.
    """
    try:
        errors: list[Any] | None = None
        source = "none"
        if step3 is not None:
            raw_errors = step3.get("intraday_news_errors")
            if isinstance(raw_errors, list):
                errors, source = raw_errors, "step3"
        else:
            errors, existed = _load_step3_errors(trading_date)
            if existed and errors is not None:
                source = "step3"
        if source == "none":
            raw_errors = _load_raw_news_errors(trading_date, raw)
            if raw_errors is not None:
                errors, source = raw_errors, "step0"
        parsed = parse_news_error_entries(errors)
        return {
            "blocked_symbols": sorted(parsed["blocked_symbols"]),
            "global_block": parsed["global_block"],
            "broad_failed": parsed["broad_failed"],
            "whole_collection_failed": parsed["whole_collection_failed"],
            "source": source,
            "reasons": parsed["reasons"],
        }
    except Exception:
        # Never let gate plumbing halt trading: fail open, loudly.
        logger.warning(
            "news gate load failed for %s; gate open", trading_date, exc_info=True
        )
        return {
            "blocked_symbols": [],
            "global_block": False,
            "broad_failed": False,
            "whole_collection_failed": False,
            "source": "error",
            "reasons": [],
        }


def news_block_reason(gate: dict[str, Any] | None, symbol: str) -> str | None:
    """Reason to block a new entry in ``symbol``, or None when allowed."""
    if not gate:
        return None
    if gate.get("global_block"):
        return "全局新闻数据缺失（Polygon 新闻接口失败），暂停全部新开仓"
    sym = (symbol or "").upper()
    if sym and sym in (gate.get("blocked_symbols") or []):
        return f"{sym} 新闻数据缺失（Polygon 429/失败），暂停新开仓"
    return None
