"""Trading-day integrity checks for Step 0 raw payloads."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.utils.paths import raw_data_path
from src.utils.quote_session import parse_quote_session_date, quote_fresh_for_checklist
from src.utils.trading_calendar import ET, UTC, prior_close_utc_iso, prior_trading_day

logger = logging.getLogger(__name__)

_PRIOR_SESSION_TYPES = frozenset({"premarket", "prior_close", "prior_session"})
_CHANGE_PCT_TOLERANCE = 0.15


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_sessions: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "field_sessions": self.field_sessions,
            "trading_date": None,
            "prior_trading_day": None,
            "collected_date": None,
        }


def _collected_date_et(payload: dict[str, Any]) -> date | None:
    raw = payload.get("collected_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).astimezone(ET).date()
    except ValueError:
        return None


def _collected_at_et(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("collected_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).astimezone(ET)
    except ValueError:
        return None


def _parse_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(UTC)
    except ValueError:
        return None


def _iter_quotes(raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for section in ("market", "sector", "stocks"):
        quotes = ((raw.get(section) or {}).get("quotes") or {})
        for ticker, q in quotes.items():
            out.append((f"{section}.{ticker}", q))
    return out


def _quote_session_label(q: dict[str, Any]) -> str | None:
    qsd = parse_quote_session_date(q)
    if qsd is None:
        return None
    st = q.get("session_type") or "?"
    return f"{qsd.isoformat()} ({st})"


def _validate_quote(
    label: str,
    q: dict[str, Any],
    *,
    trading_date: date,
    prior_day: date,
) -> list[str]:
    issues: list[str] = []
    if "error" in q:
        issues.append(f"{label}: {q.get('error')}")
        return issues

    # FRED series observations (value/date) are not equity session quotes.
    if q.get("value") is not None and q.get("close") is None:
        return issues

    qsd = parse_quote_session_date(q)
    if qsd is None:
        issues.append(f"{label}: 缺少 quote_session_date")
        return issues

    if qsd < prior_day:
        issues.append(f"{label}: quote_session_date={qsd} 早于上一交易日 {prior_day}")
        return issues

    if qsd == trading_date:
        pass
    elif qsd == prior_day:
        if not (
            q.get("prior_session")
            or q.get("session_type") in _PRIOR_SESSION_TYPES
        ):
            issues.append(
                f"{label}: quote_session_date={prior_day} 未标记 prior_session"
            )
    else:
        issues.append(
            f"{label}: quote_session_date={qsd} 不属于 {trading_date} 或 {prior_day}"
        )

    if not quote_fresh_for_checklist(q, trading_date, prior_day):
        issues.append(f"{label}: 报价未通过 session 校验 (date={qsd})")

    close = q.get("close")
    prior_close = q.get("prior_close")
    if close is not None and prior_close is not None and prior_close != 0:
        expected = round((float(close) - float(prior_close)) / float(prior_close) * 100, 2)
        embedded = q.get("change_pct")
        if embedded is not None and abs(float(embedded) - expected) > _CHANGE_PCT_TOLERANCE:
            issues.append(
                f"{label}: change_pct 嵌入值 {embedded} ≠ 重算 {expected} (prior_close)"
            )
        pcd = q.get("prior_close_date")
        if pcd and str(pcd)[:10] != prior_day.isoformat():
            issues.append(
                f"{label}: prior_close_date={pcd} 应为 {prior_day.isoformat()}"
            )

    if q.get("session_type") == "fallback":
        if qsd not in (trading_date, prior_day):
            issues.append(f"{label}: fallback 报价日期 {qsd} 过期")

    return issues


def _validate_news(
    raw: dict[str, Any],
    *,
    trading_date: date,
    prior_day: date,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    news = raw.get("news") or {}
    expected_gte = prior_close_utc_iso(trading_date)
    actual_gte = news.get("published_gte")
    if actual_gte and actual_gte != expected_gte:
        reasons.append(
            f"news.published_gte={actual_gte} 应为上一交易日收盘 {expected_gte}"
        )

    collected = _collected_at_et(raw)
    if collected is None:
        return reasons, warnings

    prior_close_et = ET.localize(
        datetime(prior_day.year, prior_day.month, prior_day.day, 16, 0, 0)
    )
    window_start = prior_close_et.astimezone(UTC)
    window_end = collected.astimezone(UTC)

    stale_count = 0
    for article in (news.get("polygon") or [])[:40]:
        pub = _parse_utc(article.get("published_utc"))
        if pub is None:
            continue
        if pub < window_start or pub > window_end:
            stale_count += 1
            if stale_count <= 3:
                warnings.append(
                    f"news: published_utc={article.get('published_utc')} "
                    f"超出 [{prior_close_et.isoformat()}, {collected.isoformat()}]"
                )
    if stale_count > 3:
        warnings.append(f"news: 另有 {stale_count - 3} 条 Polygon 新闻超出时间窗")

    return reasons, warnings


def _validate_macro(
    raw: dict[str, Any],
    *,
    prior_day: date,
) -> list[str]:
    warnings: list[str] = []
    series = (raw.get("macro") or {}).get("series") or {}
    for key, row in series.items():
        obs_raw = row.get("date") or row.get("observation_date")
        if not obs_raw:
            continue
        try:
            obs = date.fromisoformat(str(obs_raw)[:10])
        except ValueError:
            continue
        if obs < prior_day:
            warnings.append(
                f"macro.{key}: observation_date={obs} 早于上一交易日 {prior_day}"
            )
    treasury = (raw.get("market") or {}).get("treasury_10y_fred") or {}
    obs_raw = treasury.get("date")
    if obs_raw:
        try:
            obs = date.fromisoformat(str(obs_raw)[:10])
            if obs < prior_day:
                warnings.append(f"market.10Y: observation_date={obs} 可能过期")
        except ValueError:
            pass
    return warnings


def validate_raw_for_trading_date(raw: dict[str, Any], trading_date: date) -> ValidationResult:
    """Full trading-day integrity check for a Step 0 payload."""
    prior_raw = raw.get("prior_trading_day")
    prior_day = (
        date.fromisoformat(prior_raw)
        if prior_raw
        else prior_trading_day(trading_date)
    )
    reasons: list[str] = []
    warnings: list[str] = []
    field_sessions: dict[str, str | None] = {}

    if str(raw.get("trading_date", ""))[:10] != trading_date.isoformat():
        reasons.append(
            f"trading_date={raw.get('trading_date')} ≠ 期望 {trading_date.isoformat()}"
        )

    coll_date = _collected_date_et(raw)
    if coll_date is not None and coll_date != trading_date:
        reasons.append(f"collected_at 不在交易日 {trading_date.isoformat()}")

    for label, q in _iter_quotes(raw):
        field_sessions[label] = _quote_session_label(q)
        reasons.extend(
            _validate_quote(label, q, trading_date=trading_date, prior_day=prior_day)
        )

    news_reasons, news_warnings = _validate_news(
        raw, trading_date=trading_date, prior_day=prior_day
    )
    reasons.extend(news_reasons)
    warnings.extend(news_warnings)
    warnings.extend(_validate_macro(raw, prior_day=prior_day))

    result = ValidationResult(
        ok=len(reasons) == 0,
        reasons=reasons,
        warnings=warnings,
        field_sessions=field_sessions,
    )
    d = result.to_dict()
    d["trading_date"] = trading_date.isoformat()
    d["prior_trading_day"] = prior_day.isoformat()
    d["collected_date"] = coll_date.isoformat() if coll_date else None
    return result


def freshness_dict(raw: dict[str, Any], trading_date: date | None = None) -> dict[str, Any]:
    """Serialize validation for storage in raw JSON and UI."""
    td = trading_date or date.fromisoformat(str(raw["trading_date"]))
    result = validate_raw_for_trading_date(raw, td)
    out = result.to_dict()
    out["trading_date"] = td.isoformat()
    out["prior_trading_day"] = (
        raw.get("prior_trading_day") or prior_trading_day(td).isoformat()
    )
    coll = _collected_date_et(raw)
    out["collected_date"] = coll.isoformat() if coll else None
    return out


def load_raw_payload(trading_date: date) -> dict[str, Any] | None:
    path = raw_data_path(trading_date.isoformat())
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def stale_raw_response(
    trading_date: date,
    result: ValidationResult,
    *,
    step: str = "step",
) -> dict[str, Any]:
    msg = result.reasons[0] if result.reasons else "未知校验失败"
    logger.error(
        "%s blocked for %s — raw validation failed: %s",
        step,
        trading_date.isoformat(),
        "; ".join(result.reasons[:5]),
    )
    return {
        "skipped": True,
        "reason": "stale_raw",
        "trading_date": trading_date.isoformat(),
        "freshness": {
            "ok": result.ok,
            "reasons": result.reasons,
            "warnings": result.warnings,
            "field_sessions": result.field_sessions,
            "trading_date": trading_date.isoformat(),
            "prior_trading_day": prior_trading_day(trading_date).isoformat(),
        },
        "conclusion": {
            "judgment": "SKIP",
            "confidence": None,
            "one_liner": f"数据未通过交易日校验 — 请勿用于决策 ({msg})",
        },
    }


def missing_raw_response(trading_date: date, *, step: str = "step") -> dict[str, Any]:
    logger.error("%s blocked for %s — raw data file missing", step, trading_date.isoformat())
    return {
        "skipped": True,
        "reason": "missing_raw",
        "trading_date": trading_date.isoformat(),
        "conclusion": {
            "judgment": "SKIP",
            "confidence": None,
            "one_liner": f"Step 0 原始数据缺失 ({trading_date.isoformat()})",
        },
    }


def guard_fresh_raw(
    trading_date: date,
    *,
    step: str = "step",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (raw, None) when valid, or (None, error_payload) when blocked."""
    raw = load_raw_payload(trading_date)
    if raw is None:
        return None, missing_raw_response(trading_date, step=step)
    result = validate_raw_for_trading_date(raw, trading_date)
    if not result.ok:
        return None, stale_raw_response(trading_date, result, step=step)
    return raw, None


def require_fresh_raw(trading_date: date, *, step: str = "step") -> dict[str, Any]:
    """Load raw or return a standard skip/error payload (never raises)."""
    raw, err = guard_fresh_raw(trading_date, step=step)
    if err:
        return err
    assert raw is not None
    return raw
