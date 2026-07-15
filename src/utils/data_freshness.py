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

# FRED macro series tiers (keys in macro.series and/or series_id).
_MONTHLY_MACRO = frozenset(
    {"FEDFUNDS", "CPIAUCSL", "PCEPI", "PAYEMS", "UNRATE", "AHETPI"}
)
_WEEKLY_MACRO = frozenset({"ICSA"})
_DAILY_RATE_MACRO = frozenset({"DGS10", "DGS2"})
_BOND_PROXY_TICKERS = frozenset({"^TNX", "TNX"})
# Weekly claims: week-ending Saturday, typically released the following Thursday.
# Mid-week before the next release the prior Saturday can be ~11–12 calendar days old;
# holiday weeks (e.g. Jul 4) need a little more slack before flagging.
_ICSA_STALE_CALENDAR_DAYS = 14


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attention: list[str] = field(default_factory=list)
    macro_reference: list[dict[str, str]] = field(default_factory=list)
    field_sessions: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "attention": self.attention,
            "macro_reference": self.macro_reference,
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


def _is_bond_proxy_label(label: str) -> bool:
    return label.rsplit(".", 1)[-1] in _BOND_PROXY_TICKERS


def _bond_proxy_attention(
    label: str,
    q: dict[str, Any],
    *,
    prior_day: date,
) -> str | None:
    """Bond yield proxies (^TNX) may lag like FRED — attention only, never blocks ok."""
    err = q.get("error")
    if err:
        err_s = str(err)
        if "stale bar date" in err_s:
            # e.g. "stale bar date 2026-07-02 (expected ...)"
            parts = err_s.split("stale bar date ", 1)
            bar = parts[1].split(" ", 1)[0] if len(parts) > 1 else "?"
            return (
                f"{label}: proxy 最新 {bar}，上一交易日 {prior_day.isoformat()}（债券数据正常滞后）"
            )
        return f"{label}: {err_s}"

    qsd = parse_quote_session_date(q)
    if qsd is None:
        if q.get("close") is None:
            return f"{label}: 债券 proxy 无可用报价"
        return None

    if qsd < prior_day:
        return (
            f"{label}: proxy 最新 {qsd.isoformat()}，上一交易日 {prior_day.isoformat()}（债券数据正常滞后）"
        )
    return None


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


def macro_series_tier(key: str, row: dict[str, Any] | None = None) -> str:
    """Return ``monthly`` | ``weekly`` | ``daily_rate`` | ``other`` for a FRED series."""
    sid = str((row or {}).get("series_id") or key).upper()
    key_u = str(key).upper()
    if sid in _MONTHLY_MACRO or key_u in _MONTHLY_MACRO:
        return "monthly"
    if sid in _WEEKLY_MACRO or key_u in _WEEKLY_MACRO:
        return "weekly"
    if sid in _DAILY_RATE_MACRO or key_u in _DAILY_RATE_MACRO:
        return "daily_rate"
    return "other"


def _parse_obs_date(obs_raw: Any) -> date | None:
    if not obs_raw:
        return None
    try:
        return date.fromisoformat(str(obs_raw)[:10])
    except ValueError:
        return None


def _tnx_quote_fresh(
    raw: dict[str, Any],
    *,
    trading_date: date,
    prior_day: date,
) -> bool:
    quotes = ((raw.get("market") or {}).get("quotes") or {})
    for ticker in ("^TNX", "TNX"):
        q = quotes.get(ticker) or {}
        if quote_fresh_for_checklist(q, trading_date, prior_day):
            return True
    return False


def _classify_macro_observation(
    label: str,
    obs: date,
    *,
    tier: str,
    trading_date: date,
    prior_day: date,
    tnx_fresh: bool,
) -> tuple[str | None, dict[str, str] | None]:
    """Return (attention_message, macro_reference_entry) — at most one non-None."""
    if tier == "monthly":
        return None, {
            "key": label,
            "date": obs.isoformat(),
            "label": f"{label}: 最新可用 {obs.isoformat()}",
        }

    if tier == "weekly":
        age_days = (trading_date - obs).days
        if age_days > _ICSA_STALE_CALENDAR_DAYS:
            return (
                f"{label}: observation_date={obs} 已超过 {age_days} 日历日未更新",
                None,
            )
        return None, {
            "key": label,
            "date": obs.isoformat(),
            "label": f"{label}: 最新可用 {obs.isoformat()}（周度）",
        }

    if tier == "daily_rate":
        if obs < prior_day:
            suffix = "；盘中 ^TNX 已补充" if tnx_fresh else "（FRED 正常滞后）"
            return (
                f"{label}: FRED 最新 {obs.isoformat()}，上一交易日 {prior_day.isoformat()}{suffix}",
                None,
            )
        return None, None

    if obs < prior_day:
        return (
            f"{label}: observation_date={obs} 早于上一交易日 {prior_day}",
            None,
        )
    return None, None


def _validate_macro(
    raw: dict[str, Any],
    *,
    trading_date: date,
    prior_day: date,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Classify macro lag into attention (需关注) vs macro_reference (正常滞后)."""
    attention: list[str] = []
    macro_reference: list[dict[str, str]] = []
    legacy_warnings: list[str] = []
    tnx_fresh = _tnx_quote_fresh(raw, trading_date=trading_date, prior_day=prior_day)

    series = (raw.get("macro") or {}).get("series") or {}
    for key, row in series.items():
        obs = _parse_obs_date(row.get("date") or row.get("observation_date"))
        if obs is None:
            continue
        tier = macro_series_tier(key, row)
        msg, ref = _classify_macro_observation(
            f"macro.{key}",
            obs,
            tier=tier,
            trading_date=trading_date,
            prior_day=prior_day,
            tnx_fresh=tnx_fresh,
        )
        if ref:
            macro_reference.append(ref)
        elif msg:
            attention.append(msg)
            if tier not in ("daily_rate", "weekly"):
                legacy_warnings.append(msg)

    treasury = (raw.get("market") or {}).get("treasury_10y_fred") or {}
    obs = _parse_obs_date(treasury.get("date"))
    if obs is not None:
        msg, ref = _classify_macro_observation(
            "market.10Y",
            obs,
            tier="daily_rate",
            trading_date=trading_date,
            prior_day=prior_day,
            tnx_fresh=tnx_fresh,
        )
        if ref:
            macro_reference.append(ref)
        elif msg:
            attention.append(msg)

    return legacy_warnings, attention, macro_reference


def dgs10_fred_stale(raw: dict[str, Any], prior_day: date) -> bool:
    """True when embedded DGS10 / treasury_10y_fred predates the prior session."""
    dgs10_row = ((raw.get("macro") or {}).get("series") or {}).get("DGS10") or {}
    macro_obs = _parse_obs_date(dgs10_row.get("date"))
    mkt_obs = _parse_obs_date(
        ((raw.get("market") or {}).get("treasury_10y_fred") or {}).get("date")
    )
    for obs in (macro_obs, mkt_obs):
        if obs is not None and obs < prior_day:
            return True
    return False


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
        if _is_bond_proxy_label(label):
            continue
        reasons.extend(
            _validate_quote(label, q, trading_date=trading_date, prior_day=prior_day)
        )

    news_reasons, news_warnings = _validate_news(
        raw, trading_date=trading_date, prior_day=prior_day
    )
    reasons.extend(news_reasons)
    warnings.extend(news_warnings)
    macro_legacy, attention, macro_reference = _validate_macro(
        raw, trading_date=trading_date, prior_day=prior_day
    )
    for label, q in _iter_quotes(raw):
        if not _is_bond_proxy_label(label):
            continue
        msg = _bond_proxy_attention(label, q, prior_day=prior_day)
        if msg:
            attention.append(msg)
    warnings.extend(macro_legacy)

    result = ValidationResult(
        ok=len(reasons) == 0,
        reasons=reasons,
        warnings=warnings,
        attention=attention,
        macro_reference=macro_reference,
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
            "attention": result.attention,
            "macro_reference": result.macro_reference,
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
