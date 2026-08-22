"""Instrument classification for paper routing (equity vs option).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

from typing import Any


def instrument_label(signal_or_slot: dict[str, Any] | None) -> str:
    raw = signal_or_slot or {}
    inst = raw.get("instrument")
    if not inst and isinstance(raw.get("raw_slot"), dict):
        inst = raw["raw_slot"].get("instrument")
    return str(inst or "").strip()


def is_option_instrument(instrument: str | None) -> bool:
    text = (instrument or "").upper()
    return "CALL" in text or "PUT" in text


def option_right(instrument: str | None) -> str | None:
    text = (instrument or "").upper()
    if "PUT" in text:
        return "put"
    if "CALL" in text:
        return "call"
    return None


def is_zero_dte(instrument: str | None, horizon: str | None = None) -> bool:
    text = f"{instrument or ''} {horizon or ''}".upper()
    return "0DTE" in text.replace(" ", "")


def asset_class_for(instrument: str | None) -> str:
    return "option" if is_option_instrument(instrument) else "equity"
