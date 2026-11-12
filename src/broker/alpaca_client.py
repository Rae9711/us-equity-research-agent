"""Alpaca paper-trading adapter (item 5, approved 2026-11-12).

Purpose: validate the Daily Trading OS execution against a real broker's
paper environment (resting bracket orders, real fills) before any live
money is considered.

PAPER ONLY. The client refuses the live endpoint
(https://api.alpaca.markets) and only talks to
https://paper-api.alpaca.markets. Keys come from the environment
(APCA_API_KEY_ID / APCA_API_SECRET_KEY) and are never logged, never
written to disk, and never sent anywhere except the paper host.

API reference: https://docs.alpaca.markets/us/docs/orders-at-alpaca
(bracket: order_class "bracket", take_profit.limit_price,
stop_loss.stop_price + optional limit_price; TP must be above SL for a
buy; TIF day/gtc; no extended hours for brackets).

ADVISORY ONLY -- 模拟交易 · 不构成投资建议.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from src.utils.paths import data_root
from src.utils.trading_calendar import ET

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
ALPACA_STATUS_FILENAME = "alpaca_status.json"


class AlpacaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _resolve_base_url(explicit: str | None = None) -> str:
    url = (explicit or os.environ.get("APCA_API_BASE_URL") or PAPER_BASE_URL).rstrip("/")
    if url == LIVE_BASE_URL or url == "https://api.alpaca.markets":
        raise ValueError(
            "Refusing live Alpaca endpoint: this adapter is paper-only "
            "(use https://paper-api.alpaca.markets)"
        )
    if url != PAPER_BASE_URL:
        raise ValueError(
            f"Unexpected Alpaca base URL {url!r}: this adapter only allows {PAPER_BASE_URL}"
        )
    return url


def build_bracket_order(signal: dict[str, Any]) -> dict[str, Any]:
    """Map a system signal to an Alpaca bracket-order request body.

    Signal: {symbol, direction LONG|SHORT, shares, stop, target,
    entry (optional, validated when present), horizon Intraday|Swing
    (Intraday -> TIF day, Swing -> TIF gtc), client_order_id (optional)}.
    No network call; pure function, safe for dry runs and tests.
    """
    symbol = str(signal.get("symbol") or "").upper()
    if not symbol:
        raise ValueError("signal is missing symbol")
    direction = str(signal.get("direction") or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"bad direction {direction!r}: expected LONG or SHORT")
    try:
        shares = int(signal.get("shares"))
    except (TypeError, ValueError):
        raise ValueError(f"bad shares {signal.get('shares')!r}: expected a positive integer")
    if shares <= 0:
        raise ValueError(f"bad shares {shares}: expected a positive integer")

    def _price(name: str) -> float:
        try:
            value = float(signal.get(name))
        except (TypeError, ValueError):
            raise ValueError(f"signal is missing a valid {name}")
        if value <= 0:
            raise ValueError(f"bad {name} {value}: expected a positive price")
        return value

    stop = _price("stop")
    target = _price("target")
    entry = signal.get("entry")
    entry_price: float | None = None
    if entry is not None:
        try:
            entry_price = float(entry)
        except (TypeError, ValueError):
            raise ValueError(f"bad entry {entry!r}")
        if entry_price <= 0:
            raise ValueError(f"bad entry {entry_price}: expected a positive price")

    # Alpaca requires TP above SL for a buy, and the mirror for a sell.
    if direction == "LONG":
        if not stop < target:
            raise ValueError(f"bad bracket geometry for LONG: stop {stop} must be below target {target}")
        if entry_price is not None and not (stop < entry_price < target):
            raise ValueError(
                f"bad bracket geometry for LONG: expected stop {stop} < entry {entry_price} < target {target}"
            )
    else:
        if not target < stop:
            raise ValueError(f"bad bracket geometry for SHORT: target {target} must be below stop {stop}")
        if entry_price is not None and not (target < entry_price < stop):
            raise ValueError(
                f"bad bracket geometry for SHORT: expected target {target} < entry {entry_price} < stop {stop}"
            )

    horizon = str(signal.get("horizon") or "Intraday").lower()
    time_in_force = "gtc" if "swing" in horizon else "day"
    if signal.get("time_in_force"):
        time_in_force = str(signal["time_in_force"]).lower()
        if time_in_force not in ("day", "gtc"):
            raise ValueError(f"bad time_in_force {time_in_force!r}: bracket orders allow day or gtc")

    return {
        "symbol": symbol,
        "qty": shares,
        "side": "buy" if direction == "LONG" else "sell",
        "type": "market",
        "time_in_force": time_in_force,
        "order_class": "bracket",
        "take_profit": {"limit_price": target},
        "stop_loss": {"stop_price": stop},
        "client_order_id": str(
            signal.get("client_order_id") or f"tos-{symbol.lower()}-{uuid.uuid4().hex[:12]}"
        ),
    }


class AlpacaPaperClient:
    """Thin REST client for the Alpaca paper Trading API."""

    def __init__(
        self,
        key_id: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.key_id = key_id or os.environ.get("APCA_API_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")
        if not self.key_id or not self.secret_key:
            raise ValueError(
                "Alpaca paper keys are not set: add APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY to ~/workspace/trading-os/.env"
            )
        self.base_url = _resolve_base_url(base_url)
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={
                "APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
            },
            timeout=30,
            transport=self._transport,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with self._client() as client:
            resp = client.request(method, path, **kwargs)
        if resp.status_code in (401, 403):
            raise AlpacaError(
                f"Alpaca rejected the request ({resp.status_code}): check that "
                "APCA_API_KEY_ID / APCA_API_SECRET_KEY are paper keys for "
                "paper-api.alpaca.markets",
                status=resp.status_code,
                body=_safe_json(resp),
            )
        if resp.status_code >= 400:
            raise AlpacaError(
                f"Alpaca request failed ({resp.status_code}): {resp.text[:500]}",
                status=resp.status_code,
                body=_safe_json(resp),
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def get_account(self) -> dict[str, Any]:
        """Paper account: cash, equity, buying power, status (read-only)."""
        return self._request("GET", "/v2/account")

    def list_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/positions")

    def get_position(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/positions/{symbol.upper()}")

    def list_orders(self, status: str = "open") -> list[dict[str, Any]]:
        return self._request("GET", "/v2/orders", params={"status": status, "limit": 100})

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{order_id}")

    def submit_bracket_order(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Submit a bracket order for a system signal. Returns the order
        JSON (id, legs). The entry is a market order; TP/SL legs activate
        only after the entry fully fills."""
        body = build_bracket_order(signal)
        return self._request("POST", "/v2/orders", json=body)


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text[:500]


# ---------------------------------------------------------------------------
# Daily-report status file (added 2026-11-12)
#
# The daily report (17:24 ET) reads ONLY this file to know the Alpaca paper
# state: it never calls the adapter itself. The file must always exist --
# when there are no keys or the paper API is unreachable it carries
# connected=false with a reason, and never fabricated numbers.
# ---------------------------------------------------------------------------


def alpaca_status_path(state_path: str | Path | None = None) -> Path:
    if state_path is not None:
        return Path(state_path)
    d = data_root() / "paper"
    d.mkdir(parents=True, exist_ok=True)
    return d / ALPACA_STATUS_FILENAME


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_account(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "equity": _to_float(raw.get("equity")),
        "cash": _to_float(raw.get("cash")),
        "buying_power": _to_float(raw.get("buying_power")),
        "status": raw.get("status"),
    }


def _normalize_position(raw: dict[str, Any]) -> dict[str, Any]:
    side = str(raw.get("side") or "").lower()
    return {
        "symbol": str(raw.get("symbol") or "").upper(),
        "side": side,
        "direction": "LONG" if side == "long" else "SHORT",
        "qty": _to_float(raw.get("qty")),
        "avg_entry_price": _to_float(raw.get("avg_entry_price")),
        "unrealized_pl": _to_float(raw.get("unrealized_pl")),
    }


def _normalize_order(raw: dict[str, Any]) -> dict[str, Any]:
    side = str(raw.get("side") or "").lower()
    return {
        "id": raw.get("id"),
        "symbol": str(raw.get("symbol") or "").upper(),
        "side": side,
        "direction": "LONG" if side == "buy" else "SHORT",
        "type": raw.get("type"),
        "qty": _to_float(raw.get("qty")),
        "status": raw.get("status"),
        "order_class": raw.get("order_class"),
    }


def _write_status(status: dict[str, Any], state_path: str | Path | None) -> Path:
    path = alpaca_status_path(state_path)
    path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_disconnected_status(
    reason: str, *, state_path: str | Path | None = None
) -> dict[str, Any]:
    """Write a connected=false status file (no keys / not yet connected).

    Numbers are left empty (account None, lists empty) rather than
    fabricated; the daily report reads ``reason`` to say why.
    """
    status: dict[str, Any] = {
        "last_sync": datetime.now(ET).isoformat(),
        "connected": False,
        "reason": reason,
        "account": None,
        "positions": [],
        "open_orders": [],
    }
    _write_status(status, state_path)
    return status


def read_status(state_path: str | Path | None = None) -> dict[str, Any] | None:
    """Read the status file for the daily report. None when missing/corrupt."""
    path = alpaca_status_path(state_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def sync_status(
    client: "AlpacaPaperClient", *, state_path: str | Path | None = None
) -> dict[str, Any]:
    """Fetch paper account/positions/open orders and rewrite the status file.

    Returns the status dict. On an Alpaca error it writes connected=false
    with a reason instead of raising, so the daily report always has a
    fresh file to read.
    """
    now = datetime.now(ET).isoformat()
    try:
        account = client.get_account()
        positions = client.list_positions()
        orders = client.list_orders(status="open")
    except AlpacaError as exc:
        if exc.status in (401, 403):
            reason = "auth_failed"
        elif exc.status is not None:
            reason = f"request_failed: HTTP {exc.status}"
        else:
            reason = f"request_failed: {exc}"
        status: dict[str, Any] = {
            "last_sync": now,
            "connected": False,
            "reason": reason,
            "account": None,
            "positions": [],
            "open_orders": [],
        }
        _write_status(status, state_path)
        return status
    status = {
        "last_sync": now,
        "connected": True,
        "reason": None,
        "account": _normalize_account(account or {}),
        "positions": [_normalize_position(p) for p in (positions or [])],
        "open_orders": [_normalize_order(o) for o in (orders or [])],
    }
    _write_status(status, state_path)
    return status


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="alpaca_client")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("account", help="Print paper account (read-only)")
    sub.add_parser("positions", help="List paper positions (read-only)")
    sub.add_parser("orders", help="List open paper orders (read-only)")
    sub.add_parser(
        "sync",
        help="Sync paper account state to the status file for the daily report",
    )
    dry = sub.add_parser("dry-run", help="Build a bracket order body without sending it")
    dry.add_argument("--signal", required=True, help="Signal JSON, e.g. {\"symbol\":\"AMD\",\"direction\":\"LONG\",\"shares\":2,\"stop\":493.4,\"target\":643.27}")
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        print(json.dumps(build_bracket_order(json.loads(args.signal)), indent=2))
        return 0
    try:
        client = AlpacaPaperClient()
    except ValueError:
        # No keys: the status file must still exist so the daily report
        # can say "Alpaca 未接入" instead of failing on a missing file.
        status = write_disconnected_status("no_keys")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 1
    if args.command == "sync":
        status = sync_status(client)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status.get("connected") else 1
    if args.command == "account":
        print(json.dumps(client.get_account(), indent=2))
    elif args.command == "positions":
        print(json.dumps(client.list_positions(), indent=2))
    elif args.command == "orders":
        print(json.dumps(client.list_orders(), indent=2))
    # Every adapter run refreshes the status file for the daily report.
    # A sync failure must not fail the read command that just succeeded.
    try:
        sync_status(client)
    except Exception as exc:  # noqa: BLE001 - best-effort status refresh
        print(f"warning: status sync failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
