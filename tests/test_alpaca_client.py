"""Tests for the Alpaca paper adapter (item 5, approved 2026-11-12).

All HTTP is mocked (httpx.MockTransport). No real request is ever sent
and no real key is needed: tests pass fake keys explicitly.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.broker.alpaca_client import (
    AlpacaError,
    AlpacaPaperClient,
    LIVE_BASE_URL,
    PAPER_BASE_URL,
    build_bracket_order,
)


def _client(handler):
    transport = httpx.MockTransport(handler)
    return AlpacaPaperClient(
        key_id="PK_TEST_KEY", secret_key="test-secret", transport=transport
    )


def test_build_long_bracket_intraday():
    body = build_bracket_order(
        {
            "symbol": "AMD",
            "direction": "LONG",
            "shares": 2,
            "entry": 557.495,
            "stop": 493.4,
            "target": 643.27,
            "horizon": "Intraday",
        }
    )
    assert body["symbol"] == "AMD"
    assert body["side"] == "buy"
    assert body["type"] == "market"
    assert body["order_class"] == "bracket"
    assert body["time_in_force"] == "day"
    assert body["qty"] == 2
    assert body["take_profit"] == {"limit_price": 643.27}
    assert body["stop_loss"] == {"stop_price": 493.4}
    assert body["client_order_id"].startswith("tos-amd-")


def test_build_short_bracket_swing_uses_gtc_and_sell():
    body = build_bracket_order(
        {
            "symbol": "NVDA",
            "direction": "SHORT",
            "shares": 6,
            "entry": 176.14,
            "stop": 189.69,
            "target": 161.62,
            "horizon": "Swing",
        }
    )
    assert body["side"] == "sell"
    assert body["time_in_force"] == "gtc"
    assert body["take_profit"] == {"limit_price": 161.62}
    assert body["stop_loss"] == {"stop_price": 189.69}


def test_build_rejects_bad_geometry():
    with pytest.raises(ValueError, match="stop .* must be below target"):
        build_bracket_order({"symbol": "AMD", "direction": "LONG", "shares": 1, "stop": 600.0, "target": 500.0})
    with pytest.raises(ValueError, match="target .* must be below stop"):
        build_bracket_order({"symbol": "AMD", "direction": "SHORT", "shares": 1, "stop": 100.0, "target": 200.0})
    with pytest.raises(ValueError, match="expected stop .* < entry .* < target"):
        build_bracket_order({"symbol": "AMD", "direction": "LONG", "shares": 1, "entry": 700.0, "stop": 493.4, "target": 643.27})


def test_build_rejects_bad_shares_and_missing_prices():
    with pytest.raises(ValueError, match="positive integer"):
        build_bracket_order({"symbol": "AMD", "direction": "LONG", "shares": 0, "stop": 1.0, "target": 2.0})
    with pytest.raises(ValueError, match="missing a valid stop"):
        build_bracket_order({"symbol": "AMD", "direction": "LONG", "shares": 1, "target": 2.0})
    with pytest.raises(ValueError, match="missing symbol"):
        build_bracket_order({"direction": "LONG", "shares": 1, "stop": 1.0, "target": 2.0})


def test_live_endpoint_is_refused():
    with pytest.raises(ValueError, match="paper-only"):
        AlpacaPaperClient(key_id="k", secret_key="s", base_url=LIVE_BASE_URL)
    with pytest.raises(ValueError, match="only allows"):
        AlpacaPaperClient(key_id="k", secret_key="s", base_url="https://example.com")


def test_missing_keys_raise_without_network():
    with pytest.raises(ValueError, match="not set"):
        AlpacaPaperClient(key_id="", secret_key="")


def test_submit_bracket_order_posts_to_paper_host():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "order-123",
                "client_order_id": seen["body"]["client_order_id"],
                "status": "accepted",
                "legs": [
                    {"id": "leg-tp", "role": "take_profit"},
                    {"id": "leg-sl", "role": "stop_loss"},
                ],
            },
        )

    client = _client(handler)
    order = client.submit_bracket_order(
        {"symbol": "AMD", "direction": "LONG", "shares": 2, "stop": 493.4, "target": 643.27, "horizon": "Intraday"}
    )
    assert seen["url"] == PAPER_BASE_URL + "/v2/orders"
    assert seen["body"]["order_class"] == "bracket"
    assert seen["headers"]["apca-api-key-id"] == "PK_TEST_KEY"
    assert "test-secret" not in json.dumps(seen["body"])
    assert order["id"] == "order-123"
    assert [leg["role"] for leg in order["legs"]] == ["take_profit", "stop_loss"]


def test_get_account_positions_and_cancel():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"id": "acct", "cash": "9000.00", "equity": "10194.60", "status": "ACTIVE"})
        if request.url.path == "/v2/positions":
            return httpx.Response(200, json=[{"symbol": "AMD", "qty": "2", "side": "long"}])
        if request.url.path == "/v2/orders/order-123" and request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404, json={"message": "not found"})

    client = _client(handler)
    acct = client.get_account()
    assert acct["equity"] == "10194.60"
    assert client.list_positions()[0]["symbol"] == "AMD"
    assert client.cancel_order("order-123") is None
    assert ("DELETE", PAPER_BASE_URL + "/v2/orders/order-123") in calls


def test_auth_failure_raises_alpaca_error_with_guidance():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": 40110000, "message": "authentication failed"})

    client = _client(handler)
    with pytest.raises(AlpacaError, match="paper keys"):
        client.get_account()


def test_dry_run_builder_never_touches_network(monkeypatch):
    # build_bracket_order is pure: no client, no transport, no env keys.
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    body = build_bracket_order(
        {"symbol": "MU", "direction": "LONG", "shares": 3, "stop": 100.0, "target": 120.0}
    )
    assert body["symbol"] == "MU" and body["order_class"] == "bracket"


# ---------------------------------------------------------------------------
# Daily-report status file (added 2026-11-12): state/paper/alpaca_status.json
# is the only interface between Alpaca and the 17:24 daily report.
# ---------------------------------------------------------------------------

from src.broker.alpaca_client import (
    read_status,
    sync_status,
    write_disconnected_status,
)


def _sync_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v2/account":
        return httpx.Response(
            200,
            json={
                "id": "acct",
                "cash": "9000.00",
                "equity": "10194.60",
                "buying_power": "18000.00",
                "status": "ACTIVE",
            },
        )
    if request.url.path == "/v2/positions":
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "AMD",
                    "side": "long",
                    "qty": "2",
                    "avg_entry_price": "557.495",
                    "unrealized_pl": "9.76",
                }
            ],
        )
    if request.url.path == "/v2/orders":
        return httpx.Response(
            200,
            json=[
                {
                    "id": "order-123",
                    "symbol": "AMD",
                    "side": "buy",
                    "type": "market",
                    "qty": "2",
                    "status": "accepted",
                    "order_class": "bracket",
                }
            ],
        )
    return httpx.Response(404, json={"message": "not found"})


def test_sync_status_writes_connected_file(tmp_path):
    client = _client(_sync_handler)
    path = tmp_path / "alpaca_status.json"
    status = sync_status(client, state_path=path)
    assert status["connected"] is True
    assert status["reason"] is None
    assert status["last_sync"]
    assert status["account"] == {
        "equity": 10194.60,
        "cash": 9000.00,
        "buying_power": 18000.00,
        "status": "ACTIVE",
    }
    assert status["positions"] == [
        {
            "symbol": "AMD",
            "side": "long",
            "direction": "LONG",
            "qty": 2.0,
            "avg_entry_price": 557.495,
            "unrealized_pl": 9.76,
        }
    ]
    assert status["open_orders"] == [
        {
            "id": "order-123",
            "symbol": "AMD",
            "side": "buy",
            "direction": "LONG",
            "type": "market",
            "qty": 2.0,
            "status": "accepted",
            "order_class": "bracket",
        }
    ]
    # The file is the daily report's only interface: it must round-trip.
    assert read_status(path) == status
    assert "test-secret" not in path.read_text(encoding="utf-8")


def test_sync_status_auth_failure_writes_disconnected(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "authentication failed"})

    client = _client(handler)
    path = tmp_path / "alpaca_status.json"
    status = sync_status(client, state_path=path)
    assert status["connected"] is False
    assert status["reason"] == "auth_failed"
    assert status["account"] is None
    assert status["positions"] == []
    assert status["open_orders"] == []
    assert read_status(path) == status


def test_write_disconnected_status_no_keys(tmp_path):
    path = tmp_path / "alpaca_status.json"
    status = write_disconnected_status("no_keys", state_path=path)
    assert path.exists()
    assert status["connected"] is False
    assert status["reason"] == "no_keys"
    # No fabricated numbers: account is None, lists are empty.
    assert status["account"] is None
    assert status["positions"] == [] and status["open_orders"] == []
    assert status["last_sync"]
    assert read_status(path) == status


def test_read_status_missing_returns_none(tmp_path):
    assert read_status(tmp_path / "does-not-exist.json") is None


def test_cli_sync_without_keys_writes_file(tmp_path, monkeypatch, capsys):
    from src.broker.alpaca_client import _main

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    rc = _main(["sync"])
    assert rc == 1
    status = read_status(tmp_path / "paper" / "alpaca_status.json")
    assert status is not None
    assert status["connected"] is False
    assert status["reason"] == "no_keys"
