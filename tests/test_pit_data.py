"""Deterministic tests for the historical point-in-time data contract."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from src.data.pit import (
    PITError,
    PITPartitionStore,
    PartitionExistsError,
    PartitionNotFoundError,
    audit_rows,
    data_gap_report,
    filter_known_rows,
    load_contract,
)


DECISION = "2026-07-08T14:00:00Z"


def row(**changes):
    value = {
        "identifier": "daily:ABC:2026-07-08",
        "symbol": "ABC",
        "kind": "daily",
        "category": "daily",
        "observed_at": "2026-07-08T13:59:00Z",
        "known_at": "2026-07-08T13:59:30Z",
        "member_from": "2020-01-01",
        "member_to": "2030-01-01",
    }
    value.update(changes)
    return value


def issue_codes(report):
    return [issue.code for issue in report.issues]


def test_default_contract_declares_all_required_categories():
    contract = load_contract()
    assert contract["required_categories"] == [
        "daily",
        "intraday",
        "news",
        "earnings",
        "borrow",
        "universe",
        "macro",
    ]
    assert contract["storage"]["immutable"] is True


def test_valid_historical_row_passes_audit():
    report = audit_rows([row()], DECISION)
    assert report.ok
    assert report.as_dict()["issues"] == []
    assert report.decision_at == DECISION


def test_audit_detects_future_knowledge_and_invalid_timestamps():
    rows = [
        row(identifier="future", known_at="2026-07-08T14:00:01Z"),
        row(identifier="naive", known_at="2026-07-08T13:00:00"),
        row(identifier="bad-observed", observed_at="not-a-time"),
        row(
            identifier="impossible-order",
            observed_at="2026-07-08T13:59:00Z",
            known_at="2026-07-08T13:58:00Z",
        ),
    ]
    report = audit_rows(rows, DECISION)
    assert "known_after_decision" in issue_codes(report)
    assert issue_codes(report).count("invalid_timestamp") == 2
    assert "known_before_observed" in issue_codes(report)


def test_audit_detects_survivorship_and_membership_errors():
    rows = [
        row(identifier="joined-later", member_from="2026-07-09"),
        row(identifier="removed-today", member_to="2026-07-08"),
        row(identifier="missing-membership", member_to=None),
        row(identifier="backwards", member_from="2027-01-01", member_to="2026-01-01"),
    ]
    report = audit_rows(rows, DECISION)
    assert issue_codes(report).count("outside_membership") == 2
    assert "missing_membership" in issue_codes(report)
    assert "invalid_membership_range" in issue_codes(report)


def test_macro_rows_are_not_subject_to_tradable_membership():
    macro = row(
        identifier="macro:cpi",
        symbol="CPIAUCSL",
        kind="macro",
        category="macro",
        observed_at="2026-06-01T00:00:00Z",
        known_at="2026-07-01T12:30:00Z",
        member_from=None,
        member_to=None,
    )
    assert audit_rows([macro], DECISION).ok


def test_duplicate_explicit_and_natural_identifiers_are_detected():
    explicit = [row(identifier="same"), row(identifier="same")]
    natural = [
        row(identifier=None),
        row(identifier=None, known_at="2026-07-08T13:59:45Z"),
    ]
    assert "duplicate_identifier" in issue_codes(audit_rows(explicit, DECISION))
    assert "duplicate_identifier" in issue_codes(audit_rows(natural, DECISION))


def test_short_candidate_requires_borrow_fields_but_long_does_not():
    short = row(
        identifier="short",
        kind="candidate",
        category="candidate",
        side="SHORT",
    )
    long = row(
        identifier="long",
        kind="candidate",
        category="candidate",
        side="LONG",
    )
    report = audit_rows([short, long], DECISION)
    borrow_issues = [issue for issue in report.issues if issue.code == "missing_borrow_field"]
    assert [issue.field for issue in borrow_issues] == [
        "borrow_available",
        "borrow_fee_rate",
        "ssr_restricted",
        "forced_cover",
    ]

    short["borrow_available"] = False
    short["borrow_fee_rate"] = 0.125
    short["ssr_restricted"] = False
    short["forced_cover"] = False
    assert audit_rows([short], DECISION).ok


def test_required_fields_and_duplicate_rows_have_structured_findings():
    incomplete = row(identifier="incomplete")
    del incomplete["symbol"]
    report = audit_rows([incomplete], DECISION)
    finding = next(issue for issue in report.issues if issue.code == "missing_required_field")
    assert finding.field == "symbol"
    assert finding.row_index == 0
    assert finding.identifier == "incomplete"


def test_filter_known_rows_honors_offsets_and_ignores_invalid_rows():
    rows = [
        row(identifier="before", known_at="2026-07-08T09:59:59-04:00"),
        row(identifier="at", known_at="2026-07-08T10:00:00-04:00"),
        row(identifier="after", known_at="2026-07-08T10:00:01-04:00"),
        row(identifier="invalid", known_at="unknown"),
        row(identifier="missing", known_at=None),
    ]
    assert [item["identifier"] for item in filter_known_rows(rows, DECISION)] == [
        "before",
        "at",
    ]
    with pytest.raises(ValueError, match="timezone-aware"):
        filter_known_rows(rows, datetime(2026, 7, 8, 14, 0))


def test_data_gap_report_counts_only_rows_known_at_decision():
    categories = ["daily", "intraday", "news", "earnings", "borrow", "universe", "macro"]
    rows = []
    for category in categories:
        item = row(
            identifier=category,
            kind=category,
            category=category,
            tradable=False,
        )
        rows.append(item)
    rows[-1]["known_at"] = "2026-07-08T14:00:01Z"

    report = data_gap_report(rows, DECISION)
    assert report["ok"] is False
    assert report["missing_categories"] == ["macro"]
    assert report["counts"]["daily"] == 1
    rows[-1]["known_at"] = DECISION
    assert data_gap_report(rows, DECISION)["ok"] is True


def test_partition_store_round_trip_is_canonical_and_immutable(tmp_path):
    store = PITPartitionStore(tmp_path)
    rows = [
        row(identifier="one", payload={"z": 1, "a": "é"}),
        row(identifier="two", symbol="XYZ"),
    ]
    path = store.save_partition("daily", date(2026, 7, 8), rows)

    assert path == tmp_path / "daily" / "date=2026-07-08" / "data.jsonl"
    assert store.partition_exists("daily", "2026-07-08")
    assert store.load_partition("daily", date(2026, 7, 8)) == rows
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(raw_lines[0]) == rows[0]
    assert raw_lines[0].startswith('{"category":"daily"')
    assert "é" in raw_lines[0]

    before = path.read_bytes()
    with pytest.raises(PartitionExistsError):
        store.save_partition("daily", "2026-07-08", [row(identifier="replacement")])
    assert path.read_bytes() == before


def test_partition_store_uses_environment_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PIT_DATA_ROOT", str(tmp_path))
    store = PITPartitionStore()
    assert store.save_partition("news", "2026-07-08", []).is_file()
    assert store.load_partition("news", "2026-07-08") == []


def test_partition_store_rejects_unsafe_paths_and_non_json_values(tmp_path):
    store = PITPartitionStore(tmp_path)
    with pytest.raises(PITError, match="unsafe category"):
        store.partition_path("../daily", "2026-07-08")
    with pytest.raises(ValueError, match="valid ISO-8601 date"):
        store.partition_path("daily", "08-07-2026")
    with pytest.raises(PITError, match="not JSON serializable"):
        store.save_partition("daily", "2026-07-08", [{"value": float("nan")}])
    assert not store.partition_exists("daily", "2026-07-08")


def test_loading_missing_or_corrupt_partition_has_clear_error(tmp_path):
    store = PITPartitionStore(tmp_path)
    with pytest.raises(PartitionNotFoundError):
        store.load_partition("daily", "2026-07-08")

    path = store.partition_path("daily", "2026-07-08")
    path.parent.mkdir(parents=True)
    path.write_text('{"valid":true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(PITError, match="line 2"):
        store.load_partition("daily", "2026-07-08")


def test_audit_accepts_datetime_decision_and_serializes_as_utc():
    decision = datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)
    report = audit_rows([row()], decision)
    assert report.decision_at == "2026-07-08T10:00:00Z"
    assert "known_after_decision" in issue_codes(report)
