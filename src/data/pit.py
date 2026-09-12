"""Historical point-in-time data validation and immutable partition storage.

The module deliberately keeps the persistence format simple: canonical UTF-8
JSON Lines in one immutable file per category and UTC partition date.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "config" / "pit_contract.yaml"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PITError(ValueError):
    """Base error for PIT contract and storage failures."""


class ContractError(PITError):
    """Raised when the contract configuration is malformed."""


class PartitionExistsError(PITError):
    """Raised when an immutable partition already exists."""


class PartitionNotFoundError(PITError):
    """Raised when a requested partition does not exist."""


@dataclass(frozen=True)
class AuditIssue:
    """A single machine-readable audit finding."""

    code: str
    message: str
    row_index: Optional[int] = None
    field: Optional[str] = None
    identifier: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in {
                "code": self.code,
                "message": self.message,
                "row_index": self.row_index,
                "field": self.field,
                "identifier": self.identifier,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class AuditReport:
    """Result of auditing rows as they would have been seen at a decision."""

    decision_at: str
    row_count: int
    issues: Tuple[AuditIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def errors(self) -> Tuple[AuditIssue, ...]:
        return self.issues

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "decision_at": self.decision_at,
            "row_count": self.row_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def load_contract(path: Optional[Union[os.PathLike, str]] = None) -> Dict[str, Any]:
    """Load and minimally validate a PIT YAML contract."""

    contract_path = Path(path) if path is not None else DEFAULT_CONTRACT_PATH
    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError("cannot read PIT contract {}: {}".format(contract_path, exc)) from exc
    except yaml.YAMLError as exc:
        raise ContractError("invalid PIT contract YAML {}: {}".format(contract_path, exc)) from exc
    if not isinstance(raw, dict):
        raise ContractError("PIT contract must be a mapping")
    required_categories = raw.get("required_categories")
    if not isinstance(required_categories, list) or not all(
        isinstance(value, str) and value for value in required_categories
    ):
        raise ContractError("required_categories must be a non-empty string list")
    row_contract = raw.get("row")
    if not isinstance(row_contract, dict):
        raise ContractError("row must be a mapping")
    required_fields = row_contract.get("required_fields")
    if not isinstance(required_fields, list) or not all(
        isinstance(value, str) and value for value in required_fields
    ):
        raise ContractError("row.required_fields must be a string list")
    return raw


def _contract(contract: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return load_contract() if contract is None else contract


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty ISO-8601 timestamp".format(field))
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("{} is not a valid ISO-8601 timestamp".format(field)) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("{} must include a UTC offset".format(field))
    return parsed


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be an ISO-8601 date".format(field))
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("{} is not a valid ISO-8601 date".format(field)) from exc


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _identifier(row: Mapping[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    explicit = row.get("identifier", row.get("id"))
    if explicit not in (None, ""):
        return ("explicit", str(explicit))
    return ("natural",) + tuple(row.get(field) for field in fields)


def _identifier_text(identifier: Tuple[Any, ...]) -> str:
    return "|".join("" if value is None else str(value) for value in identifier[1:])


def _is_tradable(row: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    explicit = row.get("tradable")
    if isinstance(explicit, bool):
        return explicit
    asset_type = str(row.get("asset_type", "")).lower()
    if asset_type:
        return asset_type in set(contract.get("tradable_asset_types", ()))
    category = str(row.get("category", row.get("kind", ""))).lower()
    return category not in set(contract.get("non_tradable_categories", ("macro",)))


def _is_short_candidate(row: Mapping[str, Any]) -> bool:
    kind = str(row.get("kind", "")).lower()
    category = str(row.get("category", "")).lower()
    side = row.get("side", row.get("direction", row.get("action", row.get("candidate_side", ""))))
    return (kind == "candidate" or category == "candidate") and str(side).upper() == "SHORT"


def audit_rows(
    rows: Iterable[Mapping[str, Any]],
    decision_at: Union[str, datetime],
    contract: Optional[Mapping[str, Any]] = None,
) -> AuditReport:
    """Audit PIT invariants for rows used by a decision.

    Membership ``member_to`` is exclusive, which permits an asset removed on a
    date to be unavailable for decisions made on that date.
    """

    cfg = _contract(contract)
    decision = (
        _parse_timestamp(decision_at, "decision_at")
        if isinstance(decision_at, str)
        else decision_at
    )
    if not isinstance(decision, datetime) or decision.tzinfo is None or decision.utcoffset() is None:
        raise ValueError("decision_at must be a timezone-aware datetime")

    materialized = list(rows)
    issues: List[AuditIssue] = []
    required = tuple(cfg.get("row", {}).get("required_fields", ()))
    identifier_fields = tuple(
        cfg.get("row", {}).get("identifier_fields", ("symbol", "kind", "observed_at"))
    )
    borrow_fields = tuple(cfg.get("short_candidate", {}).get("required_borrow_fields", ()))
    seen: Dict[Tuple[Any, ...], int] = {}

    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            issues.append(AuditIssue("invalid_row", "row must be a mapping", index))
            continue
        row_id = _identifier(row, identifier_fields)
        row_id_text = _identifier_text(row_id)
        if row_id in seen:
            issues.append(
                AuditIssue(
                    "duplicate_identifier",
                    "identifier duplicates row {}".format(seen[row_id]),
                    index,
                    identifier=row_id_text,
                )
            )
        else:
            seen[row_id] = index

        for field in required:
            if field not in row or row[field] is None or row[field] == "":
                issues.append(
                    AuditIssue(
                        "missing_required_field",
                        "required field {!r} is missing".format(field),
                        index,
                        field,
                        row_id_text,
                    )
                )

        parsed_times: Dict[str, datetime] = {}
        for field in ("observed_at", "known_at"):
            if field not in row or row[field] in (None, ""):
                continue
            try:
                parsed_times[field] = _parse_timestamp(row[field], field)
            except ValueError as exc:
                issues.append(
                    AuditIssue("invalid_timestamp", str(exc), index, field, row_id_text)
                )
        if "known_at" in parsed_times and parsed_times["known_at"] > decision:
            issues.append(
                AuditIssue(
                    "known_after_decision",
                    "row was not known by the decision timestamp",
                    index,
                    "known_at",
                    row_id_text,
                )
            )
        if (
            "observed_at" in parsed_times
            and "known_at" in parsed_times
            and parsed_times["known_at"] < parsed_times["observed_at"]
        ):
            issues.append(
                AuditIssue(
                    "known_before_observed",
                    "known_at precedes observed_at",
                    index,
                    "known_at",
                    row_id_text,
                )
            )

        if _is_tradable(row, cfg):
            membership: Dict[str, date] = {}
            for field in ("member_from", "member_to"):
                if field not in row or row[field] in (None, ""):
                    issues.append(
                        AuditIssue(
                            "missing_membership",
                            "tradable row requires {!r}".format(field),
                            index,
                            field,
                            row_id_text,
                        )
                    )
                    continue
                try:
                    membership[field] = _parse_date(row[field], field)
                except ValueError as exc:
                    issues.append(
                        AuditIssue("invalid_membership_date", str(exc), index, field, row_id_text)
                    )
            decision_date = decision.date()
            if "member_from" in membership and "member_to" in membership:
                if membership["member_from"] >= membership["member_to"]:
                    issues.append(
                        AuditIssue(
                            "invalid_membership_range",
                            "member_from must precede member_to",
                            index,
                            "member_to",
                            row_id_text,
                        )
                    )
                elif not (
                    membership["member_from"] <= decision_date < membership["member_to"]
                ):
                    issues.append(
                        AuditIssue(
                            "outside_membership",
                            "asset was not a universe member on the decision date",
                            index,
                            identifier=row_id_text,
                        )
                    )

        if _is_short_candidate(row):
            for field in borrow_fields:
                if field not in row or row[field] is None:
                    issues.append(
                        AuditIssue(
                            "missing_borrow_field",
                            "SHORT candidate requires borrow field {!r}".format(field),
                            index,
                            field,
                            row_id_text,
                        )
                    )

    return AuditReport(_iso_utc(decision), len(materialized), tuple(issues))


def filter_known_rows(
    rows: Iterable[Mapping[str, Any]],
    decision_at: Union[str, datetime],
) -> List[Mapping[str, Any]]:
    """Return rows with valid ``known_at`` timestamps no later than a decision."""

    decision = (
        _parse_timestamp(decision_at, "decision_at")
        if isinstance(decision_at, str)
        else decision_at
    )
    if not isinstance(decision, datetime) or decision.tzinfo is None or decision.utcoffset() is None:
        raise ValueError("decision_at must be a timezone-aware datetime")
    known: List[Mapping[str, Any]] = []
    for row in rows:
        try:
            known_at = _parse_timestamp(row.get("known_at"), "known_at")
        except (AttributeError, ValueError):
            continue
        if known_at <= decision:
            known.append(row)
    return known


def data_gap_report(
    rows: Iterable[Mapping[str, Any]],
    decision_at: Union[str, datetime],
    contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Report required categories absent from the decision-time row set."""

    cfg = _contract(contract)
    known = filter_known_rows(rows, decision_at)
    required = list(cfg.get("required_categories", ()))
    counts = {category: 0 for category in required}
    for row in known:
        category = str(row.get("category", row.get("kind", ""))).lower()
        if category in counts:
            counts[category] += 1
    missing = [category for category in required if counts[category] == 0]
    decision = (
        _parse_timestamp(decision_at, "decision_at")
        if isinstance(decision_at, str)
        else decision_at
    )
    return {
        "ok": not missing,
        "decision_at": _iso_utc(decision),
        "required_categories": required,
        "counts": counts,
        "missing_categories": missing,
    }


class PITPartitionStore:
    """Immutable JSONL partitions rooted at a configurable directory."""

    def __init__(
        self,
        root: Optional[Union[os.PathLike, str]] = None,
        contract: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.contract = _contract(contract)
        storage = self.contract.get("storage", {})
        environment_root = storage.get("environment_root", "PIT_DATA_ROOT")
        default_root = storage.get("default_root", "data/pit")
        configured = root if root is not None else os.environ.get(environment_root, default_root)
        self.root = Path(configured)

    @staticmethod
    def _component(value: str, label: str) -> str:
        text = str(value)
        if not _SAFE_COMPONENT.fullmatch(text) or text in (".", ".."):
            raise PITError("unsafe {}: {!r}".format(label, value))
        return text

    def partition_path(
        self, category: str, partition_date: Union[str, date]
    ) -> Path:
        category_name = self._component(category, "category").lower()
        parsed_date = _parse_date(partition_date, "partition_date")
        return self.root / category_name / ("date=" + parsed_date.isoformat()) / "data.jsonl"

    def save_partition(
        self,
        category: str,
        partition_date: Union[str, date],
        rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        """Atomically create a partition; existing data is never overwritten."""

        path = self.partition_path(category, partition_date)
        materialized = list(rows)
        encoded: List[bytes] = []
        for index, row in enumerate(materialized):
            if not isinstance(row, Mapping):
                raise PITError("row {} must be a mapping".format(index))
            try:
                line = json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise PITError("row {} is not JSON serializable: {}".format(index, exc)) from exc
            encoded.append((line + "\n").encode("utf-8"))

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix=".data.jsonl.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                os.chmod(handle.name, 0o640)
                for line in encoded:
                    handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Hard-link publication is atomic and cannot replace an existing
                # destination, including under concurrent writers.
                os.link(str(temporary_path), str(path))
            except FileExistsError as exc:
                raise PartitionExistsError("partition already exists: {}".format(path)) from exc
            directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
        return path

    def load_partition(
        self, category: str, partition_date: Union[str, date]
    ) -> List[Dict[str, Any]]:
        path = self.partition_path(category, partition_date)
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            raise PartitionNotFoundError("partition does not exist: {}".format(path)) from exc
        rows: List[Dict[str, Any]] = []
        with handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PITError("invalid JSON in {} line {}".format(path, number)) from exc
                if not isinstance(row, dict):
                    raise PITError("JSONL row in {} line {} is not an object".format(path, number))
                rows.append(row)
        return rows

    def partition_exists(self, category: str, partition_date: Union[str, date]) -> bool:
        return self.partition_path(category, partition_date).is_file()


# Concise compatibility aliases for callers that use the contract terminology.
PartitionStore = PITPartitionStore
filter_known = filter_known_rows


__all__ = [
    "AuditIssue",
    "AuditReport",
    "ContractError",
    "PITError",
    "PITPartitionStore",
    "PartitionExistsError",
    "PartitionNotFoundError",
    "PartitionStore",
    "audit_rows",
    "data_gap_report",
    "filter_known",
    "filter_known_rows",
    "load_contract",
]
