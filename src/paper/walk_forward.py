"""Purged chronological walk-forward split construction."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Iterator, Optional


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class WalkForwardSplit:
    """Indices for one train/validation/test fold."""

    fold: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]
    test_dates: tuple[date, ...]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "train_dates": [item.isoformat() for item in self.train_dates],
            "validation_dates": [item.isoformat() for item in self.validation_dates],
            "test_dates": [item.isoformat() for item in self.test_dates],
        }


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 24
    validation_months: int = 3
    test_months: int = 3
    purge_days: int = 0
    embargo_days: int = 0
    step_months: Optional[int] = None


def purged_walk_forward_splits(
    dates: Iterable[Any],
    *,
    train_months: int = 24,
    validation_months: int = 3,
    test_months: int = 3,
    purge_days: int = 0,
    embargo_days: int = 0,
    step_months: Optional[int] = None,
) -> list[WalkForwardSplit]:
    """Create rolling calendar splits in strictly chronological order.

    ``purge_days`` separates train from validation and ``embargo_days``
    separates validation from test. Windows are half-open, and each returned
    fold has three non-empty, non-overlapping sets.
    """

    if min(train_months, validation_months, test_months) <= 0:
        raise ValueError("train, validation and test months must be positive")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge and embargo days cannot be negative")
    step = test_months if step_months is None else step_months
    if step <= 0:
        raise ValueError("step_months must be positive")

    indexed = sorted(
        ((_as_date(value), index) for index, value in enumerate(dates)),
        key=lambda item: (item[0], item[1]),
    )
    if not indexed:
        return []

    first = indexed[0][0]
    last = indexed[-1][0]
    splits: list[WalkForwardSplit] = []
    anchor = first
    fold = 1
    while True:
        train_end = _add_months(anchor, train_months)
        validation_start = train_end + timedelta(days=purge_days)
        validation_end = _add_months(validation_start, validation_months)
        test_start = validation_end + timedelta(days=embargo_days)
        test_end = _add_months(test_start, test_months)
        if test_start > last:
            break

        train_rows = [(d, i) for d, i in indexed if anchor <= d < train_end]
        validation_rows = [
            (d, i) for d, i in indexed if validation_start <= d < validation_end
        ]
        test_rows = [(d, i) for d, i in indexed if test_start <= d < test_end]
        if train_rows and validation_rows and test_rows:
            splits.append(
                WalkForwardSplit(
                    fold=fold,
                    train=tuple(i for _, i in train_rows),
                    validation=tuple(i for _, i in validation_rows),
                    test=tuple(i for _, i in test_rows),
                    train_dates=tuple(d for d, _ in train_rows),
                    validation_dates=tuple(d for d, _ in validation_rows),
                    test_dates=tuple(d for d, _ in test_rows),
                )
            )
            fold += 1
        anchor = _add_months(anchor, step)
        if anchor > last or test_end > last + timedelta(days=366 * (train_months + 6) // 12):
            break
    return splits


def walk_forward_splits(
    dates: Iterable[Any], **kwargs: Any
) -> Iterator[WalkForwardSplit]:
    """Yield purged walk-forward folds."""

    yield from purged_walk_forward_splits(dates, **kwargs)


def splits_from_config(
    dates: Iterable[Any], config: WalkForwardConfig
) -> list[WalkForwardSplit]:
    return purged_walk_forward_splits(
        dates,
        train_months=config.train_months,
        validation_months=config.validation_months,
        test_months=config.test_months,
        purge_days=config.purge_days,
        embargo_days=config.embargo_days,
        step_months=config.step_months,
    )
