from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable


def require_one(rows: Iterable[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(k) == v for k, v in keys.items())]
    if len(matches) != 1:
        raise KeyError(f"expected one row for {keys}, found {len(matches)}")
    return matches[0]


def filter_rows(rows: Iterable[dict[str, Any]], **keys: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(k) == v for k, v in keys.items())]


def sum_field(rows: Iterable[dict[str, Any]], field: str) -> float:
    return sum(row[field] for row in rows)


def rank_rows(rows: Iterable[dict[str, Any]], *fields: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(row[field] for field in fields))


def compare_sets(required: Iterable[Any], actual: Iterable[Any]) -> dict[str, Any]:
    missing = sorted(set(required) - set(actual))
    return {"complete": not missing, "missing": missing}


def calculate_business_date(start: str, days: int) -> str:
    current = date.fromisoformat(start)
    remaining = days
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current.isoformat()


def normalize_money(value: float, digits: int = 2) -> float:
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def stable_sample(values: Iterable[Any], size: int, seed: str) -> list[Any]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{json.dumps(value, sort_keys=True)}".encode()).hexdigest())[:size]

