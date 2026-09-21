"""Whole-action sequence probabilities and grouped stopping margins."""

from __future__ import annotations
import math
from typing import Iterable
from envtoolbench.v2.types import JsonObject


def _logsumexp(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return float("nan")
    maximum = max(items)
    return maximum + math.log(sum(math.exp(value - maximum) for value in items))


def _margins(
    *,
    family: str,
    scores: list[JsonObject],
    legal_names: set[str],
) -> JsonObject:
    by_name = {
        str(item["candidate_name"]): float(item["sequence_logprob"]) for item in scores
    }
    unavailable = by_name["harness_unavailable"]
    finish = by_name["harness_finish"]
    restart = by_name["harness_restart"]
    environment = [name for name in by_name if not name.startswith("harness_")]
    if family == "semantic":
        return {
            "unavailable_margin": unavailable
            - _logsumexp(
                value
                for name, value in by_name.items()
                if name != "harness_unavailable"
            ),
            "legal_margin": None,
            "fixed_universe_margin": None,
        }
    legal_environment = [by_name[name] for name in environment if name in legal_names]
    environment_lme = _logsumexp(legal_environment) - math.log(len(legal_environment))
    return {
        "unavailable_margin": None,
        "legal_margin": unavailable - _logsumexp((finish, restart, environment_lme)),
        "fixed_universe_margin": unavailable
        - _logsumexp(
            value for name, value in by_name.items() if name != "harness_unavailable"
        ),
    }
