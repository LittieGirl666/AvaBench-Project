from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .stateful_operations import StatefulRegistry
from .types import JsonObject, V2QueryCase


STATEFUL_VARIANT_SPEC_VERSION = "paired-state-actions-v1"
STATE_GOLD_ONLY = "state-gold-only"
STATE_GOLD_DECOY = "state-gold-decoy"
STATE_DECOY_SUCCESS = "state-decoy-success"
STATE_DECOY_FAILURE = "state-decoy-failure"

STATEFUL_VARIANT_IDS = (
    STATE_GOLD_ONLY,
    STATE_GOLD_DECOY,
    STATE_DECOY_SUCCESS,
    STATE_DECOY_FAILURE,
)

_DEFINITIONS: dict[str, dict[str, str]] = {
    STATE_GOLD_ONLY: {
        "formula": "N∪{g}",
        "decoy_runtime": "absent",
        "purpose": "executable upper bound",
    },
    STATE_GOLD_DECOY: {
        "formula": "N∪{g,d}",
        "decoy_runtime": "success_mutation",
        "purpose": "selection under a valid counterfactual twin",
    },
    STATE_DECOY_SUCCESS: {
        "formula": "N∪{d}",
        "decoy_runtime": "success_mutation",
        "purpose": "successful wrong-state affordance",
    },
    STATE_DECOY_FAILURE: {
        "formula": "N∪{d}",
        "decoy_runtime": "explicit_failure",
        "purpose": "matched recoverability control",
    },
}


def stateful_variant_manifest(
    variant_ids: tuple[str, ...] = STATEFUL_VARIANT_IDS,
) -> JsonObject:
    unknown = [variant_id for variant_id in variant_ids if variant_id not in _DEFINITIONS]
    if unknown:
        raise ValueError(f"unknown stateful variant: {unknown[0]}")
    return {
        "version": STATEFUL_VARIANT_SPEC_VERSION,
        "variants": [
            {"variant_id": variant_id, **_DEFINITIONS[variant_id]}
            for variant_id in variant_ids
        ],
    }


def stateful_variant_hash(
    variant_ids: tuple[str, ...] = STATEFUL_VARIANT_IDS,
) -> str:
    payload = json.dumps(
        stateful_variant_manifest(variant_ids),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class StatefulVariantResolution:
    variant_id: str
    formula: str
    decoy_runtime: str
    target_operation_id: str
    decoy_operation_id: str
    nuisance_operation_ids: tuple[str, ...]
    visible_operation_ids: tuple[str, ...]
    forced_failure_operation_ids: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        return {
            "variant_id": self.variant_id,
            "formula": self.formula,
            "decoy_runtime": self.decoy_runtime,
            "target_operation_id": self.target_operation_id,
            "decoy_operation_id": self.decoy_operation_id,
            "nuisance_operation_ids": list(self.nuisance_operation_ids),
            "visible_operation_ids": list(self.visible_operation_ids),
            "forced_failure_operation_ids": list(self.forced_failure_operation_ids),
        }

    def execution_failures(self) -> dict[str, str]:
        return {
            operation_id: (
                "The requested state transition failed its deterministic "
                "execution precondition; no business state was changed."
            )
            for operation_id in self.forced_failure_operation_ids
        }


@dataclass(frozen=True)
class StatefulVariantSpec:
    variant_id: str

    def resolve(
        self,
        registry: StatefulRegistry,
        case: V2QueryCase,
        *,
        seed: int | None = None,
    ) -> StatefulVariantResolution:
        if self.variant_id not in STATEFUL_VARIANT_IDS:
            raise ValueError(f"unknown stateful variant: {self.variant_id}")
        target = case.expected_call.canonical_operation_id
        transition = registry.state_transitions.get(target)
        if transition is None:
            raise ValueError(f"case target is not a state action: {target}")
        pair = registry.state_action_pairs[transition.pair_id]
        decoy = transition.paired_operation_id
        nuisance = tuple(pair.observer_operation_ids)

        visible = set(nuisance)
        if self.variant_id in {STATE_GOLD_ONLY, STATE_GOLD_DECOY}:
            visible.add(target)
        if self.variant_id in {
            STATE_GOLD_DECOY,
            STATE_DECOY_SUCCESS,
            STATE_DECOY_FAILURE,
        }:
            visible.add(decoy)
        forced = (decoy,) if self.variant_id == STATE_DECOY_FAILURE else ()

        effective_seed = registry.seed if seed is None else seed
        ordered_visible = tuple(sorted(
            visible,
            key=lambda operation_id: (
                hashlib.sha256(
                    f"{effective_seed}:{case.id}:{operation_id}".encode()
                ).digest(),
                operation_id,
            ),
        ))
        definition = _DEFINITIONS[self.variant_id]
        return StatefulVariantResolution(
            self.variant_id,
            definition["formula"],
            definition["decoy_runtime"],
            target,
            decoy,
            nuisance,
            ordered_visible,
            forced,
        )

    def visible_operation_ids(
        self,
        registry: StatefulRegistry,
        case: V2QueryCase,
        *,
        seed: int | None = None,
    ) -> tuple[str, ...]:
        return self.resolve(registry, case, seed=seed).visible_operation_ids


def get_stateful_variant(variant_id: str) -> StatefulVariantSpec:
    if variant_id not in STATEFUL_VARIANT_IDS:
        raise ValueError(f"unknown stateful variant: {variant_id}")
    return StatefulVariantSpec(variant_id)

