from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .registry import V2Registry
from .types import V2QueryCase


VARIANT_SPEC_VERSION = "scheme-a-visibility-v2"

BASELINE = "baseline"
SUB_OPERATIONS = "sub-operations"
NO_GOLD = "no-gold"
SUB_OPERATIONS_NO_GOLD = "sub-operations-no-gold"
SUB_OPERATIONS_NO_FIRST = "sub-operations-no-first"
SUB_OPERATIONS_FIRST_ONLY = "sub-operations-first-only"
SUB_OPERATIONS_NO_LAST = "sub-operations-no-last"
SUB_OPERATIONS_RANDOM = "sub-operations-random"

VARIANT_IDS = (
    BASELINE,
    SUB_OPERATIONS,
    NO_GOLD,
    SUB_OPERATIONS_NO_GOLD,
    SUB_OPERATIONS_NO_FIRST,
    SUB_OPERATIONS_FIRST_ONLY,
    SUB_OPERATIONS_NO_LAST,
    SUB_OPERATIONS_RANDOM,
)

_VARIANT_DEFINITIONS: dict[str, dict[str, str]] = {
    BASELINE: {
        "formula": "B",
        "removal_policy": "base_all",
    },
    SUB_OPERATIONS: {
        "formula": "S",
        "removal_policy": "sub_operations_all",
    },
    NO_GOLD: {
        "formula": r"B\{g}",
        "removal_policy": "base_gold_removed",
    },
    SUB_OPERATIONS_NO_GOLD: {
        "formula": "N",
        "removal_policy": "target_chain_removed",
    },
    SUB_OPERATIONS_NO_FIRST: {
        "formula": "N∪{g2…gk}",
        "removal_policy": "target_first_removed",
    },
    SUB_OPERATIONS_FIRST_ONLY: {
        "formula": "N∪{g1}",
        "removal_policy": "target_first_only",
    },
    SUB_OPERATIONS_NO_LAST: {
        "formula": "N∪{g1…g(k-1)}",
        "removal_policy": "target_last_removed",
    },
    SUB_OPERATIONS_RANDOM: {
        "formula": r"N∪(G\{gr})",
        "removal_policy": "target_policy_step_removed",
    },
}


def variant_spec_manifest(variant_ids: tuple[str, ...] = VARIANT_IDS) -> dict[str, Any]:
    """Return the canonical, model-independent visibility specification."""

    unknown = [variant_id for variant_id in variant_ids if variant_id not in _VARIANT_DEFINITIONS]
    if unknown:
        raise ValueError(f"unknown agent-evaluation variant: {unknown[0]}")
    return {
        "version": VARIANT_SPEC_VERSION,
        "variants": [
            {"variant_id": variant_id, **_VARIANT_DEFINITIONS[variant_id]}
            for variant_id in variant_ids
        ],
    }


def variant_spec_hash(variant_ids: tuple[str, ...] = VARIANT_IDS) -> str:
    payload = json.dumps(
        variant_spec_manifest(variant_ids),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def select_random_removed_step_index(
    *,
    step_count: int,
    operation_id: str,
    seed: int,
) -> int:
    """Select one causally required step using a reproducible operation-level draw."""

    if step_count < 2:
        raise ValueError("sub-operation decompositions must contain at least two steps")
    if step_count == 2:
        candidates = (0, 1)
    elif step_count == 3:
        candidates = (1,)
    else:
        candidates = tuple(range(1, step_count - 1))
    material = f"{VARIANT_SPEC_VERSION}:{seed}:{operation_id}:{SUB_OPERATIONS_RANDOM}"
    draw = int.from_bytes(hashlib.sha256(material.encode()).digest(), "big")
    return candidates[draw % len(candidates)]


@dataclass(frozen=True)
class VariantResolution:
    variant_id: str
    formula: str
    removal_policy: str
    target_step_ids: tuple[str, ...]
    included_target_step_ids: tuple[str, ...]
    removed_target_step_ids: tuple[str, ...]
    removed_target_step_indices: tuple[int, ...]
    visible_operation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "formula": self.formula,
            "removal_policy": self.removal_policy,
            "target_step_ids": list(self.target_step_ids),
            "included_target_step_ids": list(self.included_target_step_ids),
            "removed_target_step_ids": list(self.removed_target_step_ids),
            "removed_target_step_indices": list(self.removed_target_step_indices),
            "visible_operation_ids": list(self.visible_operation_ids),
        }


@dataclass(frozen=True)
class VariantSpec:
    """One visibility-only ablation condition.

    Prompt policy, budgets, history handling, and execution policy deliberately do
    not live here.  A variant can only select and order environment operations.
    """

    variant_id: str

    def resolve(
        self,
        registry: V2Registry,
        case: V2QueryCase,
        *,
        seed: int | None = None,
    ) -> VariantResolution:
        if self.variant_id not in VARIANT_IDS:
            raise ValueError(f"unknown agent-evaluation variant: {self.variant_id}")
        gold = registry.get_base(case.expected_call.canonical_operation_id)
        target_steps = gold.decomposition.steps
        domain_bases = tuple(
            sorted(
                (operation for operation in registry.base_operations.values() if operation.domain == case.domain),
                key=lambda operation: operation.operation_id,
            )
        )
        base_ids = {operation.operation_id for operation in domain_bases}
        all_sub_ids = {step_id for operation in domain_bases for step_id in operation.decomposition.steps}
        non_gold_sub_ids = {
            step_id
            for operation in domain_bases
            if operation.operation_id != gold.operation_id
            for step_id in operation.decomposition.steps
        }

        if self.variant_id == BASELINE:
            visible = base_ids
        elif self.variant_id == SUB_OPERATIONS:
            visible = all_sub_ids
        elif self.variant_id == NO_GOLD:
            visible = base_ids - {gold.operation_id}
        elif self.variant_id == SUB_OPERATIONS_NO_GOLD:
            visible = non_gold_sub_ids
        elif self.variant_id == SUB_OPERATIONS_NO_FIRST:
            visible = non_gold_sub_ids | set(target_steps[1:])
        elif self.variant_id == SUB_OPERATIONS_FIRST_ONLY:
            visible = non_gold_sub_ids | {target_steps[0]}
        elif self.variant_id == SUB_OPERATIONS_NO_LAST:
            visible = non_gold_sub_ids | set(target_steps[:-1])
        else:
            removed_index = select_random_removed_step_index(
                step_count=len(target_steps),
                operation_id=gold.operation_id,
                seed=registry.seed if seed is None else seed,
            )
            visible = non_gold_sub_ids | {
                step_id for index, step_id in enumerate(target_steps) if index != removed_index
            }

        effective_seed = registry.seed if seed is None else seed
        ordered_visible = tuple(
            sorted(
                visible,
                key=lambda operation_id: (
                    hashlib.sha256(f"{effective_seed}:{case.id}:{operation_id}".encode()).digest(),
                    operation_id,
                ),
            )
        )
        if self.variant_id in {BASELINE, NO_GOLD}:
            included_target_steps: tuple[str, ...] = ()
            removed_target_steps: tuple[str, ...] = ()
            removed_target_indices: tuple[int, ...] = ()
        else:
            included_target_steps = tuple(step_id for step_id in target_steps if step_id in visible)
            removed_target_steps = tuple(step_id for step_id in target_steps if step_id not in visible)
            removed_target_indices = tuple(
                index for index, step_id in enumerate(target_steps) if step_id not in visible
            )
        definition = _VARIANT_DEFINITIONS[self.variant_id]
        return VariantResolution(
            variant_id=self.variant_id,
            formula=definition["formula"],
            removal_policy=definition["removal_policy"],
            target_step_ids=target_steps,
            included_target_step_ids=included_target_steps,
            removed_target_step_ids=removed_target_steps,
            removed_target_step_indices=removed_target_indices,
            visible_operation_ids=ordered_visible,
        )

    def visible_operation_ids(
        self,
        registry: V2Registry,
        case: V2QueryCase,
        *,
        seed: int | None = None,
    ) -> tuple[str, ...]:
        return self.resolve(registry, case, seed=seed).visible_operation_ids


VARIANTS = {variant_id: VariantSpec(variant_id) for variant_id in VARIANT_IDS}


def get_variant(variant_id: str) -> VariantSpec:
    try:
        return VARIANTS[variant_id]
    except KeyError as exc:
        raise ValueError(f"unknown agent-evaluation variant: {variant_id}") from exc
