from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .equivalence import validate_decomposition_equivalence
from .registry import V2Registry
from .types import TaskGoalSpec, V2ExpectedCall, V2QueryCase, public_dataclass_dict
from .uniqueness import validate_single_step_unique

TEMPLATES = (
    "For enterprise subject {entity_id} in context {scope_id}, using reference {qualifier}, determine the {effect}.",
    "Operations review for subject {entity_id}, context {scope_id}, and reference {qualifier}: report the {effect}.",
    "Given subject {entity_id} under business context {scope_id} at reference {qualifier}, what is the {effect}?",
)


@dataclass(frozen=True)
class QueryGenerationResult:
    cases: tuple[V2QueryCase, ...]
    rejections: tuple[dict[str, Any], ...]


def canonical_to_visible(operation, arguments: dict[str, Any]) -> dict[str, Any]:
    return {visible: arguments[canonical] for visible, canonical in operation.visible_to_canonical_arguments.items()}


def visible_to_canonical(operation, arguments: dict[str, Any]) -> dict[str, Any]:
    expected = set(operation.visible_to_canonical_arguments)
    if set(arguments) != expected:
        missing, extra = expected - set(arguments), set(arguments) - expected
        raise ValueError(f"visible arguments mismatch; missing={sorted(missing)}, unexpected={sorted(extra)}")
    return {canonical: arguments[visible] for visible, canonical in operation.visible_to_canonical_arguments.items()}


def generate_v2_queries(registry: V2Registry, *, queries_per_base: int = 5, seed: int | None = None) -> QueryGenerationResult:
    cases, rejections = [], []
    for operation in registry.base_operations.values():
        fixtures = registry.fixtures[operation.operation_id]
        for variant in range(queries_per_base):
            fixture = fixtures[variant % len(fixtures)]
            arguments = fixture["arguments"]
            template_id = variant % len(TEMPLATES)
            prompt = TEMPLATES[template_id].format(**arguments, effect=registry.operation_effects[operation.operation_id])
            case_id = f"{operation.domain}-{operation.operation_id.removeprefix('resolve_')}-{variant:02d}"
            reasons = []
            if operation.operation_id.lower() in prompt.lower():
                reasons.append("operation_name_leakage")
            if any(str(value) not in prompt for value in arguments.values()):
                reasons.append("query_not_grounded")
            equivalence = validate_decomposition_equivalence(base=operation, sub_operations=registry.sub_operations, state=registry.new_state(), arguments=arguments)
            if not equivalence.valid:
                reasons.append("decomposition_not_equivalent")
            goal = TaskGoalSpec(operation.effect.postconditions, operation.effect.output_schema, fixture["output"])
            same_domain = [op for op in registry.base_operations.values() if op.domain == operation.domain]
            domain_subs = [sub for sub in registry.sub_operations.values() if sub.domain == operation.domain]
            unique = validate_single_step_unique(goal=goal, gold=operation, candidate_bases=same_domain, candidate_sub_operations=domain_subs, state=registry.new_state(), arguments=arguments)
            if not unique.valid:
                reasons.append("not_single_step_unique")
            if reasons:
                rejections.append({"case_id": case_id, "operation_id": operation.operation_id, "reasons": reasons, "witnesses": list(unique.witnesses)})
                continue
            visible_arguments = canonical_to_visible(operation, arguments)
            cases.append(V2QueryCase(
                case_id, operation.domain, prompt, operation.effect.postconditions,
                V2ExpectedCall(operation.operation_id, arguments, visible_arguments),
                {"scheme": registry.scheme, "base_operation_id": operation.operation_id,
                 "target_postconditions": list(operation.effect.postconditions),
                 "resource_entities": arguments, "signature_family": operation.signature_family.value,
                 "query_template_id": template_id, "decomposition_steps": list(operation.decomposition.steps),
                 "validation": {"query_grounded": True, "base_execution_valid": True,
                                "decomposition_equivalent": True, "single_step_unique": True,
                                "no_operation_name_leakage": True}},
            ))
    return QueryGenerationResult(tuple(cases), tuple(rejections))


def query_case_to_dict(case: V2QueryCase) -> dict[str, Any]:
    return public_dataclass_dict(case)


def query_case_from_dict(data: dict[str, Any]) -> V2QueryCase:
    expected = data["expected_call"]
    return V2QueryCase(data["id"], data["domain"], data["prompt"], tuple(data["target_postconditions"]), V2ExpectedCall(expected["canonical_operation_id"], expected["canonical_arguments"], expected["visible_arguments"]), data.get("hidden_metadata", {}))

