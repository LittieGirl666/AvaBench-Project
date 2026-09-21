from __future__ import annotations

from typing import Any

from .execution import (
    EnvironmentState, canonical_json, execute_base, execute_decomposition,
    execute_sub_operation, resolve_step_arguments,
)
from .types import BaseOperationSpec, JsonObject, SubOperationSpec, ValidationResult


def normalize_output(value: Any) -> Any:
    return value


def validate_decomposition_equivalence(*, base: BaseOperationSpec, sub_operations: dict[str, SubOperationSpec], state: EnvironmentState, arguments: JsonObject) -> ValidationResult:
    required_resources = set(base.effect.read_set)
    for step_id in base.decomposition.steps:
        required_resources.update(sub_operations[step_id].effect.read_set)
    state = EnvironmentState({resource_id: state.resources[resource_id] for resource_id in required_resources})
    try:
        base_result = execute_base(base, state.clone(), arguments)
        chain_result = execute_decomposition(base.decomposition, sub_operations, state.clone(), arguments)
    except Exception as exc:
        return ValidationResult(False, (str(exc),))
    errors = []
    if normalize_output(base_result.output) != normalize_output(chain_result.output):
        errors.append("base and decomposition outputs differ")
    if base_result.state_digest_after != chain_result.state_digest_after:
        errors.append("base and decomposition state digests differ")
    return ValidationResult(not errors, tuple(errors), details={"base_output": base_result.output, "chain_output": chain_result.output, "trace": list(chain_result.trace)})


def validate_decomposition_necessity(
    *, base: BaseOperationSpec, sub_operations: dict[str, SubOperationSpec],
    state: EnvironmentState, arguments: JsonObject,
) -> ValidationResult:
    """Prove that every declared stage has a causal, non-identity role."""
    required_resources = set(base.effect.read_set)
    for step_id in base.decomposition.steps:
        required_resources.update(sub_operations[step_id].effect.read_set)
    state = EnvironmentState({resource_id: state.resources[resource_id] for resource_id in required_resources})
    errors: list[str] = []
    expected = execute_base(base, state.clone(), arguments).output
    working = state.clone()
    outputs: JsonObject = {}
    runtime_digests: list[str] = []
    stage_outputs: list[Any] = []
    for index, step_id in enumerate(base.decomposition.steps):
        step_arguments = resolve_step_arguments(
            base.decomposition, step_id=step_id, inputs=arguments, step_outputs=outputs,
        )
        before_runtime = working.runtime_digest()
        result = execute_sub_operation(sub_operations[step_id], working, step_arguments)
        after_runtime = working.runtime_digest()
        outputs[step_id] = result.output
        stage_outputs.append(result.output)
        runtime_digests.append(after_runtime)
        if before_runtime == after_runtime:
            errors.append(f"{step_id}: stage has no transaction-workspace impact")
        is_terminal = index == len(base.decomposition.steps) - 1
        declares_goal = bool(set(base.effect.postconditions) & set(sub_operations[step_id].effect.postconditions))
        if is_terminal:
            if result.output != expected:
                errors.append(f"{step_id}: terminal output differs from base output")
            if not declares_goal:
                errors.append(f"{step_id}: terminal stage does not declare the target postcondition")
        else:
            if result.output == expected:
                errors.append(f"{step_id}: intermediate stage already returns the final output")
            if declares_goal:
                errors.append(f"{step_id}: intermediate stage declares the target postcondition")

    if len({canonical_json(output) for output in stage_outputs}) != len(stage_outputs):
        errors.append(f"{base.operation_id}: decomposition contains identity-equivalent stage outputs")

    # Removing each stage must leave either no final output or an invalid direct
    # predecessor.  A fresh state ensures a copied/guessed payload has no
    # provenance in the execution workspace.
    steps = base.decomposition.steps
    for removed_index, removed_step in enumerate(steps):
        candidate = state.clone()
        prefix_outputs: JsonObject = {}
        for index, step_id in enumerate(steps[:removed_index]):
            step_arguments = resolve_step_arguments(
                base.decomposition, step_id=step_id, inputs=arguments, step_outputs=prefix_outputs,
            )
            prefix_outputs[step_id] = execute_sub_operation(
                sub_operations[step_id], candidate, step_arguments,
            ).output
        if removed_index == len(steps) - 1:
            if any(output == expected for output in prefix_outputs.values()):
                errors.append(f"{removed_step}: removing terminal stage still exposes the final output")
            continue
        next_step = steps[removed_index + 1]
        fabricated = stage_outputs[removed_index]
        if removed_index > 0:
            fabricated = prefix_outputs[steps[removed_index - 1]]
        try:
            execute_sub_operation(sub_operations[next_step], candidate, {"payload": fabricated})
        except (KeyError, TypeError, ValueError):
            pass
        else:
            errors.append(f"{removed_step}: next stage accepted execution without its direct predecessor")

    # Every declared enterprise resource is operationally required for its
    # stage: removing the grounded rows for this case must make the full chain
    # fail rather than silently falling back to a fixture or constant.
    for step_id in steps:
        for resource_id in sub_operations[step_id].effect.read_set:
            candidate = EnvironmentState(dict(state.resources))
            candidate.resources[resource_id] = [
                row for row in candidate.resources[resource_id]
                if not (
                    row.get("case_entity_id") == arguments["entity_id"]
                    and row.get("case_scope_id") == arguments["scope_id"]
                    and row.get("case_qualifier") == arguments["qualifier"]
                )
            ]
            try:
                execute_decomposition(base.decomposition, sub_operations, candidate, arguments)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                errors.append(f"{step_id}: declared resource {resource_id} is not causally required")

    return ValidationResult(
        not errors, tuple(errors),
        details={"stage_outputs": stage_outputs, "runtime_digests": runtime_digests},
    )
