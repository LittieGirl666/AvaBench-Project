from __future__ import annotations

from .execution import EnvironmentState, execute_base, execute_sub_operation
from .types import BaseOperationSpec, JsonObject, SubOperationSpec, TaskGoalSpec, ValidationResult


def validate_single_step_unique(*, goal: TaskGoalSpec, gold: BaseOperationSpec, candidate_bases: list[BaseOperationSpec], candidate_sub_operations: list[SubOperationSpec], state: EnvironmentState, arguments: JsonObject) -> ValidationResult:
    witnesses = []
    for candidate in candidate_bases:
        if candidate.operation_id == gold.operation_id:
            continue
        same_post = bool(set(goal.target_postconditions) & set(candidate.effect.postconditions))
        try:
            candidate_state = EnvironmentState({
                resource_id: state.resources[resource_id]
                for resource_id in candidate.effect.read_set
            })
            result = execute_base(candidate, candidate_state, arguments)
            same_output = result.output == goal.expected_output
        except (KeyError, TypeError, ValueError):
            same_output = False
            result = None
        if same_output or same_post:
            witnesses.append({"candidate": candidate.operation_id, "reason": "same_output" if same_output else "same_postcondition", "output": None if result is None else result.output})
    # A sub-operation is a single-step substitute only when it both declares
    # the goal and can execute directly from the query's canonical arguments.
    # A terminal payload stage whose predecessor is mandatory is not a
    # single-step substitute merely because it declares completion.
    for candidate in candidate_sub_operations:
        if not (set(goal.target_postconditions) & set(candidate.effect.postconditions)):
            continue
        if not all(name in arguments for name in candidate.canonical_parameters):
            continue
        candidate_arguments = {name: arguments[name] for name in candidate.canonical_parameters}
        try:
            candidate_state = EnvironmentState({
                resource_id: state.resources[resource_id]
                for resource_id in candidate.effect.read_set
            })
            result = execute_sub_operation(candidate, candidate_state, candidate_arguments)
        except (KeyError, TypeError, ValueError):
            continue
        if result.output == goal.expected_output:
            witnesses.append({"candidate": candidate.operation_id, "reason": "same_output_and_effect", "output": result.output})
    return ValidationResult(not witnesses, (() if not witnesses else ("single-step substitute found",)), tuple(witnesses))
