from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from .agent_harness import HarnessResult
from .execution import EnvironmentState, canonical_json, execute_base, execute_decomposition, execute_sub_operation, matches_schema
from .registry import V2Registry
from .types import JsonObject, V2QueryCase


@dataclass(frozen=True)
class ReachabilityResult:
    status: str
    witness: tuple[JsonObject, ...]
    transitions_expanded: int
    max_depth: int
    transition_limit: int

    def to_dict(self) -> JsonObject:
        return {
            "status": self.status,
            "witness": copy.deepcopy(list(self.witness)),
            "transitions_expanded": self.transitions_expanded,
            "max_depth": self.max_depth,
            "transition_limit": self.transition_limit,
        }


def _walk_entities(value: Any) -> Iterable[JsonObject]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_entities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_entities(child)


def _dedupe(values: Iterable[Any]) -> tuple[Any, ...]:
    found: dict[str, Any] = {}
    for value in values:
        try:
            found.setdefault(canonical_json(value), copy.deepcopy(value))
        except (TypeError, ValueError):
            continue
    return tuple(found[key] for key in sorted(found))


def _candidate_arguments(
    operation: Any,
    *,
    case_arguments: JsonObject,
    state: EnvironmentState,
    outputs: tuple[Any, ...],
    entities: tuple[JsonObject, ...] | None = None,
    owner_operation_id: str | None = None,
) -> tuple[JsonObject, ...]:
    schema = operation.input_schema
    entities = entities if entities is not None else tuple(_walk_entities(state.resources))
    candidates: list[JsonObject] = []
    if operation.canonical_parameters == ("payload",):
        # Payloads are causal values: only outputs produced earlier on this
        # search path may be consumed.  Private rows and guessed schema-shaped
        # objects are intentionally excluded.
        candidates.extend({"payload": payload} for payload in _dedupe(outputs))
    else:
        if all(name in case_arguments for name in operation.canonical_parameters):
            candidates.append({name: case_arguments[name] for name in operation.canonical_parameters})
        owner_rows = tuple(
            entity for entity in entities
            if owner_operation_id is None or entity.get("operation_id") == owner_operation_id
        )
        for entity in owner_rows:
            if all(name in entity for name in operation.canonical_parameters):
                candidates.append({name: entity[name] for name in operation.canonical_parameters})
    return tuple(
        candidate
        for candidate in _dedupe(candidates)
        if isinstance(candidate, dict) and matches_schema(candidate, schema)
    )


def _goal_matches(*, output: Any, state: EnvironmentState, expected_output: Any, expected_digest: str) -> bool:
    return output == expected_output and state.digest() == expected_digest


def deterministic_reachability(
    registry: V2Registry,
    case: V2QueryCase,
    *,
    visible_operation_ids: tuple[str, ...],
    initial_state: EnvironmentState,
    expected_output: Any,
    expected_digest: str,
    max_depth: int = 6,
    transition_limit: int = 100_000,
) -> ReachabilityResult:
    """Search the finite benchmark argument grammar; never consult an LLM."""
    visible = tuple(sorted(visible_operation_ids))
    initial_entities = tuple(_walk_entities(initial_state.resources))
    owners = {
        step_id: base.operation_id
        for base in registry.base_operations.values()
        for step_id in base.decomposition.steps
    }

    # Fast, exact witnesses cover the two available experimental conditions.
    gold = registry.get_base(case.expected_call.canonical_operation_id)
    if gold.operation_id in visible:
        candidate = initial_state.clone()
        result = execute_base(gold, candidate, case.expected_call.canonical_arguments)
        return ReachabilityResult(
            "reachable",
            ({"operation_id": gold.operation_id, "canonical_arguments": case.expected_call.canonical_arguments, "output": result.output},),
            1,
            max_depth,
            transition_limit,
        )
    if gold.decomposition.steps and set(gold.decomposition.steps).issubset(visible):
        candidate = initial_state.clone()
        result = execute_decomposition(gold.decomposition, registry.sub_operations, candidate, case.expected_call.canonical_arguments)
        return ReachabilityResult(
            "reachable",
            tuple(
                {"operation_id": item["operation"], "canonical_arguments": item["arguments"], "output": item["output"]}
                for item in result.trace
            ),
            len(result.trace),
            max_depth,
            transition_limit,
        )

    # Scheme A is currently read-only.  Its reachability graph can therefore be
    # collapsed by (latest output, depth), avoiding a combinatorial collection
    # of equivalent state/output-pool permutations.
    read_only = all(
        not (registry.base_operations.get(operation_id) or registry.sub_operations[operation_id]).effect.write_set
        for operation_id in visible
    )
    if read_only:
        queue = deque([(initial_state.clone(), tuple(), tuple())])
        visited_outputs: dict[str, int] = {}
        transitions = 0
        while queue:
            state, outputs, trace = queue.popleft()
            depth = len(trace)
            if depth >= max_depth:
                continue
            for operation_id in visible:
                operation = registry.base_operations.get(operation_id) or registry.sub_operations.get(operation_id)
                if operation is None:
                    continue
                if depth > 0 and operation.canonical_parameters != ("payload",):
                    continue
                for arguments in _candidate_arguments(
                    operation,
                    case_arguments=case.expected_call.canonical_arguments,
                    state=state,
                    outputs=outputs,
                    entities=initial_entities if depth == 0 else (),
                    owner_operation_id=operation_id if operation_id in registry.base_operations else owners.get(operation_id),
                ):
                    if transitions >= transition_limit:
                        return ReachabilityResult("indeterminate", (), transitions, max_depth, transition_limit)
                    transitions += 1
                    candidate = state.clone()
                    try:
                        result = (
                            execute_base(operation, candidate, arguments)
                            if operation_id in registry.base_operations
                            else execute_sub_operation(operation, candidate, arguments)
                        )
                    except Exception:
                        continue
                    step = {
                        "operation_id": operation_id,
                        "canonical_arguments": copy.deepcopy(arguments),
                        "output": copy.deepcopy(result.output),
                    }
                    next_trace = trace + (step,)
                    if _goal_matches(
                        output=result.output,
                        state=candidate,
                        expected_output=expected_output,
                        expected_digest=expected_digest,
                    ):
                        return ReachabilityResult("reachable", next_trace, transitions, max_depth, transition_limit)
                    output_key = canonical_json(result.output)
                    next_depth = len(next_trace)
                    if visited_outputs.get(output_key, max_depth + 1) <= next_depth:
                        continue
                    visited_outputs[output_key] = next_depth
                    queue.append((candidate, (result.output,), next_trace))
        return ReachabilityResult("unreachable", (), transitions, max_depth, transition_limit)

    queue = deque([(initial_state.clone(), tuple(), tuple())])
    visited = {(initial_state.digest(), "[]", 0)}
    transitions = 0
    while queue:
        state, outputs, trace = queue.popleft()
        depth = len(trace)
        if depth >= max_depth:
            continue
        for operation_id in visible:
            operation = registry.base_operations.get(operation_id) or registry.sub_operations.get(operation_id)
            if operation is None:
                continue
            arguments_list = _candidate_arguments(
                operation,
                case_arguments=case.expected_call.canonical_arguments,
                state=state,
                outputs=outputs,
                entities=tuple(_walk_entities(state.resources)),
                owner_operation_id=operation_id if operation_id in registry.base_operations else owners.get(operation_id),
            )
            for arguments in arguments_list:
                if transitions >= transition_limit:
                    return ReachabilityResult("indeterminate", (), transitions, max_depth, transition_limit)
                transitions += 1
                candidate = state.clone()
                try:
                    result = (
                        execute_base(operation, candidate, arguments)
                        if operation_id in registry.base_operations
                        else execute_sub_operation(operation, candidate, arguments)
                    )
                except Exception:
                    continue
                step = {
                    "operation_id": operation_id,
                    "canonical_arguments": copy.deepcopy(arguments),
                    "output": copy.deepcopy(result.output),
                }
                next_trace = trace + (step,)
                if _goal_matches(
                    output=result.output,
                    state=candidate,
                    expected_output=expected_output,
                    expected_digest=expected_digest,
                ):
                    return ReachabilityResult("reachable", next_trace, transitions, max_depth, transition_limit)
                next_outputs = _dedupe(outputs + (result.output,))
                key = (candidate.digest(), canonical_json(next_outputs), len(next_trace))
                if key in visited:
                    continue
                visited.add(key)
                queue.append((candidate, next_outputs, next_trace))
    return ReachabilityResult("unreachable", (), transitions, max_depth, transition_limit)


@dataclass(frozen=True)
class AgentEvaluation:
    outcome_success: bool
    successful_suffix_exact: bool | None
    unavailable_correct: bool | None
    unnecessary_call_count: int | None
    oracle_contradiction: bool
    expected_output: Any
    expected_final_state_digest: str
    reachability: ReachabilityResult

    def to_dict(self) -> JsonObject:
        return {
            "outcome_success": self.outcome_success,
            "successful_suffix_exact": self.successful_suffix_exact,
            "unavailable_correct": self.unavailable_correct,
            "unnecessary_call_count": self.unnecessary_call_count,
            "oracle_contradiction": self.oracle_contradiction,
            "expected_output": copy.deepcopy(self.expected_output),
            "expected_final_state_digest": self.expected_final_state_digest,
            "reachability": self.reachability.to_dict(),
        }


class Evaluator:
    def __init__(self, *, max_depth: int = 6, transition_limit: int = 100_000) -> None:
        self.max_depth = max_depth
        self.transition_limit = transition_limit

    def evaluate(
        self,
        registry: V2Registry,
        case: V2QueryCase,
        *,
        visible_operation_ids: tuple[str, ...],
        run: HarnessResult,
        reachability_override: ReachabilityResult | None = None,
    ) -> AgentEvaluation:
        gold = registry.get_base(case.expected_call.canonical_operation_id)
        expected_state = run.initial_state.clone()
        expected_execution = execute_base(gold, expected_state, case.expected_call.canonical_arguments)
        expected_output = expected_execution.output
        expected_digest = expected_state.digest()
        reachability = reachability_override or deterministic_reachability(
            registry,
            case,
            visible_operation_ids=visible_operation_ids,
            initial_state=run.initial_state,
            expected_output=expected_output,
            expected_digest=expected_digest,
            max_depth=self.max_depth,
            transition_limit=self.transition_limit,
        )

        outcome_success = bool(
            run.termination_reason == "finish"
            and run.evidence_call is not None
            and run.evidence_call.output == expected_output
            and run.final_digest == expected_digest
        )

        expected_chain: tuple[tuple[str, JsonObject], ...] | None
        if gold.operation_id in visible_operation_ids:
            expected_chain = ((gold.operation_id, case.expected_call.canonical_arguments),)
        elif gold.decomposition.steps and set(gold.decomposition.steps).issubset(visible_operation_ids):
            decomposition_state = registry.new_state()
            trace = execute_decomposition(
                gold.decomposition,
                registry.sub_operations,
                decomposition_state,
                case.expected_call.canonical_arguments,
            ).trace
            expected_chain = tuple((item["operation"], item["arguments"]) for item in trace)
        else:
            expected_chain = None

        suffix_exact: bool | None = None
        if expected_chain is not None and run.evidence_call is not None:
            successful = [
                call
                for call in run.calls
                if call.success and call.epoch == run.evidence_call.epoch
            ]
            evidence_index = next(
                (index for index, call in enumerate(successful) if call.call_id == run.evidence_call.call_id),
                None,
            )
            if evidence_index is not None:
                successful = successful[: evidence_index + 1]
                actual_suffix = tuple(
                    (call.operation_id, call.canonical_arguments or {})
                    for call in successful[-len(expected_chain) :]
                )
                suffix_exact = actual_suffix == expected_chain

        if reachability.status == "unreachable":
            unavailable_correct = run.termination_reason == "unavailable"
            unnecessary_calls = run.environment_call_count
        elif reachability.status == "indeterminate":
            unavailable_correct = None
            unnecessary_calls = None
        else:
            unavailable_correct = None
            unnecessary_calls = None
        contradiction = outcome_success and reachability.status == "unreachable"
        return AgentEvaluation(
            outcome_success=outcome_success,
            successful_suffix_exact=suffix_exact,
            unavailable_correct=unavailable_correct,
            unnecessary_call_count=unnecessary_calls,
            oracle_contradiction=contradiction,
            expected_output=expected_output,
            expected_final_state_digest=expected_digest,
            reachability=reachability,
        )
