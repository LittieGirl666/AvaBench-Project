from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .types import BaseOperationSpec, DecompositionSpec, ExecutionResult, JsonObject, SubOperationSpec


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class EnvironmentState:
    resources: JsonObject
    runtime_resources: JsonObject = field(default_factory=dict)

    def clone(self) -> "EnvironmentState":
        return EnvironmentState(copy.deepcopy(self.resources), copy.deepcopy(self.runtime_resources))

    def digest(self) -> str:
        """Digest authoritative enterprise resources, excluding transaction state."""
        return hashlib.sha256(canonical_json(self.resources).encode()).hexdigest()

    def runtime_digest(self) -> str:
        return hashlib.sha256(canonical_json(self.runtime_resources).encode()).hexdigest()

    def restore(self, snapshot: "EnvironmentState") -> None:
        self.resources = copy.deepcopy(snapshot.resources)
        self.runtime_resources = copy.deepcopy(snapshot.runtime_resources)

    def begin_transaction(self, *, case_id: str, root_operation_id: str) -> None:
        self.runtime_resources = {
            "case_id": case_id,
            "root_operation_id": root_operation_id,
            "state_version": 0,
            "status": "in_progress",
            "attempts": [],
        }

    def record_tool_attempt(
        self,
        *,
        operation_id: str | None,
        arguments: Any,
        success: bool,
        output: Any = None,
        error: str | None = None,
        status: str | None = None,
    ) -> None:
        if not self.runtime_resources:
            raise RuntimeError("transaction workspace is not initialized")
        self.runtime_resources["state_version"] += 1
        attempt = {
            "operation_id": operation_id,
            "arguments": copy.deepcopy(arguments),
            "success": success,
        }
        if output is not None:
            attempt["output"] = copy.deepcopy(output)
        if error is not None:
            attempt["error"] = error
        self.runtime_resources["attempts"].append(attempt)
        if status is not None:
            self.runtime_resources["status"] = status

    def public_runtime_snapshot(self, *, observed_resources: tuple[str, ...] = ()) -> JsonObject:
        runtime = copy.deepcopy(self.runtime_resources)
        runtime.pop("root_operation_id", None)
        observations = {
            resource_id: copy.deepcopy(self.resources[resource_id])
            for resource_id in observed_resources
            if resource_id in self.resources and not resource_id.endswith(".operation_fixtures")
        }
        return {
            **runtime,
            "resource_observations": observations,
        }


def _check_required(schema: JsonObject, arguments: JsonObject) -> None:
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    if schema.get("additionalProperties") is False:
        unexpected = set(arguments) - set(schema.get("properties", {}))
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(sorted(unexpected))}")
    for name, value in arguments.items():
        expected = schema.get("properties", {}).get(name, {}).get("type")
        valid = expected is None or (
            (expected == "string" and isinstance(value, str))
            or (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (expected == "boolean" and isinstance(value, bool))
            or (expected == "object" and isinstance(value, dict))
            or (expected == "array" and isinstance(value, list))
        )
        if not valid:
            raise TypeError(f"argument {name!r} must have type {expected}")


def matches_schema(value: Any, schema: JsonObject) -> bool:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        if any(k not in value for k in schema.get("required", [])):
            return False
        return all(matches_schema(value[k], child) for k, child in schema.get("properties", {}).items() if k in value)
    if kind == "array":
        return isinstance(value, list) and all(matches_schema(v, schema.get("items", {})) for v in value)
    return kind is None or {"string": isinstance(value, str), "number": isinstance(value, (int, float)), "integer": isinstance(value, int), "boolean": isinstance(value, bool)}.get(kind, True)


def changed_resource_ids(before: JsonObject, after: JsonObject) -> tuple[str, ...]:
    """Return the exact authoritative resource-level write set."""

    return tuple(
        resource_id
        for resource_id in sorted(set(before) | set(after))
        if canonical_json(before.get(resource_id)) != canonical_json(after.get(resource_id))
    )


def _validate_exact_write_set(
    operation: BaseOperationSpec | SubOperationSpec,
    *,
    before: JsonObject,
    after: JsonObject,
) -> tuple[str, ...]:
    actual = changed_resource_ids(before, after)
    declared = tuple(sorted(operation.effect.write_set))
    if actual != declared:
        raise RuntimeError(
            f"{operation.operation_id} write-set mismatch: "
            f"declared={list(declared)}, actual={list(actual)}"
        )
    return actual


def execute_base(operation: BaseOperationSpec, state: EnvironmentState, canonical_arguments: JsonObject) -> ExecutionResult:
    _check_required(operation.input_schema, canonical_arguments)
    missing_resources = set(operation.effect.read_set) - set(state.resources)
    if missing_resources:
        raise KeyError(f"missing declared resources: {sorted(missing_resources)}")
    resources_before = (
        copy.deepcopy(state.resources)
        if operation.effect.write_set
        else None
    )
    before = state.digest()
    output = operation.handler(state, **canonical_arguments)
    after = state.digest()
    if resources_before is None:
        if before != after:
            raise RuntimeError(
                f"read-only operation {operation.operation_id} mutated state"
            )
        changed_resources = ()
    else:
        changed_resources = _validate_exact_write_set(
            operation,
            before=resources_before,
            after=state.resources,
        )
    if not matches_schema(output, operation.effect.output_schema):
        raise TypeError(f"output from {operation.operation_id} does not match schema")
    return ExecutionResult(
        output,
        operation.effect.output_schema,
        before,
        after,
        ({"operation": operation.operation_id, "arguments": canonical_arguments, "output": output},),
        changed_resources,
    )


def execute_sub_operation(operation: SubOperationSpec, state: EnvironmentState, canonical_arguments: JsonObject) -> ExecutionResult:
    _check_required(operation.input_schema, canonical_arguments)
    missing_resources = set(operation.effect.read_set) - set(state.resources)
    if missing_resources:
        raise KeyError(f"missing declared resources: {sorted(missing_resources)}")
    resources_before = (
        copy.deepcopy(state.resources)
        if operation.effect.write_set
        else None
    )
    before = state.digest()
    output = operation.handler(state, **canonical_arguments)
    after = state.digest()
    if resources_before is None:
        if before != after:
            raise RuntimeError(
                f"read-only operation {operation.operation_id} mutated state"
            )
        changed_resources = ()
    else:
        changed_resources = _validate_exact_write_set(
            operation,
            before=resources_before,
            after=state.resources,
        )
    if not matches_schema(output, operation.effect.output_schema):
        raise TypeError(f"output from {operation.operation_id} does not match schema")
    return ExecutionResult(
        output,
        operation.effect.output_schema,
        before,
        after,
        ({"operation": operation.operation_id, "arguments": canonical_arguments, "output": output},),
        changed_resources,
    )


def _resolve(value: Any, inputs: JsonObject, step_outputs: JsonObject) -> Any:
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    if value.startswith("$input."):
        return inputs[value[7:]]
    if value.startswith("$step."):
        parts = value.split(".")[1:]
        current: Any = step_outputs[parts[0]]
        if len(parts) > 1 and parts[1] == "output":
            parts = parts[2:]
        else:
            parts = parts[1:]
        for part in parts:
            current = current[part]
        return current
    raise ValueError(f"unsupported binding {value}")


def resolve_step_arguments(
    decomposition: DecompositionSpec,
    *,
    step_id: str,
    inputs: JsonObject,
    step_outputs: JsonObject,
) -> JsonObject:
    bindings = {binding.step_id: binding.arguments for binding in decomposition.bindings}
    if step_id not in bindings:
        raise KeyError(f"missing bindings for {step_id}")
    return {name: _resolve(value, inputs, step_outputs) for name, value in bindings[step_id].items()}


def execute_decomposition(decomposition: DecompositionSpec, sub_operations: dict[str, SubOperationSpec], state: EnvironmentState, canonical_arguments: JsonObject) -> ExecutionResult:
    if len(decomposition.steps) not in (2, 3):
        raise ValueError("decomposition must contain 2 or 3 steps")
    before = state.digest()
    outputs: JsonObject = {}
    trace = []
    for step_id in decomposition.steps:
        operation = sub_operations[step_id]
        arguments = resolve_step_arguments(decomposition, step_id=step_id, inputs=canonical_arguments, step_outputs=outputs)
        result = execute_sub_operation(operation, state, arguments)
        outputs[step_id] = result.output
        trace.append({"operation": step_id, "arguments": arguments, "output": result.output})
    after = state.digest()
    final = outputs[decomposition.final_output_step]
    return ExecutionResult(final, sub_operations[decomposition.final_output_step].effect.output_schema, before, after, tuple(trace))
