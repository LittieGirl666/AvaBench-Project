"""Canonical replay construction, extracted from the original Stage 1 experiment."""

from __future__ import annotations
import json
from typing import Any
from envtoolbench.v2.agent_environment import EnvironmentCall, EnvironmentSession
from envtoolbench.v2.agent_harness import HarnessConfig
from envtoolbench.v2.query_generation import canonical_to_visible
from envtoolbench.v2.types import JsonObject, V2QueryCase


def _tool_call_message(
    call_id: str, operation_id: str, arguments: JsonObject
) -> JsonObject:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": operation_id,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        ],
    }


def _observation(
    call: EnvironmentCall,
    *,
    turn: int,
    harness_config: HarnessConfig,
) -> JsonObject:
    value: JsonObject = {
        "status": "success" if call.success else "error",
        "call_id": call.call_id,
        "operation_id": call.operation_id,
        "output": call.output if call.success else None,
        "error_category": call.error_category,
        "error": call.error,
        "remaining_budget": {
            "model_turns": max(harness_config.max_model_turns - turn, 0),
            "environment_calls": max(harness_config.max_environment_calls - turn, 0),
            "restarts": harness_config.max_restarts,
        },
    }
    if call.success:
        value["payload_ref"] = {"ref": call.payload_transport["issued_ref"]}
        if call.changed_resources:
            value["state_delta"] = {
                "changed_resources": list(call.changed_resources),
                "state_digest_before": call.state_digest_before,
                "state_digest_after": call.state_digest_after,
            }
    return value


def _append_call(
    messages: list[JsonObject],
    *,
    call: EnvironmentCall,
    arguments: JsonObject,
    turn: int,
    harness_config: HarnessConfig,
) -> None:
    messages.append(_tool_call_message(call.call_id, call.operation_id, arguments))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": json.dumps(
                _observation(call, turn=turn, harness_config=harness_config),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )


def _execute_structural_prefix(
    *,
    registry: Any,
    case: V2QueryCase,
    visible_ids: tuple[str, ...],
    step_ids: tuple[str, ...],
    harness_config: HarnessConfig,
) -> list[JsonObject]:
    session = EnvironmentSession(registry, case, visible_operation_ids=visible_ids)
    history: list[JsonObject] = []
    payload_ref: JsonObject | None = None
    for index, operation_id in enumerate(step_ids, start=1):
        arguments = (
            case.expected_call.visible_arguments
            if index == 1
            else {"payload": payload_ref}
        )
        call_id = f"checkpoint:{case.id}:s{index}"
        call = session.execute(
            call_id=call_id, operation_id=operation_id, arguments=arguments
        )
        if not call.success:
            raise ValueError(
                f"checkpoint replay failed: {case.id} {operation_id}: {call.error}"
            )
        payload_ref = {"ref": call.payload_transport["issued_ref"]}
        _append_call(
            history,
            call=call,
            arguments=arguments,
            turn=index,
            harness_config=harness_config,
        )
    return history


def _execute_semantic_history(
    *,
    registry: Any,
    case: V2QueryCase,
    resolution: Any,
    harness_config: HarnessConfig,
) -> list[JsonObject]:
    session = EnvironmentSession(
        registry,
        case,
        visible_operation_ids=resolution.visible_operation_ids,
        execution_failures=resolution.execution_failures(),
    )
    history: list[JsonObject] = []
    operations = tuple(resolution.nuisance_operation_ids) + (
        resolution.decoy_operation_id,
    )
    if len(resolution.nuisance_operation_ids) != 2:
        raise ValueError(
            f"semantic design requires exactly two nuisance observers: {case.id}"
        )
    for index, operation_id in enumerate(operations, start=1):
        operation = registry.get_base(operation_id)
        arguments = canonical_to_visible(
            operation, case.expected_call.canonical_arguments
        )
        call = session.execute(
            call_id=f"checkpoint:{case.id}:s{index}",
            operation_id=operation_id,
            arguments=arguments,
        )
        if operation_id != resolution.decoy_operation_id and not call.success:
            raise ValueError(
                f"semantic observer replay failed: {case.id} {operation_id}"
            )
        _append_call(
            history,
            call=call,
            arguments=arguments,
            turn=index,
            harness_config=harness_config,
        )
    return history
