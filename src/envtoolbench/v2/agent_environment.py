from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .execution import EnvironmentState, execute_base, execute_sub_operation
from .query_generation import visible_to_canonical
from .registry import V2Registry
from .types import JsonObject, V2QueryCase


PAYLOAD_TRANSPORT: JsonObject = {
    "mode": "call_reference",
    "version": 1,
    "strict": True,
}


class PayloadReferenceError(ValueError):
    def __init__(self, message: str, transport: JsonObject) -> None:
        super().__init__(message)
        self.transport = transport


class ForcedExecutionFailure(RuntimeError):
    """Deterministic matched-control failure injected by an environment arm."""


def _parse_arguments(raw_arguments: Any) -> JsonObject:
    if isinstance(raw_arguments, str):
        raw_arguments = json.loads(raw_arguments)
    if not isinstance(raw_arguments, dict):
        raise TypeError("tool arguments must be a JSON object")
    return raw_arguments


def _error_category(exc: Exception) -> str:
    if isinstance(exc, ForcedExecutionFailure):
        return "execution_precondition"
    if isinstance(exc, json.JSONDecodeError):
        return "argument_json"
    if isinstance(exc, TypeError):
        return "argument_type"
    if isinstance(exc, KeyError):
        return "execution_key_error"
    message = str(exc)
    if "missing=" in message or "missing required" in message:
        return "missing_argument"
    if "unexpected=" in message or "unexpected arguments" in message:
        return "unexpected_argument"
    if isinstance(exc, ValueError):
        return "argument_value"
    return "execution_error"


@dataclass(frozen=True)
class EnvironmentCall:
    call_id: str
    epoch: int
    operation_id: str
    raw_arguments: Any
    visible_arguments: JsonObject
    canonical_arguments: JsonObject | None
    success: bool
    output: Any
    error_category: str | None
    error: str | None
    state_digest_before: str
    state_digest_after: str
    runtime_digest_before: str
    runtime_digest_after: str
    payload_transport: JsonObject
    changed_resources: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        data = {
            "call_id": self.call_id,
            "epoch": self.epoch,
            "operation_id": self.operation_id,
            "raw_arguments": copy.deepcopy(self.raw_arguments),
            "visible_arguments": copy.deepcopy(self.visible_arguments),
            "canonical_arguments": copy.deepcopy(self.canonical_arguments),
            "success": self.success,
            "output": copy.deepcopy(self.output),
            "error_category": self.error_category,
            "error": self.error,
            "state_digest_before": self.state_digest_before,
            "state_digest_after": self.state_digest_after,
            "runtime_digest_before": self.runtime_digest_before,
            "runtime_digest_after": self.runtime_digest_after,
            "payload_transport": copy.deepcopy(self.payload_transport),
        }
        if self.changed_resources:
            data["changed_resources"] = list(self.changed_resources)
        return data


class EnvironmentSession:
    """Transactional execution boundary for one case.

    Failed calls run against a clone and therefore cannot leak partial writes.
    Runtime logs are kept outside authoritative enterprise state.
    """

    def __init__(
        self,
        registry: V2Registry,
        case: V2QueryCase,
        *,
        visible_operation_ids: tuple[str, ...],
        execution_failures: dict[str, str] | None = None,
    ) -> None:
        self.registry = registry
        self.case = case
        self.visible_operation_ids = frozenset(visible_operation_ids)
        self.execution_failures = dict(execution_failures or {})
        self.initial_state = registry.new_state()
        self.state = self.initial_state.clone()
        self.epoch = 0
        self.calls: list[EnvironmentCall] = []
        self._successful_by_epoch: dict[int, dict[str, EnvironmentCall]] = {0: {}}
        self._payloads_by_epoch: dict[int, dict[str, EnvironmentCall]] = {0: {}}

    def _resolve_payload_reference(
        self,
        operation: Any,
        visible_arguments: JsonObject,
    ) -> tuple[JsonObject, JsonObject]:
        if operation.canonical_parameters != ("payload",):
            return visible_arguments, {
                **PAYLOAD_TRANSPORT,
                "resolution": "not_applicable",
                "input_ref": None,
                "source_call_id": None,
                "source_epoch": None,
                "issued_ref": None,
            }

        transport = {
            **PAYLOAD_TRANSPORT,
            "resolution": "rejected",
            "input_ref": None,
            "source_call_id": None,
            "source_epoch": None,
            "issued_ref": None,
        }
        if set(visible_arguments) != {"payload"}:
            raise PayloadReferenceError("payload transport requires exactly one payload argument", transport)
        envelope = visible_arguments["payload"]
        if not isinstance(envelope, dict) or set(envelope) != {"ref"}:
            raise PayloadReferenceError(
                "payload must be an exact reference object: {'ref': 'payload:e<epoch>:s<ordinal>'}",
                transport,
            )
        reference = envelope["ref"]
        if not isinstance(reference, str) or not reference:
            raise PayloadReferenceError("payload.ref must be a non-empty string", transport)
        transport["input_ref"] = reference
        source = self._payloads_by_epoch[self.epoch].get(reference)
        if source is None:
            if any(reference in epoch_refs for epoch, epoch_refs in self._payloads_by_epoch.items() if epoch != self.epoch):
                raise PayloadReferenceError("payload.ref belongs to a stale restart epoch", transport)
            raise PayloadReferenceError(
                "payload.ref does not identify a successful call in the current epoch",
                transport,
            )
        transport.update({
            "resolution": "resolved",
            "source_call_id": source.call_id,
            "source_epoch": source.epoch,
        })
        return {"payload": copy.deepcopy(source.output)}, transport

    @property
    def initial_digest(self) -> str:
        return self.initial_state.digest()

    def execute(self, *, call_id: str, operation_id: str, arguments: Any) -> EnvironmentCall:
        before = self.state.digest()
        runtime_before = self.state.runtime_digest()
        visible_arguments: JsonObject = arguments if isinstance(arguments, dict) else {}
        canonical_arguments: JsonObject | None = None
        output: Any = None
        error_category = None
        error = None
        success = False
        changed_resources: tuple[str, ...] = ()
        payload_transport: JsonObject = {
            **PAYLOAD_TRANSPORT,
            "resolution": "not_applicable",
            "input_ref": None,
            "source_call_id": None,
            "source_epoch": None,
            "issued_ref": None,
        }

        try:
            if operation_id not in self.visible_operation_ids:
                raise LookupError(f"operation is not visible: {operation_id}")
            operation = self.registry.base_operations.get(operation_id) or self.registry.sub_operations.get(operation_id)
            if operation is None:
                raise LookupError(f"unknown operation: {operation_id}")
            visible_arguments = _parse_arguments(arguments)
            try:
                resolved_arguments, payload_transport = self._resolve_payload_reference(operation, visible_arguments)
            except PayloadReferenceError as exc:
                payload_transport = exc.transport
                raise
            canonical_arguments = visible_to_canonical(operation, resolved_arguments)
            if operation_id in self.execution_failures:
                raise ForcedExecutionFailure(self.execution_failures[operation_id])
            candidate = self.state.clone()
            if operation_id in self.registry.base_operations:
                result = execute_base(operation, candidate, canonical_arguments)
            else:
                result = execute_sub_operation(operation, candidate, canonical_arguments)
            output = result.output
            changed_resources = result.changed_resources
            self.state.restore(candidate)
            success = True
        except LookupError as exc:
            error_category = "unavailable_tool_call"
            error = str(exc)
        except Exception as exc:
            error_category = _error_category(exc)
            error = str(exc)

        issued_ref = None
        if success:
            issued_ref = f"payload:e{self.epoch}:s{len(self._payloads_by_epoch[self.epoch]) + 1}"
            payload_transport["issued_ref"] = issued_ref
        call = EnvironmentCall(
            call_id=call_id,
            epoch=self.epoch,
            operation_id=operation_id,
            raw_arguments=copy.deepcopy(arguments),
            visible_arguments=visible_arguments,
            canonical_arguments=canonical_arguments,
            success=success,
            output=output,
            error_category=error_category,
            error=error,
            state_digest_before=before,
            state_digest_after=self.state.digest(),
            runtime_digest_before=runtime_before,
            runtime_digest_after=self.state.runtime_digest(),
            payload_transport=payload_transport,
            changed_resources=changed_resources,
        )
        self.calls.append(call)
        if success:
            self._successful_by_epoch[self.epoch][call_id] = call
            self._payloads_by_epoch[self.epoch][issued_ref] = call
        return call

    def evidence(self, call_id: str) -> EnvironmentCall | None:
        call = self._successful_by_epoch[self.epoch].get(call_id)
        if call is None:
            return None
        operation = self.registry.base_operations.get(call.operation_id) or self.registry.sub_operations.get(call.operation_id)
        if operation is None or not set(self.case.target_postconditions).issubset(operation.effect.postconditions):
            return None
        return call

    def restart(self) -> dict[str, Any]:
        before = self.state.digest()
        runtime_before = self.state.runtime_digest()
        self.state.restore(self.initial_state)
        self.epoch += 1
        self._successful_by_epoch[self.epoch] = {}
        self._payloads_by_epoch[self.epoch] = {}
        return {
            "status": "restarted",
            "epoch": self.epoch,
            "state_digest_before": before,
            "state_digest_after": self.state.digest(),
            "runtime_digest_before": runtime_before,
            "runtime_digest_after": self.state.runtime_digest(),
        }

    def finalize(self) -> dict[str, Any]:
        final_state = self.state.clone()
        final_digest = final_state.digest()
        final_runtime_digest = final_state.runtime_digest()
        self.state.restore(self.initial_state)
        restored_digest = self.state.digest()
        return {
            "initial_state": self.initial_state.clone(),
            "final_state": final_state,
            "initial_digest": self.initial_digest,
            "final_digest": final_digest,
            "final_runtime_digest": final_runtime_digest,
            "rollback": {
                "restored": restored_digest == self.initial_digest,
                "restored_digest": restored_digest,
            },
            "calls": tuple(self.calls),
        }
