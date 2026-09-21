from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

JsonObject = dict[str, Any]


class V2OperationKind(str, Enum):
    BASE = "base"
    SUB = "sub"


class SignatureFamily(str, Enum):
    RESOLVE = "resolve"
    EVALUATE = "evaluate"
    CALCULATE = "calculate"


@dataclass(frozen=True)
class EffectSpec:
    read_set: tuple[str, ...]
    write_set: tuple[str, ...]
    output_schema: JsonObject
    postconditions: tuple[str, ...]
    effect_family: str


@dataclass(frozen=True)
class V2ResourceSpec:
    resource_id: str
    kind: str
    schema: JsonObject
    primary_key: tuple[str, ...] = ()
    relationships: tuple[JsonObject, ...] = ()


@dataclass(frozen=True)
class SubOperationSpec:
    operation_id: str
    domain: str
    signature_family: SignatureFamily
    canonical_parameters: tuple[str, ...]
    input_schema: JsonObject
    visible_input_schema: JsonObject
    visible_to_canonical_arguments: JsonObject
    canonical_to_visible_arguments: JsonObject
    description: str
    effect: EffectSpec
    handler: Callable[..., Any]
    implementation_source: str


@dataclass(frozen=True)
class StepBinding:
    step_id: str
    arguments: JsonObject


@dataclass(frozen=True)
class DecompositionSpec:
    steps: tuple[str, ...]
    bindings: tuple[StepBinding, ...]
    final_output_step: str


@dataclass(frozen=True)
class BaseOperationSpec:
    operation_id: str
    domain: str
    signature_family: SignatureFamily
    canonical_parameters: tuple[str, ...]
    input_schema: JsonObject
    visible_input_schema: JsonObject
    visible_to_canonical_arguments: JsonObject
    canonical_to_visible_arguments: JsonObject
    description: str
    effect: EffectSpec
    handler: Callable[..., Any]
    implementation_source: str
    decomposition: DecompositionSpec
    non_substitutability_witnesses: tuple[str, ...]


@dataclass(frozen=True)
class V2ExpectedCall:
    canonical_operation_id: str
    canonical_arguments: JsonObject
    visible_arguments: JsonObject


@dataclass(frozen=True)
class V2QueryCase:
    id: str
    domain: str
    prompt: str
    target_postconditions: tuple[str, ...]
    expected_call: V2ExpectedCall
    hidden_metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedV2Case:
    case_id: str
    ordered_operation_ids: tuple[str, ...]
    environment_context: str
    tools: tuple[JsonObject, ...]
    expected_visible_call: JsonObject
    gold_position: int


@dataclass(frozen=True)
class ExecutionResult:
    output: Any
    output_schema: JsonObject
    state_digest_before: str
    state_digest_after: str
    trace: tuple[JsonObject, ...]
    changed_resources: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    witnesses: tuple[JsonObject, ...] = ()
    details: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class TaskGoalSpec:
    target_postconditions: tuple[str, ...]
    expected_output_schema: JsonObject
    expected_output: Any


def public_dataclass_dict(value: Any) -> Any:
    """Convert nested benchmark records to JSON data, excluding callables."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [public_dataclass_dict(v) for v in value]
    if isinstance(value, list):
        return [public_dataclass_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: public_dataclass_dict(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: public_dataclass_dict(v)
            for k, v in asdict(value).items()
            if k != "handler"
        }
    return value
