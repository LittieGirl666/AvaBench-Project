from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ...types import (
    BaseOperationSpec, DecompositionSpec, EffectSpec, JsonObject, SignatureFamily,
    StepBinding, SubOperationSpec, V2ResourceSpec,
)
from .manifest import DOMAIN_TITLES, OPERATIONS, RESOURCE_DECLARATIONS
from .plans import STEP_RESOURCES

CANONICAL_PARAMETERS = ("entity_id", "scope_id", "qualifier")
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {}, "postcondition": {"type": "string"}, "fixture": {"type": "string"},
    },
    "required": ["result", "postcondition", "fixture"],
    "additionalProperties": False,
}
CONTEXT_PROPERTIES = {
    "case_entity_id": {"type": "string"},
    "case_scope_id": {"type": "string"},
    "case_qualifier": {"type": "string"},
}


@dataclass
class EnterpriseSuite:
    resources: dict[str, V2ResourceSpec]
    resource_rows: JsonObject
    bases: dict[str, BaseOperationSpec]
    subs: dict[str, SubOperationSpec]
    fixtures: dict[str, list[JsonObject]]
    operation_effects: dict[str, str]


def _schema(properties: dict[str, JsonObject], required: tuple[str, ...] | None = None) -> JsonObject:
    return {"type": "object", "properties": properties, "required": list(required or properties), "additionalProperties": False}


def _parse_resources(domain: str) -> list[V2ResourceSpec]:
    parsed = []
    for declaration in RESOURCE_DECLARATIONS[domain].split(";"):
        name, raw_fields = declaration.split("(", 1)
        fields = {}
        for field in raw_fields[:-1].split(","):
            field_name, field_type = field.split(":")
            child: JsonObject = {"type": field_type}
            if field_type == "array":
                child["items"] = {}
            fields[field_name] = child
        fields.update({key: value for key, value in CONTEXT_PROPERTIES.items() if key not in fields})
        parsed.append((name, fields))
    specs = []
    for name, fields in parsed:
        relations = []
        for field_name in fields:
            if field_name.startswith("case_"):
                continue
            if not field_name.endswith("_id"):
                continue
            for target_name, target_fields in parsed:
                if target_name != name and field_name in target_fields:
                    relations.append({"fields": [field_name], "target_resource": f"{domain}.{target_name}", "target_fields": [field_name]})
        specs.append(V2ResourceSpec(f"{domain}.{name}", "table", _schema(fields), (next(iter(fields)),), tuple(relations)))
    fixture_schema = _schema({
        "operation_id": {"type": "string"}, "entity_id": {"type": "string"},
        "scope_id": {"type": "string"}, "qualifier": {"type": "string"},
        "result": {}, "postcondition": {"type": "string"}, "fixture": {"type": "string"},
    })
    specs.append(V2ResourceSpec(f"{domain}.operation_fixtures", "private_table", fixture_schema, ("operation_id", "entity_id", "scope_id", "qualifier")))
    return specs


def _sample_row(
    spec: V2ResourceSpec,
    seed: int,
    *,
    entity_id: str,
    scope_id: str,
    qualifier: str,
    sample_index: int,
) -> JsonObject:
    row = {}
    for index, (name, child) in enumerate(spec.schema["properties"].items()):
        if name == "case_entity_id":
            row[name] = entity_id
            continue
        if name == "case_scope_id":
            row[name] = scope_id
            continue
        if name == "case_qualifier":
            row[name] = qualifier
            continue
        kind = child.get("type")
        ordinal = (seed % 97) + index + sample_index + 1
        row[name] = ({"string": f"{name.upper()}-{seed:08X}-{sample_index}", "integer": ordinal,
                      "number": round(ordinal + 0.25, 2), "boolean": ((ordinal + index) % 2 == 0),
                      "array": [f"{name.upper()}-{seed:08X}-{sample_index}"],
                      "object": {"value": f"{seed:08X}-{sample_index}"}}).get(kind)
    return row


def _family(index: int) -> SignatureFamily:
    return (SignatureFamily.RESOLVE, SignatureFamily.EVALUATE, SignatureFamily.CALCULATE)[index % 3]


def _visible(family: SignatureFamily) -> tuple[JsonObject, JsonObject, JsonObject]:
    third = {SignatureFamily.RESOLVE: "reference_id", SignatureFamily.EVALUATE: "policy_id", SignatureFamily.CALCULATE: "value"}[family]
    visible_names = ("subject_id", "context_id", third)
    visible_schema = _schema({name: {"type": "string"} for name in visible_names})
    v2c = dict(zip(visible_names, CANONICAL_PARAMETERS))
    c2v = {value: key for key, value in v2c.items()}
    return visible_schema, v2c, c2v


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fixture_label(arguments: JsonObject) -> str:
    digest = hashlib.sha256(_canonical(arguments).encode()).hexdigest()
    return f"fixture-{digest[:12]}"


def _case_rows(state, read_set: tuple[str, ...], arguments: JsonObject) -> JsonObject:
    observations: JsonObject = {}
    for resource_id in read_set:
        table = state.resources.get(resource_id)
        if not isinstance(table, list):
            raise TypeError(f"resource {resource_id} is not a table")
        selected = [
            copy.deepcopy(row)
            for row in table
            if row.get("case_entity_id") == arguments["entity_id"]
            and row.get("case_scope_id") == arguments["scope_id"]
            and row.get("case_qualifier") == arguments["qualifier"]
        ]
        if not selected:
            raise KeyError(f"no grounded rows in {resource_id} for the requested entity context")
        observations[resource_id] = selected
    return observations


def _advance_digest(previous: str, step_id: str, observations: JsonObject) -> str:
    material = {"previous": previous, "step_id": step_id, "observations": observations}
    return hashlib.sha256(_canonical(material).encode()).hexdigest()


def _semantic_result(
    operation_id: str, target_effect: str, observations: list[JsonObject], arguments: JsonObject,
) -> Any:
    """Derive a typed business result from the grounded rows, never a fixture."""
    rows = [row for stage in observations for table in stage.values() for row in table]
    values = [
        value for row in rows for name, value in row.items()
        # ``version`` is state-transition bookkeeping, not a business fact.
        # Structural registries do not contain it; excluding it keeps the
        # stateful observer semantically identical before and after the
        # orthogonal mutation layer adds optimistic-concurrency metadata.
        if not name.startswith("case_") and name != "version"
    ]
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    booleans = [value for value in values if isinstance(value, bool)]
    strings = [value for value in values if isinstance(value, str)]
    text = target_effect.lower()
    number_words = ("quantity", "amount", "price", "charge", "fee", "floor", "total", "limit", "variance", "balance", "basis", "score", "ratio", "utilization")
    date_words = ("date", "deadline", "expiry", "arrival")
    list_words = ("ordered", "ranking", "allocation", "list", "sample", "actions", "flags", "conflicts", "impact")
    boolean_words = ("decision", "eligibility", "feasibility", "permission", "compliance", "completeness", "applicability", "readiness")
    if any(word in text for word in number_words):
        if not numbers:
            raise ValueError(f"{operation_id}: numeric effect has no numeric resource facts")
        if "ratio" in text or "utilization" in text:
            midpoint = max(1, len(numbers) // 2)
            return round(sum(numbers[midpoint:]) / max(sum(numbers[:midpoint]), 1.0), 4)
        if "variance" in text or "after" in text:
            return round(numbers[0] - sum(numbers[1:]), 2)
        return round(sum(numbers) / len(numbers), 2)
    if any(word in text for word in date_words):
        reference = dt.date.fromisoformat(arguments["qualifier"])
        offset = int(sum(numbers) if numbers else len(strings)) % 28 + 1
        return (reference + dt.timedelta(days=offset)).isoformat()
    if operation_id == "resolve_section_summary":
        facts = [value for value in strings if value not in arguments.values()][:2]
        return {"heading": "Commercial terms", "facts": facts or ["No structured facts"]}
    if any(word in text for word in list_words):
        return list(dict.fromkeys(strings))[:2]
    if any(word in text for word in boolean_words):
        return bool(sum(1 for value in booleans if value) + int(sum(numbers))) and (
            (sum(1 for value in booleans if value) + int(sum(numbers))) % 2 == 0
        )
    material = _canonical({"operation_id": operation_id, "rows": rows})
    return f"{operation_id.removeprefix('resolve_').upper()}-{hashlib.sha256(material.encode()).hexdigest()[:8].upper()}"


def _final_output(
    operation_id: str, target_effect: str, digest: str, arguments: JsonObject,
    observations: list[JsonObject],
) -> JsonObject:
    domain = operation_id.split("__", 1)[0] if "__" in operation_id else ""
    canonical_operation_id = operation_id.split("__", 2)[1] if "__" in operation_id else operation_id
    postcondition = f"{domain}.{canonical_operation_id}.complete" if domain else ""
    return {
        "result": _semantic_result(canonical_operation_id, target_effect, observations, arguments),
        "postcondition": postcondition,
        "fixture": _fixture_label(arguments),
    }


def _pipeline_output(
    state,
    *,
    domain: str,
    operation_id: str,
    step_ids: tuple[str, ...],
    step_read_sets: tuple[tuple[str, ...], ...],
    target_effect: str,
    arguments: JsonObject,
) -> JsonObject:
    digest = hashlib.sha256(_canonical(arguments).encode()).hexdigest()
    accumulated_observations: list[JsonObject] = []
    for step_id, read_set in zip(step_ids, step_read_sets):
        observations = _case_rows(state, read_set, arguments)
        accumulated_observations.append(observations)
        digest = _advance_digest(digest, step_id, observations)
    return _final_output(
        f"{domain}__{operation_id}", target_effect, digest, arguments, accumulated_observations,
    )


def _base_handler(
    *, domain: str, operation_id: str, step_ids: tuple[str, ...],
    step_read_sets: tuple[tuple[str, ...], ...], target_effect: str,
):
    def handler(state, entity_id: str, scope_id: str, qualifier: str):
        return _pipeline_output(
            state, domain=domain, operation_id=operation_id, step_ids=step_ids,
            step_read_sets=step_read_sets, target_effect=target_effect,
            arguments={"entity_id": entity_id, "scope_id": scope_id, "qualifier": qualifier},
        )
    handler.__name__ = operation_id
    return handler


def _stage_handler(
    *, domain: str, operation_id: str, step_id: str, previous_step_id: str | None,
    stage_index: int, stage_count: int, read_set: tuple[str, ...], target_effect: str,
):
    def handler(state, **kwargs):
        workspace = state.runtime_resources
        workspace.setdefault("schema_version", "scheme_a.workspace.v1")
        workspace.setdefault("state_version", 0)
        artifacts = workspace.setdefault("artifacts", {})
        if stage_index == 0:
            arguments = {name: kwargs[name] for name in CANONICAL_PARAMETERS}
            previous_digest = hashlib.sha256(_canonical(arguments).encode()).hexdigest()
            accumulated_observations: list[JsonObject] = []
        else:
            payload = kwargs["payload"]
            if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
                raise ValueError("payload must be an artifact returned by the preceding operation")
            artifact_id = payload["result"].get("artifact_id")
            artifact = artifacts.get(artifact_id)
            if artifact is None or artifact.get("output") != payload:
                raise ValueError("payload provenance is not present in the current execution workspace")
            if artifact.get("operation_id") != operation_id or artifact.get("step_id") != previous_step_id:
                raise ValueError("payload was produced by an incompatible operation stage")
            if artifact.get("stage_index") != stage_index - 1:
                raise ValueError("payload stage is not the direct predecessor")
            arguments = copy.deepcopy(artifact["arguments"])
            previous_digest = artifact["evidence_digest"]
            accumulated_observations = copy.deepcopy(artifact["observations"])
        observations = _case_rows(state, read_set, arguments)
        accumulated_observations.append(observations)
        evidence_digest = _advance_digest(previous_digest, step_id, observations)
        if stage_index == stage_count - 1:
            output = _final_output(
                f"{domain}__{operation_id}", target_effect, evidence_digest,
                arguments, accumulated_observations,
            )
            workspace.setdefault("completions", []).append({
                "operation_id": operation_id, "step_id": step_id,
                "arguments": copy.deepcopy(arguments), "output": copy.deepcopy(output),
            })
        else:
            artifact_id = hashlib.sha256(
                f"{operation_id}:{stage_index}:{evidence_digest}".encode()
            ).hexdigest()[:32]
            output = {
                "result": {
                    "artifact_id": artifact_id,
                    "stage": stage_index + 1,
                    "evidence_digest": evidence_digest,
                },
                "postcondition": f"{domain}.{operation_id}.stage_{stage_index + 1}",
                "fixture": _fixture_label(arguments),
            }
            artifacts[artifact_id] = {
                "operation_id": operation_id, "step_id": step_id,
                "stage_index": stage_index, "arguments": copy.deepcopy(arguments),
                "evidence_digest": evidence_digest,
                "observations": copy.deepcopy(accumulated_observations),
                "output": copy.deepcopy(output),
            }
        workspace["state_version"] += 1
        return output
    return handler


def _description(family: SignatureFamily, source: str, resource_id: str, effect: str, postcondition: str) -> str:
    summary = {
        SignatureFamily.RESOLVE: "Resolve one deterministic object from the current enterprise environment.",
        SignatureFamily.EVALUATE: "Evaluate one deterministic decision from current enterprise records and rules.",
        SignatureFamily.CALCULATE: "Calculate one deterministic value from current enterprise records and rules.",
    }[family]
    return f"{summary}\n\nImplementation:\n{source}\n\nEffect contract:\n- reads: {resource_id}\n- returns: {effect}\n- postconditions: {postcondition}"


def _sub_description(*, label: str, source: str, read_set: tuple[str, ...], accepts_payload: bool) -> str:
    input_note = (
        "Pass the preceding tool result's operation_output as payload."
        if accepts_payload
        else "Use the request's entity, scope, and qualifier arguments."
    )
    reads = ", ".join(read_set) or "transaction workspace"
    return (
        "Execute one deterministic partial enterprise operation. This tool may need to be combined with other available tools."
        f"\n\nInput dependency:\n- {input_note}"
        f"\n\nImplementation:\n{source}"
        f"\n\nEffect contract:\n- reads: {reads}\n- returns: deterministic partial output\n- operation: {label}"
    )


def build_suite(*, domains: list[str] | None = None, seed: int = 0) -> EnterpriseSuite:
    selected = list(DOMAIN_TITLES) if domains is None else list(domains)
    unknown = set(selected) - set(DOMAIN_TITLES)
    if unknown:
        raise ValueError(f"unknown domains: {sorted(unknown)}")
    resources: dict[str, V2ResourceSpec] = {}
    rows: JsonObject = {}
    bases: dict[str, BaseOperationSpec] = {}
    subs: dict[str, SubOperationSpec] = {}
    fixtures: dict[str, list[JsonObject]] = {}
    effects: dict[str, str] = {}
    for domain_index, domain in enumerate(selected):
        for spec in _parse_resources(domain):
            resources[spec.resource_id] = spec
            rows[spec.resource_id] = []
        fixture_resource = f"{domain}.operation_fixtures"
        fixtures[domain] = []
        for operation_index, (operation_id, decomposition_text, target_effect) in enumerate(OPERATIONS[domain]):
            family = _family(operation_index)
            visible_schema, v2c, c2v = _visible(family)
            postcondition = f"{domain}.{operation_id}.complete"
            effects[operation_id] = target_effect
            steps = decomposition_text.split("+")
            step_ids = tuple(f"{domain}__{operation_id}__{label}" for label in steps)
            short_read_sets = STEP_RESOURCES.get(operation_id)
            if short_read_sets is None or len(short_read_sets) != len(step_ids):
                raise ValueError(f"{operation_id}: explicit step resource plan is missing or has the wrong length")
            step_read_sets = tuple(
                tuple(f"{domain}.{resource_name}" for resource_name in resource_names)
                for resource_names in short_read_sets
            )
            missing = {resource_id for read_set in step_read_sets for resource_id in read_set} - set(resources)
            if missing:
                raise ValueError(f"{operation_id}: step plan references missing resources: {sorted(missing)}")

            # Ground every operation case in its declared enterprise resources
            # before deriving the expected result.  Fixtures are observations of
            # executable semantics, never the source used by a handler.
            op_fixtures = []
            for fixture_index in range(5):
                arguments = {
                    "entity_id": f"{domain[:3].upper()}-{operation_index + 1:02d}-{fixture_index + 1}",
                    "scope_id": f"CTX-{domain_index + 1:02d}",
                    "qualifier": f"2026-07-{fixture_index + 1:02d}",
                }
                for resource_id in dict.fromkeys(resource_id for read_set in step_read_sets for resource_id in read_set):
                    spec = resources[resource_id]
                    for sample_index in range(2):
                        material = f"{seed}:{domain}:{operation_id}:{fixture_index}:{resource_id}:{sample_index}"
                        row_seed = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)
                        rows[resource_id].append(_sample_row(
                            spec, row_seed, entity_id=arguments["entity_id"],
                            scope_id=arguments["scope_id"], qualifier=arguments["qualifier"],
                            sample_index=sample_index,
                        ))
                state_view = type("StateView", (), {"resources": rows})()
                output = _pipeline_output(
                    state_view, domain=domain, operation_id=operation_id,
                    step_ids=step_ids, step_read_sets=step_read_sets,
                    target_effect=target_effect, arguments=arguments,
                )
                row = {"operation_id": operation_id, **arguments, **output}
                rows[fixture_resource].append(row)
                op_fixtures.append({"arguments": arguments, "output": copy.deepcopy(output)})
            fixtures[operation_id] = op_fixtures

            bindings = []
            for step_index, step_id in enumerate(step_ids):
                step_read_set = step_read_sets[step_index]
                if step_index == 0:
                    input_schema = _schema({name: {"type": "string"} for name in CANONICAL_PARAMETERS})
                    sub_visible_schema = visible_schema
                    sub_v2c, sub_c2v = v2c, c2v
                    arguments = {name: f"$input.{name}" for name in CANONICAL_PARAMETERS}
                else:
                    input_schema = _schema({"payload": {"type": "object"}})
                    sub_visible_schema = input_schema
                    sub_v2c = {"payload": "payload"}
                    sub_c2v = {"payload": "payload"}
                    arguments = {"payload": f"$step.{step_ids[step_index - 1]}.output"}
                handler = _stage_handler(
                    domain=domain, operation_id=operation_id, step_id=step_id,
                    previous_step_id=None if step_index == 0 else step_ids[step_index - 1],
                    stage_index=step_index, stage_count=len(step_ids), read_set=step_read_set,
                    target_effect=target_effect,
                )
                step_postconditions = (postcondition,) if step_index == len(step_ids) - 1 else ()
                sub_effect = EffectSpec(
                    step_read_set, (), OUTPUT_SCHEMA, step_postconditions,
                    f"{domain}.{operation_id}.stage_{step_index + 1}",
                )
                sub_source = ast.unparse(ast.parse(
                    f"def {steps[step_index]}(*args, **kwargs):\n"
                    f"    observations = read_declared_resources({tuple(r.split('.', 1)[1] for r in step_read_set)!r})\n"
                    f"    return advance_verified_stage({step_index + 1}, observations)"
                ))
                subs[step_id] = SubOperationSpec(
                    step_id, domain, family, tuple(input_schema["properties"]), input_schema,
                    sub_visible_schema, sub_v2c, sub_c2v,
                    _sub_description(label=steps[step_index], source=sub_source, read_set=step_read_set, accepts_payload=step_index > 0),
                    sub_effect, handler, sub_source,
                )
                bindings.append(StepBinding(step_id, arguments))
            input_schema = _schema({name: {"type": "string"} for name in CANONICAL_PARAMETERS})
            operation_business_resources = tuple(dict.fromkeys(
                resource_id for read_set in step_read_sets for resource_id in read_set
            ))
            visible_tables = tuple(resource_id.split(".", 1)[1] for resource_id in operation_business_resources)
            raw_source = (
                f"def {operation_id}(entity_id, scope_id, qualifier):\n"
                f"    source_tables = {visible_tables!r}\n"
                f"    return execute_deterministic_resource_pipeline(source_tables, entity_id, scope_id, qualifier)"
            )
            source = ast.unparse(ast.parse(raw_source))
            effect = EffectSpec(operation_business_resources, (), OUTPUT_SCHEMA, (postcondition,), f"{domain}.effect_{operation_index // 2}")
            bases[operation_id] = BaseOperationSpec(
                operation_id, domain, family, CANONICAL_PARAMETERS, input_schema,
                visible_schema, v2c, c2v,
                _description(family, source, ", ".join(effect.read_set), target_effect, postcondition),
                effect, _base_handler(
                    domain=domain, operation_id=operation_id, step_ids=step_ids,
                    step_read_sets=step_read_sets, target_effect=target_effect,
                ), source,
                DecompositionSpec(step_ids, tuple(bindings), step_ids[-1]),
                (f"{operation_id}:fixture-2",),
            )
    return EnterpriseSuite(resources, rows, bases, subs, fixtures, effects)
