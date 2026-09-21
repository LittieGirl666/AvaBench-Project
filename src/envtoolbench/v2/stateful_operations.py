from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

from .execution import EnvironmentState, execute_base
from .query_generation import canonical_to_visible
from .registry import V2Registry, build_v2_registry
from .stateful_manifest import (
    REJECTED_OBSERVERS,
    STATE_ACTION_PAIRS,
    StateActionDefinition,
    StateActionPairDefinition,
)
from .types import (
    BaseOperationSpec,
    DecompositionSpec,
    EffectSpec,
    JsonObject,
    SignatureFamily,
    V2ExpectedCall,
    V2QueryCase,
    V2ResourceSpec,
    public_dataclass_dict,
)


STATEFUL_SCHEME = "enterprise_state_actions"
STATEFUL_SEMANTICS_VERSION = "paired-state-transitions-v2"
STATEFUL_DOMAINS = (
    "inventory_fulfillment",
    "pricing_billing",
    "procurement_suppliers",
    "customer_support",
    "workforce_access",
    "finance_ledger",
    "compliance_audit",
    "contracts_documents",
)
CANONICAL_PARAMETERS = ("entity_id", "scope_id", "qualifier")

STATEFUL_OUTPUT_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "result": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "previous_version": {"type": "integer"},
                "new_version": {"type": "integer"},
                "changed_fields": {"type": "array", "items": {"type": "string"}},
                "effect_type": {"type": "string"},
            },
            "required": [
                "entity_id",
                "previous_version",
                "new_version",
                "changed_fields",
                "effect_type",
            ],
            "additionalProperties": False,
        },
        "postcondition": {"type": "string"},
        "fixture": {"type": "string"},
    },
    "required": ["result", "postcondition", "fixture"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class StateTransitionSpec:
    operation_id: str
    pair_id: str
    paired_operation_id: str
    effect_type: str
    target_resource_id: str
    target_field: str
    request_resource_id: str
    request_value_field: str
    observer_operation_id: str
    exact_write_set: tuple[str, ...]


@dataclass(frozen=True)
class StateActionPairSpec:
    pair_id: str
    domain: str
    signature_family: SignatureFamily
    operation_ids: tuple[str, str]
    observer_operation_ids: tuple[str, str]


@dataclass(frozen=True)
class _ActionDefinition:
    operation_id: str
    effect_phrase: str
    effect_type: str
    target_resource: str
    target_field: str
    request_value_field: str
    observer_operation_id: str


@dataclass(frozen=True)
class _PairDefinition:
    pair_id: str
    domain: str
    family: SignatureFamily
    request_resource: str
    request_fields: tuple[tuple[str, str], ...]
    actions: tuple[_ActionDefinition, _ActionDefinition]


_LEGACY_PAIR_DEFINITIONS: tuple[_PairDefinition, ...] = (
    _PairDefinition(
        "inventory-reservation-vs-count",
        "inventory_fulfillment",
        SignatureFamily.RESOLVE,
        "state_action_requests",
        (("reservation_units", "integer"), ("physical_units", "integer")),
        (
            _ActionDefinition(
                "apply_sales_reservation",
                "apply the requested active sales reservation",
                "sales_reservation",
                "active_reservations",
                "reserved_units",
                "reservation_units",
                "resolve_sellable_units",
            ),
            _ActionDefinition(
                "apply_cycle_count_adjustment",
                "apply the requested physical cycle-count adjustment",
                "cycle_count_adjustment",
                "physical_counts",
                "physical_units",
                "physical_units",
                "resolve_cycle_count_variance",
            ),
        ),
    ),
    _PairDefinition(
        "pricing-list-vs-margin",
        "pricing_billing",
        SignatureFamily.RESOLVE,
        "state_action_requests",
        (("list_price", "number"), ("minimum_margin_rate", "number")),
        (
            _ActionDefinition(
                "activate_list_price",
                "activate the requested effective list price",
                "list_price_activation",
                "price_book",
                "list_price",
                "list_price",
                "resolve_list_price",
            ),
            _ActionDefinition(
                "activate_margin_floor",
                "activate the requested minimum margin floor",
                "margin_floor_activation",
                "margin_policy",
                "minimum_margin_rate",
                "minimum_margin_rate",
                "resolve_margin_floor",
            ),
        ),
    ),
    _PairDefinition(
        "support-priority-vs-escalation",
        "customer_support",
        SignatureFamily.EVALUATE,
        "state_action_requests",
        (("priority", "string"), ("target_team", "string")),
        (
            _ActionDefinition(
                "set_case_priority",
                "set the requested support-case priority",
                "case_priority_update",
                "severity_rules",
                "priority",
                "priority",
                "resolve_case_priority",
            ),
            _ActionDefinition(
                "set_escalation_target",
                "set the requested support-case escalation target",
                "escalation_target_update",
                "escalation_policies",
                "target_team",
                "target_team",
                "resolve_escalation_target",
            ),
        ),
    ),
    _PairDefinition(
        "support-action-vs-channel",
        "customer_support",
        SignatureFamily.CALCULATE,
        "workflow_action_requests",
        (("next_action", "string"), ("allowed_channels", "array")),
        (
            _ActionDefinition(
                "set_next_case_action",
                "set the requested next support-case workflow action",
                "next_case_action_update",
                "workflow_rules",
                "next_action",
                "next_action",
                "resolve_next_action",
            ),
            _ActionDefinition(
                "set_case_contact_channel",
                "set the requested support-case contact-channel policy",
                "contact_channel_update",
                "channel_policies",
                "allowed_channels",
                "allowed_channels",
                "resolve_contact_channel",
            ),
        ),
    ),
    _PairDefinition(
        "finance-fee-vs-tax-basis",
        "finance_ledger",
        SignatureFamily.CALCULATE,
        "state_action_requests",
        (("minimum_fee", "number"), ("basis_adjustment", "number")),
        (
            _ActionDefinition(
                "post_transaction_fee",
                "post the requested transaction fee rule",
                "transaction_fee_posting",
                "fee_schedules",
                "minimum_fee",
                "minimum_fee",
                "resolve_transaction_fee",
            ),
            _ActionDefinition(
                "record_tax_basis",
                "record the requested taxable-basis adjustment",
                "tax_basis_recording",
                "tax_rules",
                "basis_adjustment",
                "basis_adjustment",
                "resolve_tax_basis",
            ),
        ),
    ),
)

# The public manifest is the only active source of benchmark pairs.  The
# private v1 declarations above are retained temporarily as a readable
# compatibility record for the original ten action IDs.
PAIR_DEFINITIONS: tuple[StateActionPairDefinition, ...] = STATE_ACTION_PAIRS


class StatefulRegistry(V2Registry):
    state_transitions: dict[str, StateTransitionSpec]
    state_action_pairs: dict[str, StateActionPairSpec]

    def __init__(
        self,
        *args: Any,
        state_transitions: dict[str, StateTransitionSpec],
        state_action_pairs: dict[str, StateActionPairSpec],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.state_transitions = state_transitions
        self.state_action_pairs = state_action_pairs

    @property
    def state_action_operation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.state_transitions))

    def to_dict(self, *, include_resources: bool = False) -> JsonObject:
        data = super().to_dict(include_resources=include_resources)
        data["scheme"] = STATEFUL_SCHEME
        data["semantics_version"] = STATEFUL_SEMANTICS_VERSION
        data["state_transitions"] = [
            public_dataclass_dict(self.state_transitions[operation_id])
            for operation_id in sorted(self.state_transitions)
        ]
        data["state_action_pairs"] = [
            public_dataclass_dict(self.state_action_pairs[pair_id])
            for pair_id in sorted(self.state_action_pairs)
        ]
        return data


def _schema(properties: dict[str, JsonObject]) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _visible_signature(
    family: SignatureFamily,
) -> tuple[JsonObject, JsonObject, JsonObject]:
    third = {
        SignatureFamily.RESOLVE: "reference_id",
        SignatureFamily.EVALUATE: "policy_id",
        SignatureFamily.CALCULATE: "value",
    }[family]
    visible = ("subject_id", "context_id", third)
    visible_schema = _schema({name: {"type": "string"} for name in visible})
    visible_to_canonical = dict(zip(visible, CANONICAL_PARAMETERS))
    canonical_to_visible = {value: key for key, value in visible_to_canonical.items()}
    return visible_schema, visible_to_canonical, canonical_to_visible


def _case_matches(row: JsonObject, arguments: JsonObject) -> bool:
    return (
        row.get("case_entity_id") == arguments["entity_id"]
        and row.get("case_scope_id") == arguments["scope_id"]
        and row.get("case_qualifier") == arguments["qualifier"]
    )


def _one_case_row(
    state: EnvironmentState,
    resource_id: str,
    arguments: JsonObject,
) -> JsonObject:
    rows = state.resources.get(resource_id)
    if not isinstance(rows, list):
        raise TypeError(f"resource {resource_id} is not a table")
    selected = [row for row in rows if _case_matches(row, arguments)]
    if len(selected) != 1:
        raise KeyError(
            f"expected one grounded row in {resource_id}, found {len(selected)}"
        )
    return selected[0]


def _fixture_label(arguments: JsonObject) -> str:
    material = "|".join(str(arguments[name]) for name in CANONICAL_PARAMETERS)
    return f"state-fixture-{hashlib.sha256(material.encode()).hexdigest()[:12]}"


def _mutation_handler(
    *,
    domain: str,
    operation_id: str,
    effect_type: str,
    target_resource_id: str,
    target_field: str,
    request_resource_id: str,
    request_value_field: str,
):
    postcondition = f"{domain}.{operation_id}.complete"

    def handler(
        state: EnvironmentState,
        entity_id: str,
        scope_id: str,
        qualifier: str,
    ) -> JsonObject:
        arguments = {
            "entity_id": entity_id,
            "scope_id": scope_id,
            "qualifier": qualifier,
        }
        request = _one_case_row(state, request_resource_id, arguments)
        target = _one_case_row(state, target_resource_id, arguments)
        previous_version = target["version"]
        target[target_field] = copy.deepcopy(request[request_value_field])
        target["version"] = previous_version + 1
        return {
            "result": {
                "entity_id": entity_id,
                "previous_version": previous_version,
                "new_version": target["version"],
                "changed_fields": [target_field],
                "effect_type": effect_type,
            },
            "postcondition": postcondition,
            "fixture": _fixture_label(arguments),
        }

    handler.__name__ = operation_id
    return handler


def _add_required_field(
    resources: dict[str, V2ResourceSpec],
    rows: JsonObject,
    *,
    resource_id: str,
    field: str,
    schema: JsonObject,
    default: Any,
) -> None:
    spec = resources[resource_id]
    updated_schema = copy.deepcopy(spec.schema)
    updated_schema["properties"][field] = copy.deepcopy(schema)
    if field not in updated_schema["required"]:
        updated_schema["required"].append(field)
    resources[resource_id] = V2ResourceSpec(
        spec.resource_id,
        spec.kind,
        updated_schema,
        spec.primary_key,
        spec.relationships,
    )
    for row in rows[resource_id]:
        row[field] = copy.deepcopy(default)


def _add_request_resource(
    resources: dict[str, V2ResourceSpec],
    rows: JsonObject,
    *,
    domain: str,
    name: str,
    fields: tuple[tuple[str, str], ...],
) -> str:
    resource_id = f"{domain}.{name}"
    properties: dict[str, JsonObject] = {
        "request_id": {"type": "string"},
        **{
            field: (
                {"type": kind, "items": {"type": "string"}}
                if kind == "array"
                else {"type": kind}
            )
            for field, kind in fields
        },
        "case_entity_id": {"type": "string"},
        "case_scope_id": {"type": "string"},
        "case_qualifier": {"type": "string"},
    }
    resources[resource_id] = V2ResourceSpec(
        resource_id,
        "table",
        _schema(properties),
        ("request_id",),
        (),
    )
    rows[resource_id] = []
    return resource_id


def _sample_arguments(pair_index: int, sample_index: int) -> JsonObject:
    return {
        "entity_id": f"STATE-{pair_index + 1:02d}-{sample_index + 1:02d}",
        "scope_id": f"STATE-CTX-{pair_index + 1:02d}",
        # A real ISO reference keeps the shared signature valid for observers
        # whose business result is a date, while entity/scope retain pair-level
        # fixture uniqueness.
        "qualifier": f"2026-07-{sample_index + 1:02d}",
    }


def _context(arguments: JsonObject) -> JsonObject:
    return {
        "case_entity_id": arguments["entity_id"],
        "case_scope_id": arguments["scope_id"],
        "case_qualifier": arguments["qualifier"],
    }


def _request_values(pair_id: str, sample_index: int) -> JsonObject:
    ordinal = sample_index + 1
    values = {
        "inventory-reservation-vs-count": {
            "reservation_units": 10 + ordinal,
            "physical_units": 80 + ordinal,
        },
        "pricing-list-vs-margin": {
            "list_price": 100.0 + ordinal,
            "minimum_margin_rate": round(0.20 + ordinal / 100, 2),
        },
        "support-priority-vs-escalation": {
            "priority": f"P{min(ordinal, 4)}",
            "target_team": f"ESCALATION-TEAM-{ordinal}",
        },
        "support-action-vs-channel": {
            "next_action": f"WORKFLOW-ACTION-{ordinal}",
            "allowed_channels": [f"CHANNEL-{ordinal}"],
        },
        "finance-fee-vs-tax-basis": {
            "minimum_fee": 5.0 + ordinal,
            "basis_adjustment": 50.0 + ordinal,
        },
    }
    return values[pair_id]


def _matching_rows(
    rows: JsonObject,
    resource_id: str,
    arguments: JsonObject,
) -> list[JsonObject]:
    return [
        row
        for row in rows[resource_id]
        if _case_matches(row, arguments)
    ]


def _requested_value(
    action: StateActionDefinition,
    current: Any,
    sample_index: int,
) -> Any:
    """Create a deterministic, type-preserving value distinct from the fixture."""

    ordinal = sample_index + 1
    if action.value_kind == "boolean":
        return not bool(current)
    if action.value_kind == "integer":
        return int(current) + 1
    if action.value_kind == "number":
        return round(float(current) + 1.0, 4)
    if action.value_kind == "array":
        return [f"REQUESTED-{action.operation_id.upper()}-{ordinal:02d}"]
    if action.value_kind == "string":
        return f"REQUESTED-{action.operation_id.upper()}-{ordinal:02d}"
    raise ValueError(
        f"{action.operation_id}: unsupported state-action value kind "
        f"{action.value_kind!r}"
    )


def _candidate_requested_values(
    action: StateActionDefinition,
    current: Any,
    sample_index: int,
) -> tuple[Any, ...]:
    primary = _requested_value(action, current, sample_index)
    if action.value_kind == "boolean":
        return (primary,)
    if action.value_kind == "integer":
        return tuple(
            dict.fromkeys(
                [primary]
                + [int(current) + delta for delta in range(2, 10)]
                + [0, 101 + sample_index]
            )
        )
    if action.value_kind == "number":
        return tuple(
            dict.fromkeys(
                [primary]
                + [round(float(current) + delta, 4) for delta in range(2, 10)]
                + [0.0, 101.5 + sample_index]
            )
        )
    if action.value_kind == "array":
        return tuple(
            [
                primary,
                [f"ALTERNATE-{action.operation_id.upper()}-{sample_index + 1:02d}"],
                [],
            ]
        )
    return tuple(
        [primary]
        + [
            f"ALTERNATE-{action.operation_id.upper()}-{sample_index + 1:02d}-{index}"
            for index in range(1, 5)
        ]
    )


def _observer_sensitive_value(
    base: V2Registry,
    rows: JsonObject,
    *,
    pair: StateActionPairDefinition,
    action: StateActionDefinition,
    arguments: JsonObject,
    sample_index: int,
) -> Any:
    resource_id = f"{pair.domain}.{action.target_resource}"
    target = _one_case_row(EnvironmentState(rows), resource_id, arguments)
    original = copy.deepcopy(target[action.target_field])
    observer = base.get_base(action.observer_operation_id)
    before = execute_base(observer, EnvironmentState(rows), arguments).output
    for candidate in _candidate_requested_values(action, original, sample_index):
        target[action.target_field] = copy.deepcopy(candidate)
        after = execute_base(observer, EnvironmentState(rows), arguments).output
        target[action.target_field] = copy.deepcopy(original)
        if before != after:
            return candidate
    raise ValueError(
        f"{action.operation_id}: no deterministic requested value changes "
        f"{action.observer_operation_id} for fixture {sample_index}"
    )


def _clone_observer_resource_row(
    rows: JsonObject,
    *,
    resource_id: str,
    source_arguments: JsonObject,
    target_arguments: JsonObject,
) -> JsonObject:
    source_rows = _matching_rows(rows, resource_id, source_arguments)
    if not source_rows:
        raise KeyError(
            f"no source row in {resource_id} for observer fixture "
            f"{source_arguments}"
        )
    cloned = copy.deepcopy(source_rows[0])
    cloned.update(_context(target_arguments))
    return cloned


def _append_generic_grounding_rows(
    base: V2Registry,
    rows: JsonObject,
    *,
    pair: StateActionPairDefinition,
    arguments: JsonObject,
    sample_index: int,
) -> None:
    """Ground both observers on one shared case without changing their handlers.

    One real row is copied from each observer's structural fixture for every
    resource it reads.  Only the case-grounding columns are replaced.  This
    produces a compact paired counterfactual state while preserving the
    resource schemas and deterministic observer implementations.
    """

    cloned_resource_ids: set[str] = set()
    observer_fixtures: dict[str, JsonObject] = {}
    for action in pair.actions:
        observer_fixture = base.fixtures[action.observer_operation_id][sample_index]
        observer_fixtures[action.operation_id] = observer_fixture["arguments"]

    # Clone target rows first so a shared read resource cannot accidentally be
    # sourced from the other observer's fixture.
    for action in pair.actions:
        resource_id = f"{pair.domain}.{action.target_resource}"
        source_arguments = observer_fixtures[action.operation_id]
        cloned = _clone_observer_resource_row(
            rows,
            resource_id=resource_id,
            source_arguments=source_arguments,
            target_arguments=arguments,
        )
        if action.operation_id == "set_document_section_text":
            # The section observer excludes literal query arguments from its
            # facts.  Grounding these identifiers makes the mutable text field
            # an explicit output fact instead of relying on identity fields.
            cloned["document_id"] = arguments["entity_id"]
            cloned["section_id"] = arguments["scope_id"]
        rows[resource_id].append(cloned)
        cloned_resource_ids.add(resource_id)

    for action in pair.actions:
        observer = base.get_base(action.observer_operation_id)
        source_arguments = observer_fixtures[action.operation_id]
        for resource_id in observer.effect.read_set:
            if resource_id in cloned_resource_ids:
                continue
            rows[resource_id].append(
                _clone_observer_resource_row(
                    rows,
                    resource_id=resource_id,
                    source_arguments=source_arguments,
                    target_arguments=arguments,
                )
            )
            cloned_resource_ids.add(resource_id)

    request_values: JsonObject = {}
    for action in pair.actions:
        request_values[action.request_value_field] = _observer_sensitive_value(
            base,
            rows,
            pair=pair,
            action=action,
            arguments=arguments,
            sample_index=sample_index,
        )
    rows[f"{pair.domain}.{pair.request_resource}"].append(
        {
            "request_id": arguments["qualifier"],
            **request_values,
            **_context(arguments),
        }
    )


def _append_grounding_rows(
    rows: JsonObject,
    *,
    pair: StateActionPairDefinition,
    arguments: JsonObject,
    sample_index: int,
) -> None:
    ctx = _context(arguments)
    request_id = arguments["qualifier"]
    rows[f"{pair.domain}.{pair.request_resource}"].append(
        {
            "request_id": request_id,
            **copy.deepcopy(_request_values(pair.pair_id, sample_index)),
            **ctx,
        }
    )

    if pair.pair_id == "inventory-reservation-vs-count":
        rows[f"{pair.domain}.inventory_snapshot"].append({
            "sku": arguments["entity_id"],
            "warehouse_id": arguments["scope_id"],
            "on_hand_units": 120,
            "book_units": 100,
            "volume_per_unit": 1.5,
            **ctx,
        })
        rows[f"{pair.domain}.active_reservations"].append({
            "reservation_id": request_id,
            "sku": arguments["entity_id"],
            "warehouse_id": arguments["scope_id"],
            "reserved_units": 5,
            "status": "active",
            "version": 0,
            **ctx,
        })
        rows[f"{pair.domain}.physical_counts"].append({
            "count_id": request_id,
            "sku": arguments["entity_id"],
            "warehouse_id": arguments["scope_id"],
            "physical_units": 100,
            "counted_at": "2026-07-01",
            "version": 0,
            **ctx,
        })
    elif pair.pair_id == "pricing-list-vs-margin":
        rows[f"{pair.domain}.price_book"].append({
            "item_id": arguments["entity_id"],
            "region": arguments["scope_id"],
            "effective_from": "2026-01-01",
            "effective_to": "2026-12-31",
            "list_price": 90.0,
            "version": 0,
            **ctx,
        })
        rows[f"{pair.domain}.item_catalog"].append({
            "item_id": arguments["entity_id"],
            "item_class": "STANDARD",
            "package_id": f"PACKAGE-{sample_index + 1}",
            **ctx,
        })
        rows[f"{pair.domain}.cost_basis"].append({
            "item_id": arguments["entity_id"],
            "region": arguments["scope_id"],
            "unit_cost": 50.0,
            **ctx,
        })
        rows[f"{pair.domain}.margin_policy"].append({
            "policy_id": request_id,
            "item_class": "STANDARD",
            "minimum_margin_rate": 0.15,
            "version": 0,
            **ctx,
        })
    elif pair.pair_id == "support-priority-vs-escalation":
        rows[f"{pair.domain}.cases"].append({
            "case_id": arguments["entity_id"],
            "customer_id": f"CUSTOMER-{sample_index + 1}",
            "issue_class": "SERVICE",
            "product_id": "PRODUCT-1",
            "created_at": "2026-07-01",
            "current_status": "open",
            "severity_signal": "medium",
            **ctx,
        })
        rows[f"{pair.domain}.severity_rules"].append({
            "rule_id": request_id,
            "issue_class": "SERVICE",
            "signal": "medium",
            "priority": "P3",
            "version": 0,
            **ctx,
        })
        rows[f"{pair.domain}.escalation_policies"].append({
            "rule_id": request_id,
            "priority": "P3",
            "current_status": "open",
            "target_team": "GENERAL-SUPPORT",
            "version": 0,
            **ctx,
        })
    elif pair.pair_id == "support-action-vs-channel":
        rows[f"{pair.domain}.cases"].append({
            "case_id": arguments["entity_id"],
            "customer_id": f"CUSTOMER-W-{sample_index + 1}",
            "issue_class": "WORKFLOW",
            "product_id": "PRODUCT-2",
            "created_at": "2026-07-01",
            "current_status": "open",
            "severity_signal": "low",
            **ctx,
        })
        rows[f"{pair.domain}.workflow_rules"].append({
            "rule_id": request_id,
            "current_status": "open",
            "issue_class": "WORKFLOW",
            "next_action": "REVIEW",
            "version": 0,
            **ctx,
        })
        rows[f"{pair.domain}.customer_preferences"].append({
            "customer_id": f"CUSTOMER-W-{sample_index + 1}",
            "preferred_channel": "EMAIL",
            "blocked_channels": [],
            **ctx,
        })
        rows[f"{pair.domain}.severity_rules"].append({
            "rule_id": f"CHANNEL-{request_id}",
            "issue_class": "WORKFLOW",
            "signal": "low",
            "priority": "P4",
            "version": 0,
            **ctx,
        })
        rows[f"{pair.domain}.channel_policies"].append({
            "issue_class": "WORKFLOW",
            "priority": "P4",
            "allowed_channels": ["EMAIL"],
            "version": 0,
            **ctx,
        })
    elif pair.pair_id == "finance-fee-vs-tax-basis":
        rows[f"{pair.domain}.transactions"].append({
            "transaction_id": arguments["entity_id"],
            "account_id": arguments["scope_id"],
            "amount": 200.0,
            "transaction_type": "PURCHASE",
            "occurred_at": "2026-07-01",
            "source_currency": "USD",
            "target_currency": "USD",
            **ctx,
        })
        rows[f"{pair.domain}.fee_schedules"].append({
            "fee_schedule_id": request_id,
            "transaction_type": "PURCHASE",
            "minimum_fee": 2.0,
            "rate": 0.02,
            "maximum_fee": 20.0,
            "version": 0,
            **ctx,
        })
        rows[f"{pair.domain}.tax_rules"].append({
            "rule_id": request_id,
            "transaction_type": "PURCHASE",
            "tax_class": "STANDARD",
            "basis_adjustment": 0.0,
            "version": 0,
            **ctx,
        })


def _prepare_mutable_resources(
    resources: dict[str, V2ResourceSpec],
    rows: JsonObject,
) -> None:
    _add_required_field(
        resources,
        rows,
        resource_id="finance_ledger.tax_rules",
        field="basis_adjustment",
        schema={"type": "number"},
        default=0.0,
    )
    _add_required_field(
        resources,
        rows,
        resource_id="compliance_audit.current_artifacts",
        field="artifact_count",
        schema={"type": "integer"},
        default=2,
    )
    mutable_resource_ids = {
        f"{pair.domain}.{action.target_resource}"
        for pair in PAIR_DEFINITIONS
        for action in pair.actions
    }
    for resource_id in sorted(mutable_resource_ids):
        _add_required_field(
            resources,
            rows,
            resource_id=resource_id,
            field="version",
            schema={"type": "integer"},
            default=0,
        )
    for pair in PAIR_DEFINITIONS:
        for action in pair.actions:
            resource_id = f"{pair.domain}.{action.target_resource}"
            actual_kind = resources[resource_id].schema["properties"][
                action.target_field
            ].get("type")
            if actual_kind != action.value_kind:
                raise ValueError(
                    f"{action.operation_id}: manifest value kind "
                    f"{action.value_kind!r} does not match "
                    f"{resource_id}.{action.target_field} ({actual_kind!r})"
                )


def build_stateful_registry(*, seed: int = 0) -> StatefulRegistry:
    base = build_v2_registry(domains=list(STATEFUL_DOMAINS), seed=seed)
    resources = copy.deepcopy(base.resources)
    rows = copy.deepcopy(base._resource_rows)
    bases = dict(base.base_operations)
    subs = dict(base.sub_operations)
    fixtures = copy.deepcopy(base.fixtures)
    operation_effects = dict(base.operation_effects)
    transitions: dict[str, StateTransitionSpec] = {}
    pairs: dict[str, StateActionPairSpec] = {}

    _prepare_mutable_resources(resources, rows)
    request_resources: set[str] = set()
    pair_arguments: dict[str, list[JsonObject]] = {}
    for pair_index, pair in enumerate(PAIR_DEFINITIONS):
        request_resource_id = f"{pair.domain}.{pair.request_resource}"
        if request_resource_id not in request_resources:
            _add_request_resource(
                resources,
                rows,
                domain=pair.domain,
                name=pair.request_resource,
                fields=pair.request_fields,
            )
            request_resources.add(request_resource_id)
        arguments_list = [
            _sample_arguments(pair_index, sample_index)
            for sample_index in range(5)
        ]
        pair_arguments[pair.pair_id] = arguments_list
        for sample_index, arguments in enumerate(arguments_list):
            if pair.legacy_grounding:
                _append_grounding_rows(
                    rows,
                    pair=pair,
                    arguments=arguments,
                    sample_index=sample_index,
                )
            else:
                _append_generic_grounding_rows(
                    base,
                    rows,
                    pair=pair,
                    arguments=arguments,
                    sample_index=sample_index,
                )

        pairs[pair.pair_id] = StateActionPairSpec(
            pair.pair_id,
            pair.domain,
            pair.family,
            tuple(action.operation_id for action in pair.actions),
            tuple(action.observer_operation_id for action in pair.actions),
        )
        for action, twin in zip(pair.actions, reversed(pair.actions)):
            target_resource_id = f"{pair.domain}.{action.target_resource}"
            visible_schema, visible_to_canonical, canonical_to_visible = _visible_signature(
                pair.family
            )
            postcondition = f"{pair.domain}.{action.operation_id}.complete"
            source = (
                f"def {action.operation_id}(state, entity_id, scope_id, qualifier):\n"
                f"    request = read_one({request_resource_id!r}, entity_id, scope_id, qualifier)\n"
                f"    return update_one({target_resource_id!r}, {action.target_field!r}, "
                f"request[{action.request_value_field!r}])"
            )
            description = (
                f"Apply one authoritative enterprise state transition to "
                f"{action.effect_phrase}. The call succeeds only after changing exactly "
                f"{target_resource_id}.{action.target_field}; it returns a normalized "
                f"state-change receipt. Postcondition: {postcondition}."
            )
            effect = EffectSpec(
                (request_resource_id, target_resource_id),
                (target_resource_id,),
                STATEFUL_OUTPUT_SCHEMA,
                (postcondition,),
                f"{pair.domain}.state_action.{pair.pair_id}",
            )
            bases[action.operation_id] = BaseOperationSpec(
                action.operation_id,
                pair.domain,
                pair.family,
                CANONICAL_PARAMETERS,
                _schema({name: {"type": "string"} for name in CANONICAL_PARAMETERS}),
                visible_schema,
                visible_to_canonical,
                canonical_to_visible,
                description,
                effect,
                _mutation_handler(
                    domain=pair.domain,
                    operation_id=action.operation_id,
                    effect_type=action.effect_type,
                    target_resource_id=target_resource_id,
                    target_field=action.target_field,
                    request_resource_id=request_resource_id,
                    request_value_field=action.request_value_field,
                ),
                source,
                DecompositionSpec((), (), ""),
                (f"{action.operation_id}:state-transition",),
            )
            transitions[action.operation_id] = StateTransitionSpec(
                action.operation_id,
                pair.pair_id,
                twin.operation_id,
                action.effect_type,
                target_resource_id,
                action.target_field,
                request_resource_id,
                action.request_value_field,
                action.observer_operation_id,
                (target_resource_id,),
            )
            operation_effects[action.operation_id] = action.effect_phrase

    state_template = EnvironmentState(rows)
    for operation_id, transition in transitions.items():
        operation = bases[operation_id]
        operation_fixtures = []
        for arguments in pair_arguments[transition.pair_id]:
            state = state_template.clone()
            before = state.digest()
            result = execute_base(operation, state, arguments)
            operation_fixtures.append({
                "arguments": copy.deepcopy(arguments),
                "output": copy.deepcopy(result.output),
                "initial_state_digest": before,
                "expected_state_digest": state.digest(),
                "changed_resources": list(result.changed_resources),
            })
        fixtures[operation_id] = operation_fixtures

    return StatefulRegistry(
        STATEFUL_SCHEME,
        seed,
        STATEFUL_DOMAINS,
        bases,
        subs,
        resources,
        rows,
        fixtures,
        operation_effects,
        state_transitions=transitions,
        state_action_pairs=pairs,
    )


STATEFUL_QUERY_TEMPLATES = (
    "What business-state change should be completed for subject {entity_id} in context {scope_id} under request {qualifier} to {effect}?",
    "Please {effect} for subject {entity_id} in context {scope_id} under request {qualifier}, and verify the resulting state.",
    "I need you to {effect} for subject {entity_id} in context {scope_id} under request {qualifier}, committing only the intended state transition.",
    "Determine how to {effect} for subject {entity_id} in context {scope_id} under request {qualifier}.",
    "Review request {qualifier} for subject {entity_id} in context {scope_id}, then {effect} and verify the resulting state.",
)


def generate_stateful_queries(
    registry: StatefulRegistry,
    *,
    queries_per_action: int = 5,
) -> tuple[V2QueryCase, ...]:
    cases = []
    for operation_id in registry.state_action_operation_ids:
        operation = registry.get_base(operation_id)
        transition = registry.state_transitions[operation_id]
        fixtures = registry.fixtures[operation_id]
        for index in range(queries_per_action):
            fixture = fixtures[index % len(fixtures)]
            arguments = fixture["arguments"]
            template_id = index % len(STATEFUL_QUERY_TEMPLATES)
            prompt = STATEFUL_QUERY_TEMPLATES[template_id].format(
                **arguments,
                effect=registry.operation_effects[operation_id],
            )
            cases.append(V2QueryCase(
                f"stateful-{operation.domain}-{operation_id}-{index:02d}",
                operation.domain,
                prompt,
                operation.effect.postconditions,
                V2ExpectedCall(
                    operation_id,
                    copy.deepcopy(arguments),
                    canonical_to_visible(operation, arguments),
                ),
                {
                    "scheme": STATEFUL_SCHEME,
                    "semantics_version": STATEFUL_SEMANTICS_VERSION,
                    "state_action": True,
                    "pair_id": transition.pair_id,
                    "paired_operation_id": transition.paired_operation_id,
                    "observer_operation_id": transition.observer_operation_id,
                    "target_resource_id": transition.target_resource_id,
                    "target_field": transition.target_field,
                    "query_template_id": template_id,
                    "pair_case_index": index,
                },
            ))
    return tuple(cases)


def build_stateful_candidate_audit(
    registry: StatefulRegistry,
) -> JsonObject:
    """Return executable evidence for all 70 post-v1 observer candidates."""

    definitions = {
        action.operation_id: (pair, action)
        for pair in PAIR_DEFINITIONS[5:]
        for action in pair.actions
    }
    selected_records: list[JsonObject] = []
    for operation_id in sorted(definitions):
        pair, action = definitions[operation_id]
        transition = registry.state_transitions[operation_id]
        evidence: list[JsonObject] = []
        for fixture_index, fixture in enumerate(registry.fixtures[operation_id]):
            arguments = fixture["arguments"]
            before_state = registry.new_state()
            before = execute_base(
                registry.get_base(action.observer_operation_id),
                before_state,
                arguments,
            ).output
            after_state = registry.new_state()
            mutation = execute_base(
                registry.get_base(operation_id),
                after_state,
                arguments,
            )
            after = execute_base(
                registry.get_base(action.observer_operation_id),
                after_state,
                arguments,
            ).output
            changed = before != after
            evidence.append(
                {
                    "fixture_index": fixture_index,
                    "arguments": copy.deepcopy(arguments),
                    "observer_result_before": copy.deepcopy(before["result"]),
                    "observer_result_after": copy.deepcopy(after["result"]),
                    "observer_output_changed": changed,
                    "changed_resources": list(mutation.changed_resources),
                }
            )
        if not all(item["observer_output_changed"] for item in evidence):
            raise ValueError(
                f"{operation_id}: observer sensitivity did not hold for all fixtures"
            )
        selected_records.append(
            {
                "observer_operation_id": action.observer_operation_id,
                "decision": "selected",
                "pair_id": pair.pair_id,
                "state_action_operation_id": operation_id,
                "target_resource_id": transition.target_resource_id,
                "target_field": transition.target_field,
                "criteria": {
                    "single_resource_write": True,
                    "non_identity_business_field": True,
                    "observer_output_changed_5_of_5": True,
                    "observer_handler_modified": False,
                },
                "evidence": evidence,
            }
        )

    rejected_records = [
        {
            **item,
            "decision": "rejected",
            "criteria": {
                "stable_business_field_found": False,
                "observer_handler_modified": False,
            },
        }
        for item in REJECTED_OBSERVERS
    ]
    legacy_observers = {
        action.observer_operation_id
        for pair in PAIR_DEFINITIONS[:5]
        for action in pair.actions
    }
    remaining_observers = {
        operation_id
        for operation_id in registry.base_operations
        if operation_id.startswith("resolve_")
    } - legacy_observers
    audited_observers = {
        item["observer_operation_id"]
        for item in (*selected_records, *rejected_records)
    }
    if remaining_observers != audited_observers:
        raise ValueError(
            "candidate audit does not exactly cover the 70 remaining observers; "
            f"missing={sorted(remaining_observers - audited_observers)}, "
            f"extra={sorted(audited_observers - remaining_observers)}"
        )
    return {
        "schema_version": "stateful-candidate-audit-v1",
        "semantics_version": STATEFUL_SEMANTICS_VERSION,
        "scope": {
            "structural_observers": 80,
            "preexisting_stateful_observers": 10,
            "remaining_candidates": 70,
            "selected": len(selected_records),
            "rejected": len(rejected_records),
        },
        "selection_standard": {
            "required": [
                "one real authoritative resource is mutated",
                "the field represents business state rather than identity passthrough",
                "the paired action has the same visible signature and a different postcondition",
                "the declared observer output changes on all five fixtures",
                "the original observer handler is not rewritten",
            ],
            "validator": "deterministic program; no LLM judge",
        },
        "records": sorted(
            [*selected_records, *rejected_records],
            key=lambda item: item["observer_operation_id"],
        ),
    }
