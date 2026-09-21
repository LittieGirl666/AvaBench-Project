"""Authoritative manifest for the paired state-transition benchmark.

The structural benchmark remains defined by ``enterprise_operations``.  This
module only declares the orthogonal stateful actions: each action changes one
real resource field and has one read-only observer whose output must change.
Pairs deliberately share a visible signature while producing different
business postconditions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import SignatureFamily


@dataclass(frozen=True)
class StateActionDefinition:
    operation_id: str
    effect_phrase: str
    effect_type: str
    target_resource: str
    target_field: str
    request_value_field: str
    observer_operation_id: str
    value_kind: str
    semantic_anchors: tuple[str, ...]


@dataclass(frozen=True)
class StateActionPairDefinition:
    pair_id: str
    domain: str
    family: SignatureFamily
    request_resource: str
    actions: tuple[StateActionDefinition, StateActionDefinition]
    legacy_grounding: bool = False

    @property
    def request_fields(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (action.request_value_field, action.value_kind)
            for action in self.actions
        )


def _action(
    operation_id: str,
    effect_phrase: str,
    effect_type: str,
    target_resource: str,
    target_field: str,
    request_value_field: str,
    observer_operation_id: str,
    value_kind: str,
    *semantic_anchors: str,
) -> StateActionDefinition:
    return StateActionDefinition(
        operation_id,
        effect_phrase,
        effect_type,
        target_resource,
        target_field,
        request_value_field,
        observer_operation_id,
        value_kind,
        tuple(semantic_anchors),
    )


def _pair(
    pair_id: str,
    domain: str,
    family: SignatureFamily,
    actions: tuple[StateActionDefinition, StateActionDefinition],
    *,
    request_resource: str | None = None,
    legacy_grounding: bool = False,
) -> StateActionPairDefinition:
    return StateActionPairDefinition(
        pair_id,
        domain,
        family,
        request_resource or f"{pair_id.replace('-', '_')}_requests",
        actions,
        legacy_grounding,
    )


# The first five pairs are the original paired-state-actions-v1 actions.  Their
# operation IDs, target resources, fields, observers, and effect semantics are
# intentionally unchanged.
STATE_ACTION_PAIRS: tuple[StateActionPairDefinition, ...] = (
    _pair(
        "inventory-reservation-vs-count",
        "inventory_fulfillment",
        SignatureFamily.RESOLVE,
        (
            _action(
                "apply_sales_reservation",
                "apply the requested active sales reservation",
                "sales_reservation",
                "active_reservations",
                "reserved_units",
                "reservation_units",
                "resolve_sellable_units",
                "integer",
                "sales reservation",
            ),
            _action(
                "apply_cycle_count_adjustment",
                "apply the requested physical cycle-count adjustment",
                "cycle_count_adjustment",
                "physical_counts",
                "physical_units",
                "physical_units",
                "resolve_cycle_count_variance",
                "integer",
                "cycle-count adjustment",
            ),
        ),
        request_resource="state_action_requests",
        legacy_grounding=True,
    ),
    _pair(
        "pricing-list-vs-margin",
        "pricing_billing",
        SignatureFamily.RESOLVE,
        (
            _action(
                "activate_list_price",
                "activate the requested effective list price",
                "list_price_activation",
                "price_book",
                "list_price",
                "list_price",
                "resolve_list_price",
                "number",
                "list price",
            ),
            _action(
                "activate_margin_floor",
                "activate the requested minimum margin floor",
                "margin_floor_activation",
                "margin_policy",
                "minimum_margin_rate",
                "minimum_margin_rate",
                "resolve_margin_floor",
                "number",
                "margin floor",
            ),
        ),
        request_resource="state_action_requests",
        legacy_grounding=True,
    ),
    _pair(
        "support-priority-vs-escalation",
        "customer_support",
        SignatureFamily.EVALUATE,
        (
            _action(
                "set_case_priority",
                "set the requested support-case priority",
                "case_priority_update",
                "severity_rules",
                "priority",
                "priority",
                "resolve_case_priority",
                "string",
                "case priority",
            ),
            _action(
                "set_escalation_target",
                "set the requested support-case escalation target",
                "escalation_target_update",
                "escalation_policies",
                "target_team",
                "target_team",
                "resolve_escalation_target",
                "string",
                "escalation target",
            ),
        ),
        request_resource="state_action_requests",
        legacy_grounding=True,
    ),
    _pair(
        "support-action-vs-channel",
        "customer_support",
        SignatureFamily.CALCULATE,
        (
            _action(
                "set_next_case_action",
                "set the requested next support-case workflow action",
                "next_case_action_update",
                "workflow_rules",
                "next_action",
                "next_action",
                "resolve_next_action",
                "string",
                "workflow action",
            ),
            _action(
                "set_case_contact_channel",
                "set the requested support-case contact-channel policy",
                "contact_channel_update",
                "channel_policies",
                "allowed_channels",
                "allowed_channels",
                "resolve_contact_channel",
                "array",
                "contact-channel policy",
            ),
        ),
        request_resource="workflow_action_requests",
        legacy_grounding=True,
    ),
    _pair(
        "finance-fee-vs-tax-basis",
        "finance_ledger",
        SignatureFamily.CALCULATE,
        (
            _action(
                "post_transaction_fee",
                "post the requested transaction fee rule",
                "transaction_fee_posting",
                "fee_schedules",
                "minimum_fee",
                "minimum_fee",
                "resolve_transaction_fee",
                "number",
                "transaction fee",
            ),
            _action(
                "record_tax_basis",
                "record the requested taxable-basis adjustment",
                "tax_basis_recording",
                "tax_rules",
                "basis_adjustment",
                "basis_adjustment",
                "resolve_tax_basis",
                "number",
                "taxable-basis adjustment",
            ),
        ),
        request_resource="state_action_requests",
        legacy_grounding=True,
    ),

    # Inventory and fulfillment.
    _pair(
        "inventory-policy-floor-vs-cover",
        "inventory_fulfillment",
        SignatureFamily.RESOLVE,
        (
            _action("set_safety_stock_floor", "set the requested safety-stock minimum", "safety_stock_floor_update", "safety_stock_policy", "minimum_units", "safety_stock_minimum", "resolve_transferable_units", "integer", "safety-stock minimum"),
            _action("set_reorder_target_cover", "set the requested replenishment target cover", "reorder_target_cover_update", "reorder_policy", "target_cover_units", "reorder_target_cover", "resolve_reorder_quantity", "integer", "target cover"),
        ),
    ),
    _pair(
        "inventory-priority-vs-pick",
        "inventory_fulfillment",
        SignatureFamily.EVALUATE,
        (
            _action("set_backorder_customer_tier", "set the requested backorder customer tier", "backorder_customer_tier_update", "open_orders", "customer_tier", "backorder_customer_tier", "resolve_backorder_priority", "string", "customer tier"),
            _action("set_bin_pick_sequence", "set the requested warehouse-bin pick sequence", "bin_pick_sequence_update", "bin_inventory", "pick_sequence", "bin_pick_sequence", "resolve_pick_location", "integer", "pick sequence"),
        ),
    ),
    _pair(
        "inventory-source-cost-vs-expiry",
        "inventory_fulfillment",
        SignatureFamily.CALCULATE,
        (
            _action("set_source_transfer_cost", "set the requested source-warehouse transfer cost", "source_transfer_cost_update", "source_inventory", "transfer_cost", "source_transfer_cost", "resolve_replenishment_source", "number", "transfer cost"),
            _action("set_expiry_warning_window", "set the requested lot-expiry warning window", "expiry_warning_window_update", "expiry_policy", "warning_days", "expiry_warning_days", "resolve_expiry_risk", "integer", "warning window"),
        ),
    ),
    _pair(
        "inventory-stock-vs-capacity",
        "inventory_fulfillment",
        SignatureFamily.RESOLVE,
        (
            _action("set_fulfillment_on_hand_units", "set the requested fulfillment on-hand quantity", "fulfillment_on_hand_update", "inventory_snapshot", "on_hand_units", "fulfillment_on_hand_units", "resolve_fulfillment_feasibility", "integer", "on-hand quantity"),
            _action("set_warehouse_capacity_volume", "set the requested warehouse capacity volume", "warehouse_capacity_update", "location_capacity", "capacity_volume", "warehouse_capacity_volume", "resolve_capacity_utilization", "number", "capacity volume"),
        ),
    ),

    # Pricing and billing.
    _pair(
        "pricing-promotion-vs-contract",
        "pricing_billing",
        SignatureFamily.EVALUATE,
        (
            _action("set_promotion_discount_value", "set the requested promotional discount value", "promotion_discount_update", "promotions", "discount_value", "promotion_discount_value", "resolve_promotional_price", "number", "promotional discount"),
            _action("set_contract_adjustment_value", "set the requested customer-contract adjustment", "contract_adjustment_update", "customer_contracts", "adjustment_value", "contract_adjustment_value", "resolve_contract_price", "number", "contract adjustment"),
        ),
    ),
    _pair(
        "pricing-tax-vs-shipping",
        "pricing_billing",
        SignatureFamily.CALCULATE,
        (
            _action("set_tax_rate", "set the requested transaction tax rate", "tax_rate_update", "tax_rules", "rate", "tax_rate", "resolve_tax_amount", "number", "tax rate"),
            _action("set_shipping_minimum_charge", "set the requested minimum shipping charge", "shipping_minimum_charge_update", "shipping_rate_cards", "minimum_charge", "shipping_minimum_charge", "resolve_shipping_charge", "number", "shipping charge"),
        ),
    ),
    _pair(
        "pricing-fee-vs-discount",
        "pricing_billing",
        SignatureFamily.RESOLVE,
        (
            _action("set_service_fee_value", "set the requested service-fee value", "service_fee_update", "fee_schedules", "fee_value", "service_fee_value", "resolve_service_fee", "number", "service-fee value"),
            _action("set_order_discount_value", "set the requested order-discount value", "order_discount_update", "discount_rules", "discount_value", "order_discount_value", "resolve_order_total", "number", "order discount"),
        ),
    ),
    _pair(
        "pricing-credit-vs-fx",
        "pricing_billing",
        SignatureFamily.EVALUATE,
        (
            _action("set_credit_adjustment_value", "set the requested invoice-credit adjustment", "credit_adjustment_update", "adjustment_rules", "adjustment_value", "credit_adjustment_value", "resolve_credit_adjustment", "number", "credit adjustment"),
            _action("set_currency_exchange_rate", "set the requested currency exchange rate", "currency_exchange_rate_update", "exchange_rates", "rate", "currency_exchange_rate", "resolve_currency_amount", "number", "exchange rate"),
        ),
    ),

    # Procurement and suppliers.
    _pair(
        "procurement-certification-vs-quote",
        "procurement_suppliers",
        SignatureFamily.CALCULATE,
        (
            _action("set_supplier_certification_level", "set the requested supplier certification level", "supplier_certification_update", "supplier_profiles", "certification_level", "supplier_certification_level", "resolve_supplier_eligibility", "integer", "certification level"),
            _action("assign_quote_supplier", "assign the requested supplier to the quote", "quote_supplier_assignment", "supplier_quotes", "supplier_id", "quote_supplier_id", "resolve_quote_rank", "string", "supplier"),
        ),
    ),
    _pair(
        "procurement-approval-vs-transit",
        "procurement_suppliers",
        SignatureFamily.RESOLVE,
        (
            _action("set_purchase_approval_threshold", "set the requested purchase-approval threshold", "purchase_approval_threshold_update", "approval_policies", "amount_threshold", "purchase_approval_threshold", "resolve_purchase_limit", "number", "approval threshold"),
            _action("set_supplier_transit_days", "set the requested supplier-lane transit duration", "supplier_transit_days_update", "supplier_lanes", "transit_business_days", "supplier_transit_days", "resolve_lead_time", "integer", "transit duration"),
        ),
    ),
    _pair(
        "procurement-contract-vs-receipt",
        "procurement_suppliers",
        SignatureFamily.EVALUATE,
        (
            _action("set_contract_allowed_price", "set the requested contract-allowed purchase price", "contract_allowed_price_update", "contracts", "allowed_price", "contract_allowed_price", "resolve_contract_compliance", "number", "allowed purchase price"),
            _action("set_goods_receipt_quantity", "set the requested goods-receipt quantity", "goods_receipt_quantity_update", "goods_receipts", "received_quantity", "goods_receipt_quantity", "resolve_receipt_variance", "integer", "receipt quantity"),
        ),
    ),
    _pair(
        "procurement-performance-vs-expedite",
        "procurement_suppliers",
        SignatureFamily.CALCULATE,
        (
            _action("set_supplier_performance_value", "set the requested supplier-performance value", "supplier_performance_update", "supplier_performance", "value", "supplier_performance_value", "resolve_vendor_score", "number", "performance value"),
            _action("set_expedited_delivery_days", "set the requested expedited-delivery duration", "expedited_delivery_update", "supplier_options", "expedited_days", "expedited_delivery_days", "resolve_expedite_decision", "integer", "expedited-delivery duration"),
        ),
    ),

    # Customer support.
    _pair(
        "support-owner-vs-sla",
        "customer_support",
        SignatureFamily.RESOLVE,
        (
            _action("set_case_routing_owner", "set the requested support-case routing owner", "case_routing_owner_update", "routing_policies", "owner_team", "case_routing_owner", "resolve_case_owner", "string", "routing owner"),
            _action("set_sla_duration_minutes", "set the requested service-level duration", "sla_duration_update", "sla_rules", "duration_minutes", "sla_duration_minutes", "resolve_sla_deadline", "integer", "service-level duration"),
        ),
    ),
    _pair(
        "support-refund-vs-duplicate",
        "customer_support",
        SignatureFamily.EVALUATE,
        (
            _action("set_refund_maximum_amount", "set the requested maximum refundable amount", "refund_maximum_update", "refund_policies", "maximum_amount", "refund_maximum_amount", "resolve_refund_eligibility", "number", "refundable amount"),
            _action("set_duplicate_similarity_score", "set the requested duplicate-case similarity score", "duplicate_similarity_update", "case_similarity_index", "similarity_score", "duplicate_similarity_score", "resolve_duplicate_case", "number", "similarity score"),
        ),
    ),

    # Workforce and access.
    _pair(
        "workforce-access-vs-role",
        "workforce_access",
        SignatureFamily.CALCULATE,
        (
            _action("set_access_minimum_clearance", "set the requested minimum resource-clearance level", "access_clearance_update", "resource_policies", "minimum_clearance", "access_minimum_clearance", "resolve_access_decision", "integer", "resource-clearance"),
            _action("set_role_assignment_priority", "set the requested workforce-role priority", "role_assignment_priority_update", "role_matrix", "priority", "role_assignment_priority", "resolve_role_assignment", "integer", "workforce-role priority"),
        ),
    ),
    _pair(
        "workforce-shift-vs-leave",
        "workforce_access",
        SignatureFamily.RESOLVE,
        (
            _action("set_shift_minimum_rest_hours", "set the requested minimum rest period for the shift", "shift_rest_update", "shift_rules", "minimum_rest_hours", "shift_minimum_rest_hours", "resolve_shift_eligibility", "number", "minimum rest period"),
            _action("set_leave_ledger_units", "set the requested employee leave-ledger units", "leave_ledger_update", "leave_ledger", "units", "leave_ledger_units", "resolve_leave_balance", "number", "leave-ledger units"),
        ),
    ),
    _pair(
        "workforce-overtime-vs-delegation",
        "workforce_access",
        SignatureFamily.EVALUATE,
        (
            _action("set_overtime_approval_threshold", "set the requested overtime-approval threshold", "overtime_threshold_update", "overtime_policy", "approval_threshold", "overtime_approval_threshold", "resolve_overtime_approval", "number", "overtime-approval threshold"),
            _action("set_delegation_allowed_actions", "set the requested delegation action scope", "delegation_scope_update", "scope_policy", "allowed_actions", "delegation_allowed_actions", "resolve_delegation_scope", "array", "delegation action scope"),
        ),
    ),
    _pair(
        "workforce-oncall-vs-badge",
        "workforce_access",
        SignatureFamily.CALCULATE,
        (
            _action("set_oncall_capacity", "set the requested on-call roster capacity", "oncall_capacity_update", "roster", "capacity", "oncall_capacity", "resolve_oncall_assignment", "integer", "roster capacity"),
            _action("set_badge_validity_days", "set the requested employee-badge validity period", "badge_validity_update", "validity_policy", "validity_days", "badge_validity_days", "resolve_badge_expiry", "integer", "badge validity"),
        ),
    ),

    # Finance and ledger.
    _pair(
        "finance-ledger-vs-hold",
        "finance_ledger",
        SignatureFamily.RESOLVE,
        (
            _action("set_ledger_entry_amount", "set the requested posted-ledger amount", "ledger_entry_amount_update", "ledger_entries", "amount", "ledger_entry_amount", "resolve_book_balance", "number", "ledger amount"),
            _action("set_account_hold_amount", "set the requested account-hold amount", "account_hold_amount_update", "holds", "amount", "account_hold_amount", "resolve_available_balance", "number", "hold amount"),
        ),
    ),
    _pair(
        "finance-statement-vs-budget",
        "finance_ledger",
        SignatureFamily.EVALUATE,
        (
            _action("set_statement_closing_total", "set the requested statement closing total", "statement_closing_update", "statements", "closing_total", "statement_closing_total", "resolve_reconciliation_status", "number", "closing total"),
            _action("set_budget_planned_amount", "set the requested planned budget amount", "budget_planned_update", "budgets", "planned_amount", "budget_planned_amount", "resolve_budget_variance", "number", "planned budget"),
        ),
    ),
    _pair(
        "finance-fx-vs-payment",
        "finance_ledger",
        SignatureFamily.CALCULATE,
        (
            _action("set_fx_exchange_rate", "set the requested foreign-exchange rate", "fx_rate_update", "exchange_rates", "rate", "fx_exchange_rate", "resolve_fx_settlement", "number", "foreign-exchange rate"),
            _action("set_payment_status_code", "set the requested payment status code", "payment_status_update", "payment_events", "status_code", "payment_status_code", "resolve_payment_status", "string", "payment status"),
        ),
    ),
    _pair(
        "finance-journal-vs-audit-event",
        "finance_ledger",
        SignatureFamily.RESOLVE,
        (
            _action("set_journal_debit_amount", "set the requested journal debit amount", "journal_debit_update", "journals", "debit", "journal_debit_amount", "resolve_journal_total", "number", "journal debit"),
            _action("set_audit_event_reference", "set the requested account-audit event reference", "audit_event_reference_update", "account_events", "event_id", "audit_event_reference", "resolve_audit_flag", "string", "audit event reference"),
        ),
    ),

    # Compliance and audit.
    _pair(
        "compliance-control-vs-exception",
        "compliance_audit",
        SignatureFamily.EVALUATE,
        (
            _action("set_control_evidence_status", "set the requested control-evidence status", "control_evidence_status_update", "control_evidence", "status", "control_evidence_status", "resolve_control_status", "string", "evidence status"),
            _action("set_exception_extension_days", "set the requested policy-exception extension", "exception_extension_update", "exception_extensions", "days", "exception_extension_days", "resolve_exception_expiry", "integer", "exception extension"),
        ),
    ),
    _pair(
        "compliance-risk-vs-sample",
        "compliance_audit",
        SignatureFamily.CALCULATE,
        (
            _action("set_risk_matrix_score", "set the requested compliance-risk matrix score", "risk_matrix_score_update", "risk_scoring_matrix", "score", "risk_matrix_score", "resolve_risk_rating", "number", "risk matrix score"),
            _action("replace_audit_population_record", "replace the requested audit-population record reference", "audit_population_record_update", "audit_population", "record_id", "audit_population_record", "resolve_audit_sample", "string", "population record"),
        ),
    ),
    _pair(
        "compliance-role-vs-retention",
        "compliance_audit",
        SignatureFamily.RESOLVE,
        (
            _action("set_assigned_role", "set the requested compliance-role assignment", "assigned_role_update", "role_assignments", "role_id", "assigned_role_id", "resolve_segregation_conflict", "string", "role assignment"),
            _action("set_retention_years", "set the requested record-retention period", "retention_years_update", "retention_rules", "years", "retention_years", "resolve_retention_deadline", "integer", "retention period"),
        ),
    ),
    _pair(
        "compliance-owner-vs-filing",
        "compliance_audit",
        SignatureFamily.EVALUATE,
        (
            _action("set_remediation_owner", "set the requested finding-remediation owner", "remediation_owner_update", "ownership_matrix", "owner_id", "remediation_owner_id", "resolve_remediation_owner", "string", "remediation owner"),
            _action("set_filing_artifact_count", "set the requested regulatory-filing artifact count", "filing_artifact_count_update", "current_artifacts", "artifact_count", "filing_artifact_count", "resolve_filing_readiness", "integer", "artifact count"),
        ),
    ),

    # Contracts and documents.
    _pair(
        "contracts-clause-vs-obligation",
        "contracts_documents",
        SignatureFamily.CALCULATE,
        (
            _action("set_effective_clause_text", "set the requested effective-clause text", "effective_clause_text_update", "clauses", "text", "effective_clause_text", "resolve_effective_clause", "string", "clause text"),
            _action("set_obligation_party", "set the requested contract-obligation party", "obligation_party_update", "party_mapping", "party_id", "obligation_party_id", "resolve_obligation_owner", "string", "obligation party"),
        ),
    ),
    _pair(
        "contracts-deadline-vs-version",
        "contracts_documents",
        SignatureFamily.RESOLVE,
        (
            _action("set_contract_deadline_offset", "set the requested contractual-deadline offset", "contract_deadline_offset_update", "calendar_rules", "offset_days", "contract_deadline_offset", "resolve_contract_deadline", "integer", "deadline offset"),
            _action("set_revision_approval", "set the requested document-revision approval state", "revision_approval_update", "revision_history", "approved", "revision_approval", "resolve_latest_version", "boolean", "revision approval"),
        ),
    ),
    _pair(
        "contracts-citation-vs-signature",
        "contracts_documents",
        SignatureFamily.EVALUATE,
        (
            _action("set_citation_target", "set the requested document-citation target", "citation_target_update", "citation_index", "target_id", "citation_target_id", "resolve_citation_target", "string", "citation target"),
            _action("set_signature_event_type", "set the requested contract-signature event type", "signature_event_type_update", "signature_events", "event_type", "signature_event_type", "resolve_signature_status", "string", "signature event"),
        ),
    ),
    _pair(
        "contracts-retention-vs-section",
        "contracts_documents",
        SignatureFamily.CALCULATE,
        (
            _action("set_retention_classification", "set the requested document-retention classification", "retention_classification_update", "classification_rules", "retention_class", "retention_classification", "resolve_retention_class", "string", "retention classification"),
            _action("set_document_section_text", "set the requested contract-section text", "document_section_text_update", "document_sections", "text", "document_section_text", "resolve_section_summary", "string", "section text"),
        ),
    ),
)


REJECTED_OBSERVERS: tuple[dict[str, str], ...] = (
    {"observer_operation_id": "resolve_order_split", "reason": "Meaningful supplier capacity does not affect the list result; only item identity does."},
    {"observer_operation_id": "resolve_return_authorization", "reason": "Policy-field sensitivity is unstable and receipt quantity duplicates the receipt-variance action."},
    {"observer_operation_id": "resolve_service_entitlement", "reason": "No non-key mutable field changes the observer output across all five fixtures."},
    {"observer_operation_id": "resolve_case_closure", "reason": "No natural mutable field changes the observer output across all five fixtures."},
    {"observer_operation_id": "resolve_training_requirement", "reason": "Only employee status is stable, which does not represent the training matrix and overlaps offboarding."},
    {"observer_operation_id": "resolve_offboarding_actions", "reason": "The action code is insensitive; only the shared employee-status field changes the result."},
    {"observer_operation_id": "resolve_evidence_completeness", "reason": "No stable business field changes the observer output across all five fixtures."},
    {"observer_operation_id": "resolve_policy_applicability", "reason": "No stable policy field changes the observer output across all five fixtures."},
    {"observer_operation_id": "resolve_document_access", "reason": "The permission field is ignored by the Boolean observer."},
    {"observer_operation_id": "resolve_amendment_impact", "reason": "Replacement text is ignored; only party identity changes the output."},
)


def selected_observer_ids() -> tuple[str, ...]:
    return tuple(
        action.observer_operation_id
        for pair in STATE_ACTION_PAIRS[5:]
        for action in pair.actions
    )


def validate_stateful_manifest() -> None:
    pair_ids = [pair.pair_id for pair in STATE_ACTION_PAIRS]
    action_ids = [
        action.operation_id
        for pair in STATE_ACTION_PAIRS
        for action in pair.actions
    ]
    observers = selected_observer_ids()
    rejected = tuple(item["observer_operation_id"] for item in REJECTED_OBSERVERS)
    if len(pair_ids) != 35 or len(set(pair_ids)) != 35:
        raise ValueError("stateful manifest must contain 35 unique pairs")
    if len(action_ids) != 70 or len(set(action_ids)) != 70:
        raise ValueError("stateful manifest must contain 70 unique actions")
    if len(observers) != 60 or len(set(observers)) != 60:
        raise ValueError("expanded manifest must select 60 unique remaining observers")
    if len(rejected) != 10 or len(set(rejected)) != 10:
        raise ValueError("expanded manifest must reject 10 unique remaining observers")
    if set(observers) & set(rejected):
        raise ValueError("selected and rejected observer sets overlap")
    normalize = lambda value: " ".join(value.casefold().replace("-", " ").split())
    for pair in STATE_ACTION_PAIRS:
        if len(pair.actions) != 2:
            raise ValueError(f"{pair.pair_id}: a pair must contain exactly two actions")
        if any(
            normalize(anchor) not in normalize(action.effect_phrase)
            for action in pair.actions
            for anchor in action.semantic_anchors
        ):
            raise ValueError(f"{pair.pair_id}: semantic anchors must occur in effect phrases")


validate_stateful_manifest()
