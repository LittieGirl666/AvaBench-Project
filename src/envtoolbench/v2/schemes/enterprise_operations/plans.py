"""Explicit resource dependencies for every canonical Scheme A operation.

The manifest fixes names and decomposition labels.  This module fixes the
executable data dependencies; keeping them explicit prevents the old fuzzy
label-to-table matcher from silently assigning an unrelated resource.
"""

from __future__ import annotations


STEP_RESOURCES: dict[str, tuple[tuple[str, ...], ...]] = {
    # Inventory and fulfillment.
    "resolve_sellable_units": (("inventory_snapshot",), ("active_reservations",), ()),
    "resolve_transferable_units": (("inventory_snapshot",), ("active_reservations",), ("safety_stock_policy",)),
    "resolve_reorder_quantity": (("demand_forecast",), ("reorder_policy",), ()),
    "resolve_backorder_priority": (("open_orders",), ("priority_rules",), ("open_orders", "order_lines")),
    "resolve_pick_location": (("bin_inventory",), ("bin_inventory",), ("bin_inventory",)),
    "resolve_replenishment_source": (("source_inventory",), ("source_inventory",), ("source_inventory",)),
    "resolve_cycle_count_variance": (("physical_counts",), ("inventory_snapshot",), ()),
    "resolve_expiry_risk": (("lots",), ("expiry_policy",), ()),
    "resolve_fulfillment_feasibility": (("order_lines",), ("inventory_snapshot", "active_reservations"), ("order_lines",)),
    "resolve_capacity_utilization": (("location_capacity",), ("inventory_snapshot",), ()),

    # Pricing and billing.
    "resolve_list_price": (("price_book",), ()),
    "resolve_promotional_price": (("price_book",), ("promotions",), ()),
    "resolve_contract_price": (("customer_contracts",), ("price_book",), ("customer_contracts",)),
    "resolve_tax_amount": (("tax_rules",), ("item_catalog",), ("tax_rules",)),
    "resolve_shipping_charge": (("shipping_rate_cards",), ("packages",), ("shipping_rate_cards",)),
    "resolve_service_fee": (("fee_schedules",), ("fee_context", "fee_schedules")),
    "resolve_margin_floor": (("cost_basis",), ("margin_policy", "item_catalog"), ()),
    "resolve_order_total": (("order_lines", "price_book"), ("order_headers", "discount_rules"), ("tax_rules", "fee_schedules")),
    "resolve_credit_adjustment": (("invoice_state",), ("adjustment_rules",), ()),
    "resolve_currency_amount": (("currency_requests", "exchange_rates"), (), ()),

    # Procurement and suppliers.
    "resolve_supplier_eligibility": (("supplier_profiles",), ("eligibility_policies",), ()),
    "resolve_quote_rank": (("supplier_quotes",), ("supplier_quotes",), ("supplier_quotes",)),
    "resolve_purchase_limit": (("budget_state",), ("approval_policies",), ()),
    "resolve_order_split": (("supplier_capacity",), ("requested_purchases",), ()),
    "resolve_lead_time": (("supplier_lanes",), ("business_calendars",), ()),
    "resolve_contract_compliance": (("contracts",), ("purchase_orders",), ()),
    "resolve_receipt_variance": (("purchase_orders",), ("goods_receipts",), ("goods_receipts",)),
    "resolve_return_authorization": (("goods_receipts",), ("return_policies",), ("return_policies",)),
    "resolve_vendor_score": (("supplier_performance",), (), ()),
    "resolve_expedite_decision": (("demand_risk",), ("supplier_options",), ()),

    # Customer support.
    "resolve_case_owner": (("cases",), ("routing_policies",), ("staff_roster",)),
    "resolve_case_priority": (("cases",), ("severity_rules",), ()),
    "resolve_service_entitlement": (("customer_plans",), ("cases",), ("service_entitlement_rules",)),
    "resolve_refund_eligibility": (("order_state",), ("refund_policies",), ("refund_policies",)),
    "resolve_sla_deadline": (("cases",), ("sla_rules", "business_calendars"), ()),
    "resolve_next_action": (("cases",), ("workflow_rules",), ()),
    "resolve_duplicate_case": (("case_features",), ("cases",), ("case_similarity_index",)),
    "resolve_escalation_target": (("cases",), ("severity_rules", "escalation_policies"), ()),
    "resolve_contact_channel": (("customer_preferences", "cases"), ("severity_rules", "channel_policies"), ("channel_policies",)),
    "resolve_case_closure": (("resolution_evidence",), ("closure_rules", "cases"), ("closure_rules",)),

    # Workforce and access.
    "resolve_access_decision": (("identity_state",), ("resource_policies", "role_assignments"), ()),
    "resolve_role_assignment": (("job_profiles",), ("role_matrix",), ("role_matrix",)),
    "resolve_shift_eligibility": (("employee_state", "role_assignments"), ("shift_rules", "work_history"), ("shift_rules",)),
    "resolve_leave_balance": (("leave_ledger",), (), ()),
    "resolve_overtime_approval": (("work_history",), ("overtime_policy", "role_assignments"), ("overtime_policy",)),
    "resolve_training_requirement": (("employee_state", "job_profiles", "role_matrix"), ("training_matrix",), ()),
    "resolve_delegation_scope": (("delegation_records",), ("scope_policy",), ("scope_policy",)),
    "resolve_oncall_assignment": (("roster",), (), ()),
    "resolve_badge_expiry": (("badge_records",), ("validity_policy",), ()),
    "resolve_offboarding_actions": (("employee_state",), ("offboarding_matrix",), ()),

    # Finance and ledger.
    "resolve_book_balance": (("ledger_entries",), (), ()),
    "resolve_available_balance": (("ledger_entries",), ("holds",), ("holds",)),
    "resolve_transaction_fee": (("transactions",), ("fee_schedules",), ("fee_schedules",)),
    "resolve_reconciliation_status": (("statements",), ("ledger_entries",), ()),
    "resolve_budget_variance": (("budgets",), ("actuals",), ()),
    "resolve_tax_basis": (("transactions",), ("tax_rules",), ()),
    "resolve_fx_settlement": (("transactions",), ("exchange_rates",), ()),
    "resolve_payment_status": (("payment_events",), (), ()),
    "resolve_journal_total": (("journals",), (), ()),
    "resolve_audit_flag": (("account_events",), ("audit_rules",), ()),

    # Compliance and audit.
    "resolve_control_status": (("controls", "control_evidence"), ("control_rules",), ()),
    "resolve_evidence_completeness": (("control_rules",), ("control_evidence",), ()),
    "resolve_policy_applicability": (("entities",), ("policies",), ()),
    "resolve_exception_expiry": (("exceptions",), ("exception_extensions",), ()),
    "resolve_risk_rating": (("risk_factors",), ("risk_scoring_matrix",), ()),
    "resolve_audit_sample": (("audit_population",), ("sampling_rules",), ()),
    "resolve_segregation_conflict": (("role_assignments",), ("conflict_matrix",), ("conflict_matrix",)),
    "resolve_retention_deadline": (("record_classes", "record_events"), ("retention_rules",), ()),
    "resolve_remediation_owner": (("findings",), ("ownership_matrix",), ()),
    "resolve_filing_readiness": (("filing_requirements",), ("current_artifacts",), ()),

    # Contracts and documents.
    "resolve_effective_clause": (("contract_versions",), ("contract_versions",), ("clauses",)),
    "resolve_obligation_owner": (("clause_obligations",), ("party_mapping",), ()),
    "resolve_contract_deadline": (("clause_dates",), ("calendar_rules", "business_calendars"), ()),
    "resolve_latest_version": (("revision_history",), ("revision_history",), ()),
    "resolve_document_access": (("document_acl",), ("requester_roles",), ()),
    "resolve_citation_target": (("citation_index",), ("citation_index",), ()),
    "resolve_amendment_impact": (("contracts", "clauses"), ("amendments",), ("clause_obligations", "clauses")),
    "resolve_signature_status": (("signature_events",), ("signature_rules",), ()),
    "resolve_retention_class": (("document_metadata",), ("classification_rules",), ()),
    "resolve_section_summary": (("document_sections",), (), ()),
}

