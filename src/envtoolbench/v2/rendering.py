from __future__ import annotations

import ast

from .registry import V2Registry
from .schemes.enterprise_operations.manifest import DOMAIN_TITLES
from .types import BaseOperationSpec, SubOperationSpec, V2QueryCase


def normalize_implementation(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.unparse(tree)


def render_tool(operation: BaseOperationSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": operation.operation_id,
            "description": operation.description,
            "parameters": operation.visible_input_schema,
        },
    }


def render_sub_tool(operation: SubOperationSpec) -> dict:
    parameters = operation.visible_input_schema
    if operation.canonical_parameters == ("payload",):
        parameters = {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "description": "Copy the preceding successful observation's payload_ref object exactly.",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "Opaque harness-issued reference; never invent or edit this value.",
                        },
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            },
            "required": ["payload"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": operation.operation_id,
            "description": operation.description,
            "parameters": parameters,
        },
    }


def render_environment_context(registry: V2Registry, domain: str) -> str:
    lines = [f"Environment: {DOMAIN_TITLES[domain]}", "", "Visible resource schemas:"]
    domain_resources = sorted(
        (
            resource
            for resource in registry.resources.values()
            if resource.resource_id.startswith(f"{domain}.") and resource.kind != "private_table"
        ),
        key=lambda resource: resource.resource_id,
    )
    for resource in domain_resources:
        fields = ", ".join(
            f"{name}: {schema.get('type', 'any')}"
            for name, schema in resource.schema["properties"].items()
        )
        lines.append(f"- {resource.resource_id.split('.', 1)[1]}({fields})")
    lines.extend(["", "Relations:"])
    relations = []
    for resource in domain_resources:
        for relation in resource.relationships:
            target = relation["target_resource"].split(".", 1)[1]
            relations.append(
                f"- {resource.resource_id.split('.', 1)[1]}.{relation['fields'][0]} "
                f"joins {target}.{relation['target_fields'][0]}"
            )
    lines.extend(sorted(set(relations)) or ["- Policy identifiers select applicable deterministic rules."])
    lines.extend([
        "",
        "Glossary: subject_id identifies the primary entity; context_id identifies its business scope; "
        "the family-specific third argument supplies the reference, policy, or value.",
        "Resource values are private and can only be accessed through the available tools.",
        "Payload transport protocol: every successful environment observation includes an opaque payload_ref object. "
        "For a sub-operation whose only argument is payload, pass that object exactly as payload, for example "
        "{\"payload\":{\"ref\":\"payload:e0:s1\"}}. The observation output is for reasoning only: never copy it into payload, "
        "never invent or edit a ref, and never reuse a ref after harness_restart. In sub-operation descriptions, "
        "'preceding tool result' means this payload_ref rather than the output body.",
        "Use the visible environment operations and harness controls to resolve the request. A complete solution may or may not be available. "
        "Make exactly one native tool call per turn, inspect every observation, and then call harness_finish, harness_unavailable, "
        "or harness_restart when appropriate.",
    ])
    return "\n".join(lines)


def render_agent_tools(registry: V2Registry, operation_ids: tuple[str, ...]) -> tuple[dict, ...]:
    return tuple(
        render_tool(registry.get_base(operation_id))
        if operation_id in registry.base_operations
        else render_sub_tool(registry.get_sub(operation_id))
        for operation_id in operation_ids
    )


def render_agent_messages(registry: V2Registry, case: V2QueryCase) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Resolve the request through the native tools. The visible environment operations may or may not contain a complete solution. "
                "Return exactly one native tool call per turn and never call tools in parallel. Inspect each tool observation before deciding "
                "whether to call another environment operation, finish with evidence, declare the request unavailable, or restart. "
                "When a tool accepts only payload, pass the preceding successful observation's payload_ref object exactly; do not copy output."
            ),
        },
        {"role": "developer", "content": render_environment_context(registry, case.domain)},
        {"role": "user", "content": case.prompt},
    ]


def render_stateful_agent_messages(
    registry: V2Registry,
    case: V2QueryCase,
) -> list[dict]:
    """Render the action-layer prompt without changing legacy evaluation prompts."""

    return [
        {
            "role": "system",
            "content": (
                "Complete the requested authoritative business-state transition through "
                "the native tools. A successful tool call is not by itself proof that "
                "the requested state utility was achieved. Inspect the operation "
                "postcondition and state_delta, and finish only with evidence for the "
                "exact requested transition. Return exactly one native tool call per "
                "turn and never call tools in parallel. If a wrong transition has "
                "already changed state, use harness_restart before attempting a clean "
                "plan. Declare unavailable when no complete target transition is "
                "possible through the visible operations."
            ),
        },
        {
            "role": "developer",
            "content": (
                render_environment_context(registry, case.domain)
                + "\n\nState-action protocol: successful mutation calls return a "
                "normalized receipt and a state_delta. The feedback is factual and "
                "does not label a successful but non-target transition as an error. "
                "Operation-specific postconditions remain distinct."
            ),
        },
        {"role": "user", "content": case.prompt},
    ]
