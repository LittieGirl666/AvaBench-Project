from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution import EnvironmentState, execute_base, execute_sub_operation
from .types import BaseOperationSpec, JsonObject, SubOperationSpec, V2ResourceSpec, public_dataclass_dict
from .schemes.enterprise_operations.manifest import DOMAIN_TITLES
from .schemes.enterprise_operations.suite import EnterpriseSuite, build_suite


@dataclass
class V2Registry:
    scheme: str
    seed: int
    domains: tuple[str, ...]
    base_operations: dict[str, BaseOperationSpec]
    sub_operations: dict[str, SubOperationSpec]
    resources: dict[str, V2ResourceSpec]
    _resource_rows: JsonObject
    fixtures: dict[str, list[JsonObject]]
    operation_effects: dict[str, str]

    def get_base(self, operation_id: str) -> BaseOperationSpec:
        return self.base_operations[operation_id]

    def get_sub(self, operation_id: str) -> SubOperationSpec:
        return self.sub_operations[operation_id]

    def new_state(self) -> EnvironmentState:
        return EnvironmentState(self._resource_rows).clone()

    def execute_base(self, operation_id: str, arguments: JsonObject, state: EnvironmentState | None = None):
        return execute_base(self.get_base(operation_id), state or self.new_state(), arguments)

    def execute_sub(self, operation_id: str, arguments: JsonObject, state: EnvironmentState | None = None) -> Any:
        operation = self.get_sub(operation_id)
        return operation.handler(state or self.new_state(), **arguments)

    def execute_sub_result(self, operation_id: str, arguments: JsonObject, state: EnvironmentState | None = None):
        return execute_sub_operation(self.get_sub(operation_id), state or self.new_state(), arguments)

    def to_dict(self, *, include_resources: bool = False) -> JsonObject:
        base_records = []
        for operation in self.base_operations.values():
            record = public_dataclass_dict(operation)
            record["implementation_diagnostics"] = {
                "character_length": len(operation.implementation_source),
                "token_length": len(operation.implementation_source.split()),
            }
            base_records.append(record)
        data = {
            "scheme": self.scheme, "semantics_version": "strict-staged-v1",
            "seed": self.seed, "domains": list(self.domains),
            "base_operations": base_records,
            "sub_operations": [public_dataclass_dict(op) for op in self.sub_operations.values()],
            "resources": [public_dataclass_dict(resource) for resource in self.resources.values()],
        }
        if include_resources:
            data["resource_rows"] = self._resource_rows
        return data


def build_v2_registry(*, scheme: str = "enterprise_operations", domains: list[str] | None = None, seed: int = 0) -> V2Registry:
    if scheme != "enterprise_operations":
        raise ValueError(f"unsupported scheme: {scheme}")
    suite = build_suite(domains=domains, seed=seed)
    selected = tuple(DOMAIN_TITLES if domains is None else domains)
    return V2Registry(scheme, seed, selected, suite.bases, suite.subs, suite.resources, suite.resource_rows, suite.fixtures, suite.operation_effects)
