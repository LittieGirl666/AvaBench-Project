"""Rebuild seed-zero resources and export the fixed paper queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DATA,
    ROOT,
    SEMANTIC,
    STRUCTURAL,
    load_registry,
    read_json,
    write_json,
)
from envtoolbench.v2.agent_harness import control_tools
from envtoolbench.v2.query_generation import query_case_from_dict
from envtoolbench.v2.rendering import (
    render_agent_messages,
    render_agent_tools,
    render_stateful_agent_messages,
)
from envtoolbench.v2.stateful_variants import get_stateful_variant
from envtoolbench.v2.variants import get_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/environments")
    args = parser.parse_args()
    for family, variants in (("structural", STRUCTURAL), ("semantic", SEMANTIC)):
        registry = load_registry(family)
        queries = read_json(DATA / family / "queries.json")
        write_json(
            args.output / family / "registry.json",
            registry.to_dict(include_resources=True),
        )
        write_json(args.output / family / "queries.json", queries)
        case = query_case_from_dict(queries[0])
        messages = (
            render_agent_messages
            if family == "structural"
            else render_stateful_agent_messages
        )(registry, case)
        examples = []
        for variant in variants:
            resolution = (
                get_variant if family == "structural" else get_stateful_variant
            )(variant).resolve(registry, case, seed=registry.seed)
            examples.append(
                {
                    "variant": variant,
                    "case_id": case.id,
                    "messages": messages,
                    "tools": list(
                        render_agent_tools(registry, resolution.visible_operation_ids)
                        + control_tools()
                    ),
                    "condition": resolution.to_dict(),
                }
            )
        write_json(args.output / family / "example_inputs.json", examples)
        print(
            f"{family}: {len(queries)} tasks × {len(variants)} conditions → {args.output / family}"
        )


if __name__ == "__main__":
    main()
