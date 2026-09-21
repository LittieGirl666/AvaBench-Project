"""Construct matched replay contexts from executable environment steps."""

from __future__ import annotations

from pathlib import Path

from common import DATA, MODELS, load_registry, read_json, write_jsonl
from envtoolbench.v2.agent_harness import HarnessConfig, control_tools
from envtoolbench.v2.query_generation import query_case_from_dict
from envtoolbench.v2.rendering import (
    render_agent_messages,
    render_stateful_agent_messages,
)
from envtoolbench.v2.stateful_variants import get_stateful_variant
from envtoolbench.v2.variants import get_variant
from stage1.replay import _execute_structural_prefix, _execute_semantic_history

LOW_VARIANTS = (
    "sub-operations-no-gold",
    "sub-operations-no-first",
    "sub-operations-first-only",
    "sub-operations-no-last",
)


def context(label, resolution, messages, phenotype):
    return {
        "label": label,
        "variant_id": resolution.variant_id,
        "visible_operation_ids": list(resolution.visible_operation_ids),
        "messages": messages,
        "phenotype": phenotype,
    }


def structural_pairs(model, cohorts=DATA / "cohorts", data=DATA):
    registry = load_registry("structural", data)
    selected = read_json(Path(cohorts) / f"{model}.json")
    eligible = set(selected["eligible_case_ids"])
    config = HarnessConfig(provider_retries=0)
    for raw in read_json(Path(data) / "structural/queries.json"):
        if raw["id"] not in eligible:
            continue
        case = query_case_from_dict(raw)
        base = render_agent_messages(registry, case)
        high = get_variant("sub-operations").resolve(registry, case, seed=registry.seed)
        steps = tuple(high.target_step_ids)
        for variant in LOW_VARIANTS:
            low = get_variant(variant).resolve(registry, case, seed=registry.seed)
            if variant in LOW_VARIANTS[:2]:
                prefix, checkpoint, depth = (), "pre_action", 0.0
            elif variant == "sub-operations-first-only":
                prefix, checkpoint, depth = (
                    steps[:1],
                    "after_first_success",
                    1.0 / max(len(steps) - 1, 1),
                )
            else:
                prefix, checkpoint, depth = steps[:-1], "after_penultimate_success", 1.0
            histories = [
                _execute_structural_prefix(
                    registry=registry,
                    case=case,
                    visible_ids=r.visible_operation_ids,
                    step_ids=prefix,
                    harness_config=config,
                )
                for r in (high, low)
            ]
            if histories[0] != histories[1]:
                raise ValueError(f"Structural histories differ: {case.id} {variant}")
            yield {
                "family": "structural",
                "model_key": model,
                "pair_id": f"structural:{case.id}:{variant}:{checkpoint}",
                "case_id": case.id,
                "domain": case.domain,
                "cluster_id": case.expected_call.canonical_operation_id,
                "low_variant": variant,
                "checkpoint_id": checkpoint,
                "progress_depth": depth,
                "alias_of": (
                    "sub-operations-first-only"
                    if variant == "sub-operations-no-last" and len(steps) == 2
                    else None
                ),
                "target_step_ids": list(steps),
                "candidate_universe": list(high.visible_operation_ids)
                + [t["function"]["name"] for t in control_tools()],
                "contexts": [
                    context("x_H", high, base + histories[0], "available_success"),
                    context(
                        "x_L",
                        low,
                        base + histories[1],
                        selected["phenotypes"][case.id][variant],
                    ),
                ],
            }


def semantic_pairs(data=DATA):
    registry = load_registry("semantic", data)
    config = HarnessConfig(provider_retries=0)
    for raw in read_json(Path(data) / "semantic/queries.json"):
        case = query_case_from_dict(raw)
        high = get_stateful_variant("state-decoy-success").resolve(
            registry, case, seed=registry.seed
        )
        low = get_stateful_variant("state-decoy-failure").resolve(
            registry, case, seed=registry.seed
        )
        histories = [
            _execute_semantic_history(
                registry=registry, case=case, resolution=r, harness_config=config
            )
            for r in (high, low)
        ]
        base = render_stateful_agent_messages(registry, case)
        if (
            high.visible_operation_ids != low.visible_operation_ids
            or histories[0][:-2] != histories[1][:-2]
        ):
            raise ValueError(f"Semantic tools or pre-decoy histories differ: {case.id}")
        yield {
            "family": "semantic",
            "pair_id": f"semantic:{case.id}:decoy-observation",
            "case_id": case.id,
            "domain": case.domain,
            "cluster_id": raw["hidden_metadata"]["pair_id"],
            "checkpoint_id": "after_decoy_observation",
            "progress_depth": None,
            "candidate_universe": list(high.visible_operation_ids)
            + [t["function"]["name"] for t in control_tools()],
            "contexts": [
                context("x_H", high, base + histories[0], "wrong_state_success"),
                context("x_L", low, base + histories[1], "explicit_failure"),
            ],
        }


def prepare(output, models=MODELS, cohorts=DATA / "cohorts", data=DATA):
    output = Path(output)
    print("Building semantic resources and replaying task pairs...", flush=True)
    semantic = []
    for pair in semantic_pairs(data):
        semantic.append(pair)
        if len(semantic) % 50 == 0:
            print(f"semantic: {len(semantic)}/350 replay pairs", flush=True)
    write_jsonl(output / "pairs/semantic.jsonl", semantic)
    print(f"semantic: {len(semantic)} replay pairs saved", flush=True)
    for model in models:
        pairs = []
        for pair in structural_pairs(model, cohorts, data):
            pairs.append(pair)
            if len(pairs) % 200 == 0:
                print(
                    f"{model}: {len(pairs)} structural comparisons replayed", flush=True
                )
        write_jsonl(output / f"pairs/{model}/structural.jsonl", pairs)
        independent = sum(not p["alias_of"] for p in pairs)
        print(
            f"{model} structural: {independent} distinct replay pairs saved", flush=True
        )
