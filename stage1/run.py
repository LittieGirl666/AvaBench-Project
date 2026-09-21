"""Prepare replay contexts, rerun qualification, or score complete action names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DATA,
    ROOT,
    MODELS,
    load_registry,
    model_runtime,
    read_json,
    read_jsonl,
    write_json,
    write_csv,
)
from envtoolbench.v2.agent_environment import EnvironmentSession
from envtoolbench.v2.agent_evaluation import Evaluator
from envtoolbench.v2.agent_harness import AgentHarness, HarnessConfig, control_tools
from envtoolbench.v2.query_generation import query_case_from_dict
from envtoolbench.v2.rendering import render_agent_messages, render_agent_tools
from envtoolbench.v2.variants import get_variant
from stage1.pairs import LOW_VARIANTS, prepare
from stage1.scoring import _margins


def qualify(args, runtime):
    registry = load_registry("structural", args.data)
    queries = read_json(args.data / "structural/queries.json")
    if args.limit:
        queries = queries[: args.limit]
    output = args.output / args.model / "qualification.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if output.exists() else []
    if existing and not args.resume:
        raise FileExistsError(f"Use --resume or a new --output: {output}")
    completed = {(r["case_id"], r["variant_id"]): r for r in existing}
    harness = AgentHarness(HarnessConfig(provider_retries=0))
    evaluator = Evaluator()
    with output.open("a" if existing else "w") as stream:
        for raw in queries:
            case = query_case_from_dict(raw)
            for variant in ("sub-operations",) + LOW_VARIANTS:
                if (case.id, variant) in completed:
                    continue
                resolution = get_variant(variant).resolve(
                    registry, case, seed=registry.seed
                )
                session = EnvironmentSession(
                    registry,
                    case,
                    visible_operation_ids=resolution.visible_operation_ids,
                )
                run = harness.run(
                    case=case,
                    driver=runtime,
                    messages=render_agent_messages(registry, case),
                    environment_tools=render_agent_tools(
                        registry, resolution.visible_operation_ids
                    ),
                    session=session,
                )
                if run.infrastructure_error:
                    raise RuntimeError(run.infrastructure_error)
                evaluation = evaluator.evaluate(
                    registry,
                    case,
                    visible_operation_ids=resolution.visible_operation_ids,
                    run=run,
                )
                row = {
                    "case_id": case.id,
                    "variant_id": variant,
                    "termination": run.termination_reason,
                    "outcome_success": evaluation.to_dict()["outcome_success"],
                    "model_turns": run.model_turns,
                }
                completed[case.id, variant] = row
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    f"qualify {args.model} {case.id} {variant}: {run.termination_reason}",
                    flush=True,
                )
    eligible, phenotypes = [], {}
    for raw in queries:
        case = raw["id"]
        if not completed[case, "sub-operations"]["outcome_success"]:
            continue
        terms = {v: completed[case, v]["termination"] for v in LOW_VARIANTS}
        if all(t in {"unavailable", "budget_exhausted"} for t in terms.values()):
            eligible.append(case)
            phenotypes[case] = terms
    write_json(
        args.output / f"cohorts/{args.model}.json",
        {"model": args.model, "eligible_case_ids": eligible, "phenotypes": phenotypes},
    )
    print(f"{len(eligible)} eligible tasks → {args.output / 'cohorts'}")


def score(args, runtime):
    families = ["structural", "semantic"] if args.family == "both" else [args.family]
    output = args.output / args.model / "context_scores.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(output) if output.exists() else []
    if rows and not args.resume:
        raise FileExistsError(f"Use --resume or a new --output: {output}")
    done = {(r["pair_id"], r["condition"]) for r in rows}
    with output.open("a" if rows else "w") as stream:
        for family in families:
            path = args.output / (
                f"pairs/{args.model}/structural.jsonl"
                if family == "structural"
                else "pairs/semantic.jsonl"
            )
            pairs = [p for p in read_jsonl(path) if not p.get("alias_of")]
            if args.limit:
                pairs = pairs[: args.limit]
            registry = load_registry(family, args.data)
            for pair in pairs:
                for context in pair["contexts"]:
                    if (pair["pair_id"], context["label"]) in done:
                        continue
                    visible = tuple(context["visible_operation_ids"])
                    tokens, scores = runtime.score_candidates(
                        messages=context["messages"],
                        tools=render_agent_tools(registry, visible) + control_tools(),
                        candidate_names=pair["candidate_universe"],
                    )
                    margins = _margins(
                        family=family, scores=scores, legal_names=set(visible)
                    )
                    row = {
                        "model": args.model,
                        "family": family,
                        "pair_id": pair["pair_id"],
                        "case_id": pair["case_id"],
                        "cluster_id": pair["cluster_id"],
                        "domain": pair["domain"],
                        "contrast": pair.get(
                            "low_variant", "decoy_success_minus_failure"
                        ),
                        "condition": context["label"],
                        "phenotype": context["phenotype"],
                        "prompt_tokens": tokens,
                        "margin": (
                            margins["legal_margin"]
                            if family == "structural"
                            else margins["unavailable_margin"]
                        ),
                        "candidate_scores": [
                            {
                                "candidate_name": s["candidate_name"],
                                "sequence_logprob": s["sequence_logprob"],
                            }
                            for s in scores
                        ],
                    }
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    rows.append(row)
                    print(
                        f"score {args.model} {pair['pair_id']} {context['label']}",
                        flush=True,
                    )
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["condition"]] = r
    effects = []
    for pair in by_pair.values():
        if set(pair) != {"x_H", "x_L"}:
            continue
        high, low = pair["x_H"], pair["x_L"]
        effect = (
            (low["margin"] - high["margin"])
            if high["family"] == "structural"
            else (high["margin"] - low["margin"])
        )
        effects.append(
            {
                **{
                    k: high[k]
                    for k in (
                        "model",
                        "family",
                        "pair_id",
                        "case_id",
                        "cluster_id",
                        "domain",
                        "contrast",
                    )
                },
                "phenotype": low["phenotype"],
                "high_margin": high["margin"],
                "low_margin": low["margin"],
                "effect": effect,
            }
        )
    if effects:
        write_csv(args.output / args.model / "pair_effects.csv", effects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "qualify", "score"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--family", choices=["both", "structural", "semantic"], default="both"
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--cohorts", type=Path, default=DATA / "cohorts")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "prepare":
        prepare(
            args.output, [args.model] if args.model else MODELS, args.cohorts, args.data
        )
        return
    if args.model is None:
        parser.error("--model is required for qualify and score")
    runtime = model_runtime(args.model, args.model_dir)
    (qualify if args.command == "qualify" else score)(args, runtime)


if __name__ == "__main__":
    main()
