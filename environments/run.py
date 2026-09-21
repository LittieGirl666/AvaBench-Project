"""Run structural and semantic interactions through a hosted model API."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from common import (
    DATA,
    ROOT,
    STRUCTURAL,
    SEMANTIC,
    load_registry,
    read_json,
    read_csv,
    write_json,
)
from envtoolbench.v2.agent_environment import EnvironmentSession
from envtoolbench.v2.agent_evaluation import Evaluator
from envtoolbench.v2.agent_harness import (
    AgentHarness,
    ChatCompletionsDriver,
    HarnessConfig,
)
from envtoolbench.v2.deepinfer import (
    DeepInferChatDriver,
    deepinfer_chat_completions_url,
)
from envtoolbench.v2.siliconflow import SiliconFlowChatDriver
from envtoolbench.v2.query_generation import query_case_from_dict
from envtoolbench.v2.rendering import (
    render_agent_messages,
    render_agent_tools,
    render_stateful_agent_messages,
)
from envtoolbench.v2.stateful_evaluation import StatefulEvaluator
from envtoolbench.v2.stateful_variants import get_stateful_variant
from envtoolbench.v2.variants import get_variant

FIELDS = [
    "family",
    "model",
    "case_id",
    "pair_id",
    "domain",
    "variant_id",
    "termination_reason",
    "outcome_success",
    "unavailable_correct",
    "behavior_correct",
    "reachability",
    "model_turns",
    "environment_calls",
    "restarts",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "decoy_call_count",
    "successful_decoy_call_count",
    "wrong_state_commit_count",
    "wrong_state_committed",
    "false_finish_after_decoy",
    "wrong_state_detected",
    "restart_recovered",
    "compensating_action_count",
    "turns_to_detection",
    "target_state_achieved",
    "residual_wrong_state_at_termination",
]


def make_driver(args, settings):
    endpoint = deepinfer_chat_completions_url(args.base_url)
    common = dict(
        model=args.model,
        endpoint=endpoint,
        api_key=os.environ["AVABENCH_API_KEY"],
        timeout=args.timeout,
    )
    if settings["provider"] == "DeepInfer":
        return DeepInferChatDriver(**common, thinking_mode="enabled", max_tokens=16384)
    if settings["provider"] == "SiliconFlow":
        return SiliconFlowChatDriver(
            **common, thinking_mode="enabled", max_tokens=16384, thinking_budget=16384
        )
    return ChatCompletionsDriver(**common, extra_body={"max_tokens": 16384})


def run_family(*, family, args, driver, config):
    registry = load_registry(family, args.data)
    queries = read_json(args.data / family / "queries.json")
    if args.limit is not None:
        queries = queries[: args.limit]
    variants = STRUCTURAL if family == "structural" else SEMANTIC
    if args.variants:
        variants = tuple(args.variants.split(","))
        allowed = STRUCTURAL if family == "structural" else SEMANTIC
        if any(v not in allowed for v in variants):
            raise ValueError(f"Choose {family} variants from {', '.join(allowed)}")
    output = args.output / f"{family}_cases.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_csv(output) if output.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"{output} already exists; use --resume or a new --output directory"
        )
    completed = {(r["case_id"], r["variant_id"]) for r in existing}
    evaluator = Evaluator() if family == "structural" else StatefulEvaluator()
    harness = AgentHarness(config)
    with output.open("a" if existing else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        for raw in queries:
            case = query_case_from_dict(raw)
            render = (
                render_agent_messages
                if family == "structural"
                else render_stateful_agent_messages
            )
            messages = render(registry, case)
            for variant in variants:
                if (case.id, variant) in completed:
                    continue
                resolution = (
                    get_variant if family == "structural" else get_stateful_variant
                )(variant).resolve(registry, case, seed=registry.seed)
                session = EnvironmentSession(
                    registry,
                    case,
                    visible_operation_ids=resolution.visible_operation_ids,
                    **(
                        {"execution_failures": resolution.execution_failures()}
                        if family == "semantic"
                        else {}
                    ),
                )
                case_driver = (
                    driver.with_user_id(f"avabench:{family}:{case.id}:{variant}")
                    if isinstance(driver, DeepInferChatDriver)
                    else driver
                )
                result = harness.run(
                    case=case,
                    driver=case_driver,
                    messages=messages,
                    environment_tools=render_agent_tools(
                        registry, resolution.visible_operation_ids
                    ),
                    session=session,
                )
                if result.infrastructure_error:
                    raise RuntimeError(result.infrastructure_error)
                evaluation = evaluator.evaluate(
                    registry,
                    case,
                    run=result,
                    **(
                        {"visible_operation_ids": resolution.visible_operation_ids}
                        if family == "structural"
                        else {"resolution": resolution}
                    ),
                ).to_dict()
                row = {key: "" for key in FIELDS}
                row.update(
                    family=family,
                    model=args.model,
                    case_id=case.id,
                    domain=case.domain,
                    pair_id=raw["hidden_metadata"].get("pair_id", ""),
                    variant_id=variant,
                    termination_reason=result.termination_reason,
                    model_turns=result.model_turns,
                    environment_calls=result.environment_call_count,
                    restarts=result.restart_count,
                    outcome_success=int(evaluation["outcome_success"]),
                    unavailable_correct=int(bool(evaluation["unavailable_correct"])),
                    reachability=evaluation["reachability"]["status"],
                )
                row["behavior_correct"] = (
                    row["outcome_success"]
                    if row["reachability"] == "reachable"
                    else row["unavailable_correct"]
                )
                row.update(
                    {
                        k: result.usage.get(k, "")
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                )
                row.update(
                    {
                        k: int(v) if isinstance(v, bool) else v
                        for k, v in evaluation.get("state_metrics", {}).items()
                    }
                )
                writer.writerow(row)
                stream.flush()
                if args.save_trajectories:
                    with (args.output / f"{family}_trajectories.jsonl").open(
                        "a"
                    ) as traces:
                        traces.write(
                            json.dumps(
                                {
                                    "case_id": case.id,
                                    "variant_id": variant,
                                    "events": result.events,
                                    "evaluation": evaluation,
                                }
                            )
                            + "\n"
                        )
                print(
                    f"{family} {case.id} {variant}: {result.termination_reason}",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--family", choices=["both", "structural", "semantic"], default="both"
    )
    parser.add_argument("--base-url", default=os.environ.get("AVABENCH_API_BASE"))
    parser.add_argument(
        "--provider", choices=["DeepInfer", "SiliconFlow", "OpenAI-compatible"]
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--variants", help="Comma-separated condition IDs; use with one --family"
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()
    if not args.base_url or not os.environ.get("AVABENCH_API_KEY"):
        parser.error("Set AVABENCH_API_BASE and AVABENCH_API_KEY, or pass --base-url")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.variants and args.family == "both":
        parser.error("--variants requires --family structural or semantic")
    settings = next(
        (
            s.copy()
            for s in read_json(ROOT / "environments/api_settings.json")
            if s["model"] == args.model
        ),
        {"provider": "OpenAI-compatible", "temperature": 0.0},
    )
    if args.provider:
        settings["provider"] = args.provider
    if args.temperature is not None:
        settings["temperature"] = args.temperature
    args.output = args.output or ROOT / "outputs/behavior" / args.model.replace(
        "/", "--"
    )
    driver = make_driver(args, settings)
    config = HarnessConfig(temperature=settings["temperature"], model_seed=None)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "settings.json",
        {
            "model": args.model,
            "provider": settings["provider"],
            "base_url": args.base_url,
            "temperature": config.temperature,
            "harness": config.__dict__,
        },
    )
    for family in (
        ["structural", "semantic"] if args.family == "both" else [args.family]
    ):
        run_family(family=family, args=args, driver=driver, config=config)


if __name__ == "__main__":
    main()
