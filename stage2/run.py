"""Capture and replace residual states at the first differing action token."""

from __future__ import annotations

import argparse
import csv
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
    read_csv,
)
from envtoolbench.v2.agent_harness import control_tools
from envtoolbench.v2.rendering import render_agent_tools
from stage2.intervention import (
    define_endpoint,
    capture_readout,
    append_token_prefix,
    score_readout_patched,
    donor_aligned_effect,
)

PAIRED_FIELDS = [
    "model_key",
    "family",
    "contrast",
    "split",
    "pair_id",
    "cluster_id",
    "direction",
    "layer",
    "clean_donor_margin",
    "clean_recipient_margin",
    "clean_gap_abs",
    "patched_margin",
    "donor_aligned_effect",
]
CONTROL_FIELDS = [
    "family",
    "model_key",
    "pair_id",
    "cluster_id",
    "control",
    "direction",
    "activation_donor_pair_id",
    "clean_recipient_margin",
    "clean_reference_paired_donor_margin",
    "patched_margin",
    "donor_aligned_effect",
    "clean_gap_abs",
]
DIRECTIONS = {"high_to_low": ("x_H", "x_L"), "low_to_high": ("x_L", "x_H")}


def selected_pairs(args, family):
    filename = (
        f"{args.model}/structural.jsonl" if family == "structural" else "semantic.jsonl"
    )
    pairs = [p for p in read_jsonl(args.pairs / filename) if not p.get("alias_of")]
    ids = set(read_json(args.cohort)[family]["pair_ids"])
    by_id = {p["pair_id"]: p for p in pairs}
    missing = ids - set(by_id)
    if missing:
        raise ValueError(
            f"Prepare the paper replay cohort first: {len(missing)} {family} pairs are missing"
        )
    selected = [p for p in pairs if p["pair_id"] in ids]
    return selected[: args.limit] if args.limit else selected


def cache_dir(args, family):
    return args.output / args.model / "activations" / family


def capture(args, runtime, family):
    folder = cache_dir(args, family)
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "index.jsonl"
    old = read_jsonl(index) if index.exists() else []
    if old and not args.resume:
        raise FileExistsError(f"Use --resume or a new --output: {index}")
    done = {r["pair_id"] for r in old}
    registry = load_registry(family, args.data)
    with index.open("a" if old else "w") as stream:
        for number, pair in enumerate(selected_pairs(args, family)):
            if pair["pair_id"] in done:
                continue
            endpoint = define_endpoint(runtime, pair)
            states, clean, lengths = {}, {}, {}
            for context in pair["contexts"]:
                label = context["label"]
                tools = (
                    render_agent_tools(
                        registry, tuple(context["visible_operation_ids"])
                    )
                    + control_tools()
                )
                count, residuals, scores = capture_readout(
                    runtime=runtime,
                    messages=context["messages"],
                    tools=tools,
                    lcp_ids=endpoint.lcp_ids,
                    high_token_id=endpoint.high_token_id,
                    low_token_id=endpoint.low_token_id,
                )
                # The source hook captures prompt-end and decision positions; only the
                # latter is needed by the paper's replacement experiment.
                states[label] = residuals[1].cpu()
                clean[label], lengths[label] = scores["margin"], count
            filename = f"pair-{number:04d}.pt"
            runtime.torch.save(states, folder / filename)
            metadata = {
                "pair_id": pair["pair_id"],
                "tensor_file": filename,
                "lcp_ids": list(endpoint.lcp_ids),
                "high_token_id": endpoint.high_token_id,
                "low_token_id": endpoint.low_token_id,
                "high_action": endpoint.high_action,
                "low_action": endpoint.low_action,
                "clean_margins": clean,
                "prompt_tokens": lengths,
                "high_candidate_ids": list(endpoint.high_candidate_ids),
                "low_candidate_ids": list(endpoint.low_candidate_ids),
            }
            stream.write(json.dumps(metadata) + "\n")
            stream.flush()
            print(f"capture {args.model} {pair['pair_id']}", flush=True)


def encoded_recipient(runtime, registry, pair, label, metadata):
    context = next(c for c in pair["contexts"] if c["label"] == label)
    encoded = runtime.render_inputs(
        messages=context["messages"],
        tools=render_agent_tools(registry, tuple(context["visible_operation_ids"]))
        + control_tools(),
    )
    return append_token_prefix(encoded, metadata["lcp_ids"], runtime.torch)


def replace(runtime, encoded, residual, metadata, layer):
    scores, _ = score_readout_patched(
        runtime=runtime,
        encoded_inputs=encoded,
        layer_index=layer,
        donor=residual[layer],
        alpha=1.0,
        position_index=-1,
        high_token_id=metadata["high_token_id"],
        low_token_id=metadata["low_token_id"],
    )
    return scores["margin"]


def patch(args, runtime, family):
    folder = cache_dir(args, family)
    cache = {r["pair_id"]: r for r in read_jsonl(folder / "index.jsonl")}
    registry = load_registry(family, args.data)
    setting = read_json(ROOT / "stage2/settings.json")
    first, last = setting["late_blocks"][args.model][family]
    sweep_ids = set(read_json(args.cohort)[family]["layer_sweep_pair_ids"])
    output = args.output / args.model / f"{family}_paired_effects.csv"
    existing = read_csv(output) if output.exists() else []
    if existing and not args.resume:
        raise FileExistsError(f"Use --resume or a new --output: {output}")
    done = {
        (r["pair_id"], r["split"], r["direction"], int(r["layer"])) for r in existing
    }
    with output.open("a" if existing else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIRED_FIELDS)
        if not existing:
            writer.writeheader()
        for pair in selected_pairs(args, family):
            metadata = cache[pair["pair_id"]]
            states = runtime.torch.load(
                folder / metadata["tensor_file"], map_location="cpu", weights_only=True
            )
            scheduled = [("confirmation", setting["comparison_block"])]
            if pair["pair_id"] in sweep_ids:
                scheduled += [
                    ("external_confirmation", layer) for layer in range(first, last + 1)
                ]
            for direction, (donor, recipient) in DIRECTIONS.items():
                pending = [
                    (split, layer)
                    for split, layer in scheduled
                    if (pair["pair_id"], split, direction, layer) not in done
                ]
                if not pending:
                    continue
                encoded = encoded_recipient(
                    runtime, registry, pair, recipient, metadata
                )
                donor_margin, recipient_margin = (
                    metadata["clean_margins"][donor],
                    metadata["clean_margins"][recipient],
                )
                measured = {}
                for split, layer in pending:
                    if layer not in measured:
                        measured[layer] = replace(
                            runtime, encoded, states[donor], metadata, layer
                        )
                    patched = measured[layer]
                    writer.writerow(
                        {
                            "model_key": args.model,
                            "family": family,
                            "contrast": pair.get(
                                "low_variant", "decoy-success-vs-failure"
                            ),
                            "split": split,
                            "pair_id": pair["pair_id"],
                            "cluster_id": pair["cluster_id"],
                            "direction": direction,
                            "layer": layer,
                            "clean_donor_margin": donor_margin,
                            "clean_recipient_margin": recipient_margin,
                            "clean_gap_abs": abs(donor_margin - recipient_margin),
                            "patched_margin": patched,
                            "donor_aligned_effect": donor_aligned_effect(
                                donor_margin=donor_margin,
                                recipient_margin=recipient_margin,
                                patched_margin=patched,
                            ),
                        }
                    )
                    stream.flush()
                print(f"patch {args.model} {pair['pair_id']} {direction}", flush=True)


def controls(args, runtime, family):
    folder = cache_dir(args, family)
    cache = {r["pair_id"]: r for r in read_jsonl(folder / "index.jsonl")}
    pairs = {p["pair_id"]: p for p in selected_pairs(args, family)}
    registry = load_registry(family, args.data)
    assignments = [
        r
        for r in read_csv(args.donors)
        if r["model"] == args.model and r["family"] == family
    ]
    output = args.output / args.model / f"{family}_control_effects.csv"
    existing = read_csv(output) if output.exists() else []
    if existing and not args.resume:
        raise FileExistsError(f"Use --resume or a new --output: {output}")
    done = {(r["pair_id"], r["control"], r["direction"]) for r in existing}
    with output.open("a" if existing else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONTROL_FIELDS)
        if not existing:
            writer.writeheader()
        for assignment in assignments:
            recipient_id, donor_id = (
                assignment["recipient_pair_id"],
                assignment["donor_pair_id"],
            )
            pair, metadata, donor_metadata = (
                pairs[recipient_id],
                cache[recipient_id],
                cache[donor_id],
            )
            if any(
                metadata[k] != donor_metadata[k]
                for k in ("lcp_ids", "high_token_id", "low_token_id")
            ):
                raise ValueError(
                    f"Donor action tokens differ: {recipient_id}, {donor_id}"
                )
            states = runtime.torch.load(
                folder / donor_metadata["tensor_file"],
                map_location="cpu",
                weights_only=True,
            )
            for direction, (paired_donor, recipient) in DIRECTIONS.items():
                if (recipient_id, assignment["control"], direction) in done:
                    continue
                encoded = encoded_recipient(
                    runtime, registry, pair, recipient, metadata
                )
                # An external donor supplies the recipient's own condition. The sign
                # still refers to the paired experimental condition change.
                patched = replace(runtime, encoded, states[recipient], metadata, 35)
                donor_margin, recipient_margin = (
                    metadata["clean_margins"][paired_donor],
                    metadata["clean_margins"][recipient],
                )
                writer.writerow(
                    {
                        "family": family,
                        "model_key": args.model,
                        "pair_id": recipient_id,
                        "cluster_id": pair["cluster_id"],
                        "control": assignment["control"],
                        "direction": direction,
                        "activation_donor_pair_id": donor_id,
                        "clean_recipient_margin": recipient_margin,
                        "clean_reference_paired_donor_margin": donor_margin,
                        "patched_margin": patched,
                        "donor_aligned_effect": donor_aligned_effect(
                            donor_margin=donor_margin,
                            recipient_margin=recipient_margin,
                            patched_margin=patched,
                        ),
                        "clean_gap_abs": abs(donor_margin - recipient_margin),
                    }
                )
                stream.flush()
            print(
                f"control {args.model} {family} {assignment['control']} {recipient_id}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capture", "patch", "controls", "all"])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--family", choices=["both", "structural", "semantic"], default="both"
    )
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs/stage1/pairs")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--cohort", type=Path, default=DATA / "cohorts/stage2.json")
    parser.add_argument(
        "--donors", type=Path, default=DATA / "cohorts/control_donors.csv"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and (
        args.limit < 1 or args.command in {"all", "controls"}
    ):
        parser.error(
            "Use a positive --limit with capture or patch; donor comparisons require the full cohort"
        )
    runtime = model_runtime(args.model, args.model_dir)
    commands = (
        [capture, patch, controls]
        if args.command == "all"
        else [{"capture": capture, "patch": patch, "controls": controls}[args.command]]
    )
    for family in (
        ["structural", "semantic"] if args.family == "both" else [args.family]
    ):
        for command in commands:
            command(args, runtime, family)


if __name__ == "__main__":
    main()
