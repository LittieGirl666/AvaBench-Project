"""Recreate matched and random donor assignments from captured replay metadata."""

import argparse
from pathlib import Path

from common import ROOT, MODELS, read_jsonl, write_csv
from stage2.assignment import assign_control_donors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", *MODELS], default="both")
    parser.add_argument("--captures", type=Path, default=ROOT / "outputs/stage2")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs/stage1/pairs")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/stage2/control_donors.csv"
    )
    args = parser.parse_args()
    rows = []
    for model in MODELS if args.model == "both" else [args.model]:
        for family in ("structural", "semantic"):
            cached = read_jsonl(
                args.captures / model / "activations" / family / "index.jsonl"
            )
            ids = {r["pair_id"] for r in cached}
            pair_file = args.pairs / (
                f"{model}/structural.jsonl"
                if family == "structural"
                else "semantic.jsonl"
            )
            pairs = [p for p in read_jsonl(pair_file) if p["pair_id"] in ids]
            lookup = {p["pair_id"]: p for p in pairs}
            metadata, endpoints = {}, {}
            for r in cached:
                pair = lookup[r["pair_id"]]
                endpoints[r["pair_id"]] = {
                    **{
                        k: r[k]
                        for k in [
                            "lcp_ids",
                            "high_candidate_ids",
                            "low_candidate_ids",
                            "high_action",
                            "low_action",
                        ]
                    },
                    "high_divergence_token_id": r["high_token_id"],
                    "low_divergence_token_id": r["low_token_id"],
                }
                for c in pair["contexts"]:
                    metadata[r["pair_id"], c["label"]] = {
                        "augmented_token_count": r["prompt_tokens"][c["label"]]
                        + len(r["lcp_ids"]),
                        "tool_count": len(c["visible_operation_ids"]) + 3,
                        "candidate_count": len(pair["candidate_universe"]),
                    }
            for control in ("same_condition_token_matched", "token_stratified_random"):
                assigned, excluded = assign_control_donors(
                    pairs=pairs,
                    metadata=metadata,
                    endpoints=endpoints,
                    control=control,
                    seed=0,
                    donor_capacity=2,
                )
                if excluded:
                    raise ValueError(
                        "Capture the full paper cohort before assigning cross-task donors"
                    )
                rows.extend(
                    {
                        "model": model,
                        "family": family,
                        **{
                            k: r[k]
                            for k in ["control", "recipient_pair_id", "donor_pair_id"]
                        },
                    }
                    for r in assigned
                )
    write_csv(args.output, rows)
    print(f"{len(rows)} donor assignments → {args.output}")


if __name__ == "__main__":
    main()
