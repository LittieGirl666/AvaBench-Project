"""Paper summaries from case, replay-pair, and intervention records."""

from __future__ import annotations

import pandas as pd


def behavioral(cases):
    keys = ["family", "model", "variant_id"]
    summary = (
        cases.groupby(keys, sort=False)
        .agg(
            cases=("case_id", "size"),
            correct=("behavior_correct", "sum"),
            mean_model_turns=("model_turns", "mean"),
            median_model_turns=("model_turns", "median"),
            mean_environment_calls=("environment_calls", "mean"),
        )
        .reset_index()
    )
    summary["accuracy_pct"] = 100 * summary.correct / summary.cases
    terminals = cases.copy()
    terminals["residual_wrong_state"] = (
        terminals.residual_wrong_state_at_termination.fillna(-1).map(
            {-1: "N/A", 0: "No", 1: "Yes"}
        )
    )
    terminal = (
        terminals.groupby(
            keys + ["termination_reason", "residual_wrong_state"], sort=False
        )
        .size()
        .rename("count")
        .reset_index()
    )
    terminal = terminal.merge(summary[keys + ["cases"]], on=keys)
    terminal["pct"] = 100 * terminal["count"] / terminal.cases
    return summary, terminal


def stage1_summary(pairs):
    rows = []
    for (model, family, contrast), group in pairs.groupby(
        ["model", "family", "contrast"], sort=False
    ):
        structural = family == "structural"
        before = group.high_margin if structural else group.low_margin
        after = group.low_margin if structural else group.high_margin
        following = int(
            (group.effect > 0).sum() if structural else (group.effect < 0).sum()
        )
        rows.append(
            dict(
                model=model,
                family=family,
                contrast=contrast,
                pairs=len(group),
                margin_before=before.mean(),
                margin_after=after.mean(),
                mean_change=group.effect.mean(),
                following_direction=following,
                following_direction_pct=100 * following / len(group),
            )
        )
    gap = []
    for model, group in pairs[pairs.contrast == "sub-operations-no-last"].groupby(
        "model", sort=False
    ):
        exhausted = group[group.phenotype == "budget_exhausted"]
        gap.append(
            dict(
                model=model,
                pairs=len(group),
                exhausted=len(exhausted),
                exhausted_with_increase=int((exhausted.effect > 0).sum()),
                exhausted_with_negative_margin=int((exhausted.low_margin < 0).sum()),
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(gap)


def group_weighted(group):
    # First average repeats (directions and/or blocks) within each pair.
    pairs = group.groupby(["cluster_id", "pair_id"])[
        ["donor_aligned_effect", "clean_gap_abs"]
    ].mean()
    clusters = pairs.groupby("cluster_id").mean()
    effect, gap = clusters.mean()
    return dict(
        pairs=len(pairs),
        task_groups=len(clusters),
        effect=float(effect),
        clean_gap=float(gap),
        recovery_pct=100 * float(effect) / float(gap),
    )


def stage2_summary(paired, controls):
    sweep = paired[paired.split == "external_confirmation"]
    layers, bands, sentinel = [], [], {}
    for key, group in sweep.groupby(
        ["model_key", "family", "direction", "layer"], sort=False
    ):
        layers.append(
            dict(
                zip(["model", "family", "direction", "layer"], key),
                **group_weighted(group),
            )
        )
    for (model, family), group in sweep.groupby(["model_key", "family"], sort=False):
        bands.append(
            dict(
                model=model,
                family=family,
                first_block=int(group.layer.min()),
                last_block=int(group.layer.max()),
                **group_weighted(group),
            )
        )
    for key, group in paired[
        (paired.split == "confirmation") & (paired.layer == 35)
    ].groupby(["model_key", "family", "direction"]):
        sentinel[key] = group_weighted(group)["effect"]
    ratios = []
    for (model, family, control, direction), group in controls.groupby(
        ["model_key", "family", "control", "direction"], sort=False
    ):
        measured = group_weighted(group)
        denominator = sentinel[model, family, direction]
        ratios.append(
            dict(
                model=model,
                family=family,
                control=control,
                direction=direction,
                pairs=measured["pairs"],
                effect=measured["effect"],
                paired_effect=denominator,
                fraction_of_paired_pct=100 * abs(measured["effect"]) / abs(denominator),
            )
        )
    return pd.DataFrame(layers), pd.DataFrame(bands), pd.DataFrame(ratios)
