"""Rebuild the paper's descriptive tables and figures from compact results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ROOT, STRUCTURAL, SEMANTIC
from analysis.summaries import behavioral, stage1_summary, stage2_summary
from analysis.plots import behavioral_plot, recovery_plot, LABELS


def load_files(root, pattern):
    files = sorted(Path(root).rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {root}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def save_table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def tex_table(path, headers, rows, align=None):
    def text(value):
        return str(value).replace("_", r"\_").replace("%", r"\%")

    columns = align or "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(text(v) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def mechanism_tables(output, summary, gap):
    main = summary[
        (summary.family == "semantic") | (summary.contrast == "sub-operations-no-last")
    ]
    rows = []
    for _, r in main.sort_values(["family", "model"], ascending=False).iterrows():
        rows.append(
            [
                (
                    "Complete to No last"
                    if r.family == "structural"
                    else "Failure to success"
                ),
                r.model,
                r.pairs,
                f"{r.margin_before:.2f}",
                f"{r.margin_after:.2f}",
                f"{r.mean_change:+.2f}",
                f"{r.following_direction}/{r.pairs}",
            ]
        )
    tex_table(
        output / "table4.tex",
        [
            "Condition change",
            "Model",
            "Pairs",
            r"$M_0$",
            r"$M_1$",
            r"$\Delta M$",
            "Following direction",
        ],
        rows,
        "llrrrrr",
    )
    tex_table(
        output / "table5.tex",
        ["Model", "Exhausted", "Evidence rises", "Margin negative"],
        [
            [
                r.model,
                f"{r.exhausted}/{r.pairs}",
                f"{r.exhausted_with_increase}/{r.exhausted}",
                f"{r.exhausted_with_negative_margin}/{r.exhausted}",
            ]
            for r in gap.itertuples()
        ],
    )
    structural = summary[summary.family == "structural"]
    tex_table(
        output / "table7.tex",
        [
            "Missing condition",
            "Model",
            "Pairs",
            r"$M_0$",
            r"$M_1$",
            r"$\Delta M$",
            "Increasing pairs",
        ],
        [
            [
                LABELS[r.contrast],
                r.model,
                r.pairs,
                f"{r.margin_before:.2f}",
                f"{r.margin_after:.2f}",
                f"{r.mean_change:+.2f}",
                f"{r.following_direction}/{r.pairs}",
            ]
            for r in structural.itertuples()
        ],
        "llrrrrr",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path, default=ROOT / "results/behavior")
    parser.add_argument("--stage1", type=Path, default=ROOT / "results/stage1")
    parser.add_argument("--stage2", type=Path, default=ROOT / "results/stage2")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["behavior", "stage1", "stage2"],
        default=["behavior", "stage1", "stage2"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper")
    args = parser.parse_args()
    tables = args.output / "tables"
    figures = args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    if "behavior" in args.sections:
        cases = load_files(args.behavior, "*_cases.csv")
        cases = cases[
            ((cases.family == "structural") & cases.variant_id.isin(STRUCTURAL))
            | ((cases.family == "semantic") & cases.variant_id.isin(SEMANTIC))
        ]
        summary, terminal = behavioral(cases)
        save_table(tables / "behavior_summary", summary)
        save_table(tables / "terminal_states", terminal)
        for family, number in [("structural", 2), ("semantic", 3)]:
            if family not in set(cases.family):
                continue
            behavioral_plot(cases, family, figures / f"figure{number}")
            rows = []
            for r in summary[summary.family == family].itertuples():
                subset = terminal[
                    (terminal.model == r.model) & (terminal.variant_id == r.variant_id)
                ]
                counts = {
                    t: int(subset[subset.termination_reason == t]["count"].sum())
                    for t in ["finish", "unavailable", "budget_exhausted"]
                }
                wrong = int(subset[subset.residual_wrong_state == "Yes"]["count"].sum())
                rows.append(
                    [
                        r.model,
                        LABELS[r.variant_id],
                        r.cases,
                        f"{r.accuracy_pct:.2f}",
                        counts["finish"],
                        counts["unavailable"],
                        counts["budget_exhausted"],
                        wrong if family == "semantic" else "N/A",
                    ]
                )
            tex_table(
                tables / f"tableB{1 if family=='structural' else 2}.tex",
                [
                    "Model",
                    "Condition",
                    "Cases",
                    r"Accuracy (\%)",
                    "Finish",
                    "Unavailable",
                    "Exhausted",
                    "Wrong state",
                ],
                rows,
                "llrrrrrr",
            )
        print(
            f"Behavior: {len(cases)} interactions, {len(summary)} condition/model summaries"
        )
    if "stage1" in args.sections:
        pairs = load_files(args.stage1, "pair_effects.csv")
        summary, gap = stage1_summary(pairs)
        save_table(tables / "stage1_summary", summary)
        save_table(tables / "recognition_action_gap", gap)
        mechanism_tables(tables, summary, gap)
        print(f"Stage 1: {len(pairs)} paired stopping-margin changes")
    if "stage2" in args.sections:
        paired = load_files(args.stage2, "*paired_effects.csv")
        controls = load_files(args.stage2, "*control_effects.csv")
        layers, bands, ratios = stage2_summary(paired, controls)
        save_table(tables / "stage2_layers", layers)
        save_table(tables / "stage2_bands", bands)
        save_table(tables / "stage2_controls", ratios)
        recovery_plot(layers, figures / "figure4")
        rows = []
        for r in bands.itertuples():

            def span(control):
                values = ratios[
                    (ratios.model == r.model)
                    & (ratios.family == r.family)
                    & (ratios.control == control)
                ].fraction_of_paired_pct
                return f"{values.min():.2f}--{values.max():.2f}"

            rows.append(
                [
                    r.family,
                    r.model,
                    r.pairs,
                    f"L{r.first_block}--{r.last_block}",
                    f"{r.effect:.2f}",
                    f"{r.recovery_pct:.2f}",
                    span("same_condition_token_matched"),
                    span("token_stratified_random"),
                ]
            )
        tex_table(
            tables / "table6.tex",
            [
                "Experiment",
                "Model",
                "Pairs",
                "Blocks",
                "Effect",
                r"Recovery (\%)",
                r"Matched (\%)",
                r"Random (\%)",
            ],
            rows,
            "llrlrrrr",
        )
        print(
            f"Stage 2: {len(paired)} paired interventions, {len(controls)} same-condition interventions"
        )
    print(f"Tables → {tables}\nFigures → {figures}")


if __name__ == "__main__":
    main()
