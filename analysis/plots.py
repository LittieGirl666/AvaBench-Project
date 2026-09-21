"""Render accuracy, interaction length, and residual-transfer plots."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import STRUCTURAL, SEMANTIC

API_MODELS = {
    "glm-5.2": ("GLM 5.2", "#79A8CC"),
    "deepseek-v4-flash": ("DeepSeek V4 Flash", "#DCA073"),
    "qwen3.5-27b": ("Qwen3.5-27B", "#80B39C"),
}
LOCAL_MODELS = {
    "qwen3-8b": ("Qwen3-8B", "#176B91"),
    "qwen3-4b": ("Qwen3-4B", "#C96936"),
}
LABELS = {
    "baseline": "Direct",
    "no-gold": "No parent",
    "sub-operations": "Full chain",
    "sub-operations-no-gold": "No chain",
    "sub-operations-no-first": "No first",
    "sub-operations-first-only": "First only",
    "sub-operations-no-last": "No last",
    "state-gold-only": "Gold only",
    "state-gold-decoy": "Gold + decoy",
    "state-decoy-failure": "Decoy fails",
    "state-decoy-success": "Decoy succeeds",
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def behavioral_plot(cases, family, path):
    variants = STRUCTURAL if family == "structural" else SEMANTIC
    data = cases[cases.family == family]
    models = [m for m in API_MODELS if m in set(data.model)]
    models += sorted(set(data.model) - set(models))
    palette = {
        m: API_MODELS.get(m, (m, plt.get_cmap("tab10")(i % 10)))
        for i, m in enumerate(models)
    }
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.25]}
        )
        seen = set()
        for i, model in enumerate(models):
            name, color = palette[model]
            groups = [
                data[(data.model == model) & (data.variant_id == v)] for v in variants
            ]
            rates = [100 * g.behavior_correct.mean() for g in groups]
            axes[0].plot(
                range(len(variants)), rates, color=color, marker="o", label=name
            )
            for j, value in enumerate(rates):
                if not np.isfinite(value) or (j, round(value, 8)) in seen:
                    continue
                seen.add((j, round(value, 8)))
                axes[0].annotate(
                    f"{value:.1f}" if value != 100 else "100",
                    (j, value),
                    xytext=(0, 7 if i % 2 == 0 else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            for j, group in enumerate(groups):
                values = group.model_turns.to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                x = i * (len(variants) + 1) + j
                if values.min() == values.max():
                    axes[1].hlines(
                        values[0], x - 0.3, x + 0.3, color=color, linewidth=3
                    )
                else:
                    grid = np.linspace(values.min(), values.max(), 220)
                    density = np.exp(
                        -0.5 * ((grid[:, None] - values[None, :]) / 0.28) ** 2
                    ).mean(axis=1)
                    width = 0.38 * density / density.max()
                    axes[1].fill_betweenx(
                        grid, x - width, x + width, color=color, alpha=0.55
                    )
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                axes[1].vlines(x, q1, q3, color="#52616B", lw=2)
                axes[1].scatter(
                    x, med, s=12, color="white", edgecolors="#52616B", zorder=4
                )
                axes[1].scatter(
                    x + 0.10, values.mean(), marker="D", s=10, color="#344551", zorder=5
                )
        axes[0].set(
            xticks=range(len(variants)),
            xticklabels=[LABELS[v] for v in variants],
            ylabel="Correct outcome (%)",
            ylim=(0, 110),
            title="Judgment accuracy",
        )
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].legend(frameon=False, fontsize=8, loc="lower left")
        axes[1].set(
            xticks=[
                i * (len(variants) + 1) + (len(variants) - 1) / 2
                for i in range(len(models))
            ],
            xticklabels=[palette[m][0] for m in models],
            yticks=range(1, 9),
            ylabel="Model turns",
            ylim=(0.5, 8.5),
            title="Interaction length",
        )
        axes[1].text(
            0.5,
            -0.22,
            "Within each model: " + " → ".join(LABELS[v] for v in variants),
            transform=axes[1].transAxes,
            ha="center",
            fontsize=7,
        )
        for ax in axes:
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
        fig.tight_layout(w_pad=3)
        save(fig, path)


def recovery_plot(layers, path):
    series = [(m, d) for m in LOCAL_MODELS for d in ("low_to_high", "high_to_low")]
    blocks = range(18, 36)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.4, 1]}
        )
        for row, family in enumerate(("structural", "semantic")):
            heat, trend = axes[row]
            for y, (model, direction) in enumerate(series):
                name, color = LOCAL_MODELS[model]
                data = layers[
                    (layers.model == model)
                    & (layers.family == family)
                    & (layers.direction == direction)
                ].sort_values("layer")
                values = dict(zip(data.layer, data.recovery_pct))
                for x, block in enumerate(blocks):
                    value = values.get(block)
                    shade = (
                        "#ECEEEF"
                        if value is None
                        else plt.matplotlib.colors.to_rgba(
                            color, float(np.clip((value + 5) / 110, 0, 1))
                        )
                    )
                    heat.add_patch(
                        plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=shade, ec="white")
                    )
                    heat.text(
                        x,
                        y,
                        "—" if value is None else str(round(value)),
                        ha="center",
                        va="center",
                        color=(
                            "white" if value is not None and value > 70 else "#25313B"
                        ),
                        fontsize=8,
                    )
                trend.plot(
                    data.layer,
                    data.recovery_pct,
                    label=f"{name} {'L → H' if direction=='low_to_high' else 'H → L'}",
                    color=color,
                    ls="-" if direction == "low_to_high" else "--",
                    marker="o" if direction == "low_to_high" else "s",
                    markersize=3,
                    markerfacecolor=color if direction == "low_to_high" else "white",
                )
            heat.set(
                xlim=(-0.5, 17.5),
                ylim=(3.5, -0.5),
                xticks=range(18),
                xticklabels=list(blocks),
                yticks=range(4),
                yticklabels=[
                    f"{LOCAL_MODELS[m][0]} {'L → H' if d=='low_to_high' else 'H → L'}"
                    for m, d in series
                ],
                title=family.title(),
                xlabel="Decoder block",
            )
            trend.set(
                xlim=(17.5, 35.5),
                ylim=(-5, 110),
                xticks=[18, 21, 24, 27, 30, 33, 35],
                yticks=[0, 25, 50, 75, 100],
                xlabel="Decoder block",
                ylabel="Recovery (%)",
            )
            trend.grid(axis="y", alpha=0.2)
            trend.axhline(100, ls=":", lw=0.7, color="gray")
        axes[0, 1].legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout(w_pad=3)
        save(fig, path)
