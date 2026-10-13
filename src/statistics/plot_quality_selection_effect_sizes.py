import csv
import os

import matplotlib.pyplot as plt


INPUT_PATH = (
    "results/statistics/"
    "quality_main_inference.csv"
)

OUTPUT_PATH = (
    "results/figures/"
    "quality_selection_effect_sizes.png"
)

MODEL_LABELS = {
    "qwen": "Qwen",
    "granite": "Granite",
    "gemma": "Gemma",
}

MODEL_ORDER = [
    "qwen",
    "granite",
    "gemma",
]

COMPARISON_ORDER = [
    "1-shot random vs semantic",
    "3-shot random vs semantic",
]


def load_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def main():

    os.makedirs(
        "results/figures",
        exist_ok=True,
    )

    rows = load_csv(
        INPUT_PATH
    )

    rows = [
        row
        for row in rows
        if row["family"]
        == "Selection effect"
    ]

    plot_rows = []

    for model in MODEL_ORDER:

        for comparison in COMPARISON_ORDER:

            matches = [
                row
                for row in rows
                if (
                    row["model"] == model
                    and
                    row["comparison"]
                    == comparison
                )
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one row for "
                    f"{model} | {comparison}, "
                    f"found {len(matches)}"
                )

            row = matches[0]

            effect = float(
                row[
                    "mean_delta_b_minus_a"
                ]
            )

            ci_low = float(
                row["effect_ci_low"]
            )

            ci_high = float(
                row["effect_ci_high"]
            )

            p_holm = float(
                row["p_value_holm"]
            )

            significant = (
                row["reject_h0"].lower()
                == "true"
            )

            shot_label = (
                comparison
                .replace(
                    " random vs semantic",
                    ""
                )
            )

            plot_rows.append(
                {
                    "label": (
                        f"{MODEL_LABELS[model]} — "
                        f"{shot_label}"
                    ),
                    "effect": effect,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_holm": p_holm,
                    "significant": significant,
                }
            )

    plot_rows = list(
        reversed(
            plot_rows
        )
    )

    y = list(
        range(
            len(plot_rows)
        )
    )

    effects = [
        row["effect"]
        for row in plot_rows
    ]

    lower_errors = [
        row["effect"]
        - row["ci_low"]
        for row in plot_rows
    ]

    upper_errors = [
        row["ci_high"]
        - row["effect"]
        for row in plot_rows
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    ax.errorbar(
        effects,
        y,
        xerr=[
            lower_errors,
            upper_errors,
        ],
        fmt="o",
        capsize=4,
        markersize=7,
    )

    ax.axvline(
        0,
        linewidth=1.2,
        linestyle="--",
    )

    ax.set_yticks(
        y
    )

    labels = []

    for row in plot_rows:

        suffix = (
            " *"
            if row["significant"]
            else ""
        )

        labels.append(
            row["label"]
            + suffix
        )

    ax.set_yticklabels(
        labels
    )

    ax.set_title(
        "QuALITY effect of semantic vs random demonstration selection"
    )

    ax.set_xlabel(
        "Semantic − random accuracy difference (percentage points)"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ticks = ax.get_xticks()

    ax.set_xticks(
        ticks
    )

    ax.set_xticklabels(
        [
            f"{tick * 100:+.0f}"
            for tick in ticks
        ]
    )

    ax.text(
        0.01,
        -0.14,
        "* Holm-adjusted p < 0.05. "
        "Error bars: 95% cluster bootstrap CI.",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved figure to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()