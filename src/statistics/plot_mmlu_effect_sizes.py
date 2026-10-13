import csv
import os

import matplotlib.pyplot as plt


INPUT_PATH = (
    "results/statistics/"
    "mmlu_main_inference.csv"
)

OUTPUT_PATH = (
    "results/figures/"
    "mmlu_effect_sizes.png"
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

    # Focus the inferential figure on
    # the primary "number of shots" question.
    rows = [
        row
        for row in rows
        if row["family"]
        == "Shots effect"
    ]

    comparison_order = [
        "0-shot vs 1-shot random",
        "0-shot vs 3-shot random",
        "0-shot vs 5-shot random",
    ]

    plot_rows = []

    for model in MODEL_ORDER:

        for comparison in comparison_order:

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
                row["effect_b_minus_a"]
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
                    "0-shot vs ",
                    ""
                )
                .replace(
                    " random",
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

    # Reverse so Qwen appears at top.
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
        figsize=(10, 6.5)
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

        significance = (
            " *"
            if row["significant"]
            else ""
        )

        labels.append(
            row["label"]
            + significance
        )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Accuracy difference vs 0-shot"
    )

    ax.set_title(
        "MMLU-Pro effect of random in-context demonstrations"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Convert x-axis labels visually
    # to percentage-point style.
    ticks = ax.get_xticks()

    ax.set_xticklabels(
        [
            f"{tick * 100:+.0f}"
            for tick in ticks
        ]
    )

    ax.set_xlabel(
        "Change in accuracy (percentage points)"
    )

    ax.text(
        0.01,
        -0.12,
        "* Holm-adjusted p < 0.05. Error bars: 95% paired bootstrap CI.",
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

    print()

    for row in reversed(
        plot_rows
    ):

        print(
            f"{row['label']}: "
            f"{row['effect'] * 100:+.1f} pp | "
            f"95% CI "
            f"[{row['ci_low'] * 100:+.1f}, "
            f"{row['ci_high'] * 100:+.1f}] | "
            f"Holm p="
            f"{row['p_holm']:.6f} | "
            f"significant="
            f"{row['significant']}"
        )


if __name__ == "__main__":
    main()