import csv
import os

import matplotlib.pyplot as plt


INPUT_PATH = (
    "results/statistics/"
    "quality_context_ablation.csv"
)

OUTPUT_PATH = (
    "results/figures/"
    "quality_context_ablation.png"
)

MODEL_ORDER = [
    "Qwen",
    "Granite",
    "Gemma",
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

    by_model = {
        row["model_label"]: row
        for row in rows
    }

    x = list(
        range(
            len(MODEL_ORDER)
        )
    )

    width = 0.34

    full_values = [
        float(
            by_model[model][
                "full_context_accuracy"
            ]
        )
        for model in MODEL_ORDER
    ]

    no_context_values = [
        float(
            by_model[model][
                "no_context_accuracy"
            ]
        )
        for model in MODEL_ORDER
    ]

    context_gains = [
        float(
            by_model[model][
                "context_gain_pp"
            ]
        )
        for model in MODEL_ORDER
    ]

    ci_lows = [
        float(
            by_model[model][
                "ci_low"
            ]
        )
        * 100
        for model in MODEL_ORDER
    ]

    ci_highs = [
        float(
            by_model[model][
                "ci_high"
            ]
        )
        * 100
        for model in MODEL_ORDER
    ]

    p_values = [
        float(
            by_model[model][
                "p_value"
            ]
        )
        for model in MODEL_ORDER
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6.5)
    )

    bars_full = ax.bar(
        [
            value - width / 2
            for value in x
        ],
        full_values,
        width=width,
        label="Full context",
    )

    bars_no = ax.bar(
        [
            value + width / 2
            for value in x
        ],
        no_context_values,
        width=width,
        label="No context",
    )

    for bar, value in zip(
        bars_full,
        full_values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar, value in zip(
        bars_no,
        no_context_values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for i, model in enumerate(
        MODEL_ORDER
    ):

        y_position = max(
            full_values[i],
            no_context_values[i],
        ) + 0.085

        ax.text(
            i,
            y_position,
            f"+{context_gains[i]:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        "QuALITY context ablation"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        MODEL_ORDER
    )

    ax.set_ylim(
        0,
        0.95,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0.01,
        -0.14,
        "Context gain = full-context accuracy − no-context accuracy. "
        "All gains significant by article-level sign-flip test.",
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

    for i, model in enumerate(
        MODEL_ORDER
    ):

        print(
            f"{model}: "
            f"full={full_values[i]:.3f} | "
            f"no_context={no_context_values[i]:.3f} | "
            f"gain={context_gains[i]:+.1f} pp | "
            f"95% CI "
            f"[{ci_lows[i]:+.1f}, "
            f"{ci_highs[i]:+.1f}] pp | "
            f"p={p_values[i]:.6f}"
        )


if __name__ == "__main__":
    main()