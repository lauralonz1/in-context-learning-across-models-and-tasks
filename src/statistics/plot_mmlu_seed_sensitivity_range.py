import csv
import os

import matplotlib.pyplot as plt


INPUT_PATH = (
    "results/statistics/"
    "mmlu_seed_sensitivity.csv"
)

OUTPUT_PATH = (
    "results/figures/"
    "mmlu_seed_sensitivity_range.png"
)

MODEL_ORDER = [
    "Qwen",
    "Granite",
    "Gemma",
]

SHOT_ORDER = [
    1,
    3,
    5,
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

    lookup = {
        (
            row["model_label"],
            int(row["shots"]),
        ): row
        for row in rows
    }

    labels = []
    means = []
    mins = []
    maxs = []
    ranges = []

    for model in MODEL_ORDER:

        for shots in SHOT_ORDER:

            row = lookup[
                (
                    model,
                    shots,
                )
            ]

            labels.append(
                f"{model} — {shots}-shot"
            )

            means.append(
                float(
                    row[
                        "mean_accuracy"
                    ]
                )
            )

            mins.append(
                float(
                    row[
                        "min_accuracy"
                    ]
                )
            )

            maxs.append(
                float(
                    row[
                        "max_accuracy"
                    ]
                )
            )

            ranges.append(
                float(
                    row[
                        "range_pp"
                    ]
                )
            )

    labels = list(
        reversed(
            labels
        )
    )

    means = list(
        reversed(
            means
        )
    )

    mins = list(
        reversed(
            mins
        )
    )

    maxs = list(
        reversed(
            maxs
        )
    )

    ranges = list(
        reversed(
            ranges
        )
    )

    y = list(
        range(
            len(labels)
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    for i in range(
        len(labels)
    ):

        ax.hlines(
            y=i,
            xmin=mins[i],
            xmax=maxs[i],
            linewidth=2.5,
        )

        ax.plot(
            means[i],
            i,
            marker="o",
            markersize=7,
        )

        ax.text(
            maxs[i] + 0.008,
            i,
            f"{ranges[i]:.1f} pp",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Accuracy"
    )

    ax.set_title(
        "MMLU-Pro robustness to random demonstration seed"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.text(
        0.01,
        -0.12,
        "Horizontal segment = min–max accuracy across seeds 42/43/44; "
        "dot = mean accuracy.",
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