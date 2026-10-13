import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


RESULT_FILES = {
    "Qwen": (
        "results/raw_predictions/"
        "quality_full_qwen.jsonl"
    ),
    "Granite": (
        "results/raw_predictions/"
        "quality_full_granite.jsonl"
    ),
    "Gemma": (
        "results/raw_predictions/"
        "quality_full_gemma.jsonl"
    ),
}

OUTPUT_PATH = (
    "results/figures/"
    "quality_main_accuracy.png"
)

CONDITION_ORDER = [
    (0, "none", None, "0-shot"),
    (1, "random", 42, "1-shot random"),
    (1, "semantic", None, "1-shot semantic"),
    (3, "random", 42, "3-shot random"),
    (3, "semantic", None, "3-shot semantic"),
]


def load_jsonl(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    os.makedirs(
        "results/figures",
        exist_ok=True,
    )

    model_condition_accuracy = defaultdict(dict)

    for model_label, path in (
        RESULT_FILES.items()
    ):

        records = load_jsonl(
            path
        )

        for shots, strategy, seed, label in (
            CONDITION_ORDER
        ):

            selected = [
                record
                for record in records
                if (
                    record["error"] is None
                    and record["shots"] == shots
                    and record["strategy"] == strategy
                    and record["seed"] == seed
                )
            ]

            if not selected:
                raise ValueError(
                    f"No records found for "
                    f"{model_label} | {label}"
                )

            accuracy = (
                sum(
                    1
                    for record in selected
                    if record["correct"] is True
                )
                / len(selected)
            )

            model_condition_accuracy[
                model_label
            ][label] = accuracy

    x_labels = [
        condition[3]
        for condition in CONDITION_ORDER
    ]

    x = list(
        range(
            len(x_labels)
        )
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    model_names = list(
        RESULT_FILES.keys()
    )

    offsets = [
        -width,
        0,
        width,
    ]

    for model_label, offset in zip(
        model_names,
        offsets,
    ):

        values = [
            model_condition_accuracy[
                model_label
            ][label]
            for label in x_labels
        ]

        bars = ax.bar(
            [
                value + offset
                for value in x
            ],
            values,
            width=width,
            label=model_label,
        )

        for bar, value in zip(
            bars,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height()
                + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title(
        "QuALITY accuracy across in-context learning conditions"
    )

    ax.set_xlabel(
        "Condition"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        x_labels,
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        0.85,
    )

    ax.legend(
        title="Model"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
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

    for model_label in model_names:

        print()
        print(
            model_label.upper()
        )

        for label in x_labels:

            print(
                f"{label}: "
                f"{model_condition_accuracy[model_label][label]:.3f}"
            )


if __name__ == "__main__":
    main()