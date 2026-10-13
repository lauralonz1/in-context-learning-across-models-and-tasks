import csv
import glob
import json
import os
import statistics
from collections import defaultdict


SPLIT_PATH = (
    "data/splits/"
    "mmlu_pro_sensitivity.jsonl"
)

OUTPUT_PATH = (
    "results/statistics/"
    "mmlu_seed_sensitivity.csv"
)

SENSITIVITY_FILES = {
    "qwen3_30b_a3b_instruct_2507": (
        "results/raw_predictions/"
        "mmlu_pro_sensitivity_qwen.jsonl"
    ),
    "granite4_micro": (
        "results/raw_predictions/"
        "mmlu_pro_sensitivity_granite.jsonl"
    ),
    "gemma3_4b": (
        "results/raw_predictions/"
        "mmlu_pro_sensitivity_gemma.jsonl"
    ),
}

MODEL_LABELS = {
    "qwen3_30b_a3b_instruct_2507": "Qwen",
    "granite4_micro": "Granite",
    "gemma3_4b": "Gemma",
}

SHOTS = [
    1,
    3,
    5,
]

SEEDS = [
    42,
    43,
    44,
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


def record_key(record):
    return (
        record["item_id"],
        record["model"],
        record["shots"],
        record["strategy"],
        record["seed"],
    )


def keep_best_record(
    storage,
    record,
):
    """
    For duplicate keys:
    prefer a successful record over
    an error record.
    """

    key = record_key(record)

    if key not in storage:
        storage[key] = record
        return

    old = storage[key]

    if (
        old["error"] is not None
        and record["error"] is None
    ):
        storage[key] = record


def accuracy(records):
    if not records:
        return None

    return (
        sum(
            1
            for record in records
            if record["correct"] is True
        )
        / len(records)
    )


def main():

    # ==========================================================
    # LOAD FIXED SENSITIVITY SUBSET
    # ==========================================================

    subset = load_jsonl(
        SPLIT_PATH
    )

    subset_ids = {
        item["item_id"]
        for item in subset
    }

    print("=" * 78)
    print("MMLU-PRO SEED SENSITIVITY ANALYSIS")
    print("=" * 78)

    print(
        f"Sensitivity items: "
        f"{len(subset_ids)}"
    )

    if len(subset_ids) != 200:
        raise ValueError(
            "Expected exactly 200 "
            "unique sensitivity items."
        )

    # ==========================================================
    # FIND MAIN/FULL MMLU FILES
    # seed 42 comes from the already-completed
    # primary experiment.
    # ==========================================================

    full_files = [
        path
        for path in glob.glob(
            "results/raw_predictions/*.jsonl"
        )
        if (
            "mmlu" in os.path.basename(path).lower()
            and
            "full" in os.path.basename(path).lower()
        )
    ]

    print(
        f"Detected MMLU full files: "
        f"{len(full_files)}"
    )

    if not full_files:
        raise FileNotFoundError(
            "No MMLU full result files found."
        )

    # ==========================================================
    # COLLECT VALID RECORDS
    # ==========================================================

    records_by_key = {}

    # Seed 42 from main experiment
    for path in full_files:

        for record in load_jsonl(path):

            if (
                record.get("item_id")
                not in subset_ids
            ):
                continue

            if (
                record.get("model")
                not in MODEL_LABELS
            ):
                continue

            if (
                record.get("strategy")
                != "random"
            ):
                continue

            if (
                record.get("shots")
                not in SHOTS
            ):
                continue

            if (
                record.get("seed")
                != 42
            ):
                continue

            keep_best_record(
                records_by_key,
                record,
            )

    # Seeds 43 and 44 from sensitivity
    for model, path in (
        SENSITIVITY_FILES.items()
    ):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing sensitivity file: "
                f"{path}"
            )

        for record in load_jsonl(path):

            if (
                record.get("item_id")
                not in subset_ids
            ):
                continue

            if (
                record.get("model")
                != model
            ):
                continue

            if (
                record.get("strategy")
                != "random"
            ):
                continue

            if (
                record.get("shots")
                not in SHOTS
            ):
                continue

            if (
                record.get("seed")
                not in [43, 44]
            ):
                continue

            keep_best_record(
                records_by_key,
                record,
            )

    # ==========================================================
    # CHECK COMPLETENESS
    # ==========================================================

    expected = (
        len(MODEL_LABELS)
        * len(SHOTS)
        * len(SEEDS)
        * len(subset_ids)
    )

    valid_records = [
        record
        for record in records_by_key.values()
        if record["error"] is None
    ]

    print(
        f"Expected valid records: "
        f"{expected}"
    )

    print(
        f"Found valid records: "
        f"{len(valid_records)}"
    )

    if len(valid_records) != expected:

        print(
            "\nWARNING: expected and found "
            "record counts differ."
        )

    # ==========================================================
    # GROUP
    # ==========================================================

    grouped = defaultdict(list)

    for record in valid_records:

        key = (
            record["model"],
            record["shots"],
            record["seed"],
        )

        grouped[key].append(
            record
        )

    # ==========================================================
    # ANALYSIS
    # ==========================================================

    rows = []

    for model in MODEL_LABELS:

        print()
        print("=" * 78)
        print(
            f"MODEL: "
            f"{MODEL_LABELS[model]}"
        )
        print("=" * 78)

        for shots in SHOTS:

            seed_accuracies = []

            print()
            print(
                f"{shots}-SHOT"
            )
            print("-" * 78)

            seed_correctness = {}

            for seed in SEEDS:

                records = grouped[
                    (
                        model,
                        shots,
                        seed,
                    )
                ]

                if len(records) != 200:
                    raise ValueError(
                        f"{model} | "
                        f"{shots}-shot | "
                        f"seed={seed}: "
                        f"expected 200 valid records, "
                        f"found {len(records)}"
                    )

                acc = accuracy(
                    records
                )

                seed_accuracies.append(
                    acc
                )

                seed_correctness[seed] = {
                    record["item_id"]:
                    bool(record["correct"])
                    for record in records
                }

                print(
                    f"seed={seed}: "
                    f"{acc:.3f} "
                    f"({int(round(acc * 200))}/200)"
                )

            mean_acc = statistics.mean(
                seed_accuracies
            )

            sd_acc = statistics.stdev(
                seed_accuracies
            )

            min_acc = min(
                seed_accuracies
            )

            max_acc = max(
                seed_accuracies
            )

            range_acc = (
                max_acc
                - min_acc
            )

            # Fraction of items for which
            # correctness status is identical
            # across all three demo seeds.
            all_seed_agreement = sum(
                1
                for item_id in subset_ids
                if len(
                    {
                        seed_correctness[seed][
                            item_id
                        ]
                        for seed in SEEDS
                    }
                )
                == 1
            ) / len(subset_ids)

            print()
            print(
                f"mean accuracy: "
                f"{mean_acc:.3f}"
            )

            print(
                f"SD across seeds: "
                f"{sd_acc:.3f}"
            )

            print(
                f"range: "
                f"{min_acc:.3f} - "
                f"{max_acc:.3f}"
            )

            print(
                f"max-min: "
                f"{range_acc * 100:.1f} pp"
            )

            print(
                f"item correctness agreement "
                f"across seeds: "
                f"{all_seed_agreement:.3f}"
            )

            rows.append(
                {
                    "model": model,
                    "model_label": (
                        MODEL_LABELS[
                            model
                        ]
                    ),
                    "shots": shots,
                    "seed42_accuracy": (
                        seed_accuracies[0]
                    ),
                    "seed43_accuracy": (
                        seed_accuracies[1]
                    ),
                    "seed44_accuracy": (
                        seed_accuracies[2]
                    ),
                    "mean_accuracy": (
                        mean_acc
                    ),
                    "sd_across_seeds": (
                        sd_acc
                    ),
                    "min_accuracy": (
                        min_acc
                    ),
                    "max_accuracy": (
                        max_acc
                    ),
                    "range_accuracy": (
                        range_acc
                    ),
                    "range_pp": (
                        range_acc * 100
                    ),
                    "item_correctness_agreement": (
                        all_seed_agreement
                    ),
                }
            )

    # ==========================================================
    # SAVE CSV
    # ==========================================================

    os.makedirs(
        "results/statistics",
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "model_label",
        "shots",
        "seed42_accuracy",
        "seed43_accuracy",
        "seed44_accuracy",
        "mean_accuracy",
        "sd_across_seeds",
        "min_accuracy",
        "max_accuracy",
        "range_accuracy",
        "range_pp",
        "item_correctness_agreement",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()