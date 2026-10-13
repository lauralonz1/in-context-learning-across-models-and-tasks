import csv
import glob
import json
import os
import random
from collections import defaultdict


SPLIT_PATH = (
    "data/splits/"
    "quality_sensitivity.jsonl"
)

OUTPUT_PATH = (
    "results/statistics/"
    "quality_context_ablation.csv"
)

ABLATION_FILES = {
    "qwen3_30b_a3b_instruct_2507": (
        "results/raw_predictions/"
        "quality_context_ablation_qwen.jsonl"
    ),
    "granite4_micro": (
        "results/raw_predictions/"
        "quality_context_ablation_granite.jsonl"
    ),
    "gemma3_4b": (
        "results/raw_predictions/"
        "quality_context_ablation_gemma.jsonl"
    ),
}

MODEL_LABELS = {
    "qwen3_30b_a3b_instruct_2507": "Qwen",
    "granite4_micro": "Granite",
    "gemma3_4b": "Gemma",
}

N_PERMUTATIONS = 100000
N_BOOTSTRAP = 10000
RANDOM_SEED = 42


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


def keep_best_record(
    storage,
    key,
    record,
):
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
    return (
        sum(
            1
            for record in records
            if record["correct"] is True
        )
        / len(records)
    )


def percentile(
    values,
    q,
):
    values = sorted(values)

    if not values:
        return None

    position = (
        (len(values) - 1)
        * q
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        * (1 - fraction)
        +
        values[upper]
        * fraction
    )


def sign_flip_test(
    article_differences,
    n_permutations,
    seed,
):
    rng = random.Random(
        seed
    )

    observed = abs(
        sum(article_differences)
        / len(article_differences)
    )

    count = 0

    for _ in range(
        n_permutations
    ):

        permuted = [
            difference
            * rng.choice(
                [-1, 1]
            )
            for difference
            in article_differences
        ]

        statistic = abs(
            sum(permuted)
            / len(permuted)
        )

        if statistic >= observed:
            count += 1

    return (
        count + 1
    ) / (
        n_permutations + 1
    )


def cluster_bootstrap_ci(
    article_differences,
    n_bootstrap,
    seed,
):
    rng = random.Random(
        seed
    )

    n_articles = len(
        article_differences
    )

    bootstrap_deltas = []

    for _ in range(
        n_bootstrap
    ):

        sample = [
            rng.choice(
                article_differences
            )
            for _ in range(
                n_articles
            )
        ]

        bootstrap_deltas.append(
            sum(sample)
            / len(sample)
        )

    return (
        percentile(
            bootstrap_deltas,
            0.025,
        ),
        percentile(
            bootstrap_deltas,
            0.975,
        ),
    )


def main():

    # ==========================================================
    # LOAD FIXED SUBSET
    # ==========================================================

    subset = load_jsonl(
        SPLIT_PATH
    )

    subset_ids = {
        item["item_id"]
        for item in subset
    }

    metadata = {
        item["item_id"]: {
            "article_id": (
                item["article_id"]
            ),
            "hard": (
                item["hard"]
            ),
        }
        for item in subset
    }

    article_ids = {
        item["article_id"]
        for item in subset
    }

    print("=" * 78)
    print("QUALITY CONTEXT ABLATION ANALYSIS")
    print("=" * 78)

    print(
        f"Items: "
        f"{len(subset_ids)}"
    )

    print(
        f"Articles: "
        f"{len(article_ids)}"
    )

    if len(subset_ids) != 200:
        raise ValueError(
            "Expected 200 items."
        )

    if len(article_ids) != 40:
        raise ValueError(
            "Expected 40 articles."
        )

    # ==========================================================
    # FIND FULL QUALITY FILES
    # ==========================================================

    full_files = [
        path
        for path in glob.glob(
            "results/raw_predictions/*.jsonl"
        )
        if (
            "quality" in os.path.basename(path).lower()
            and
            "full" in os.path.basename(path).lower()
        )
    ]

    print(
        f"Detected QuALITY full files: "
        f"{len(full_files)}"
    )

    if not full_files:
        raise FileNotFoundError(
            "No QuALITY full files found."
        )

    # ==========================================================
    # LOAD FULL CONTEXT BASELINE
    # ==========================================================

    full_records = {}

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

            if record.get("shots") != 0:
                continue

            if (
                record.get("strategy")
                not in [
                    None,
                    "none",
                ]
            ):
                continue

            if record.get("error") is not None:
                continue

            key = (
                record["model"],
                record["item_id"],
            )

            keep_best_record(
                full_records,
                key,
                record,
            )

    # ==========================================================
    # LOAD NO-CONTEXT ABLATION
    # ==========================================================

    no_context_records = {}

    for model, path in (
        ABLATION_FILES.items()
    ):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing ablation file: "
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
                record.get("error")
                is not None
            ):
                continue

            key = (
                model,
                record["item_id"],
            )

            keep_best_record(
                no_context_records,
                key,
                record,
            )

    # ==========================================================
    # ANALYZE EACH MODEL
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

        paired = []

        for item_id in subset_ids:

            full_key = (
                model,
                item_id,
            )

            no_context_key = (
                model,
                item_id,
            )

            if (
                full_key not in full_records
                or
                no_context_key
                not in no_context_records
            ):
                continue

            full_record = (
                full_records[
                    full_key
                ]
            )

            no_context_record = (
                no_context_records[
                    no_context_key
                ]
            )

            paired.append(
                {
                    "item_id": item_id,
                    "article_id": (
                        metadata[item_id][
                            "article_id"
                        ]
                    ),
                    "hard": (
                        metadata[item_id][
                            "hard"
                        ]
                    ),
                    "full_correct": bool(
                        full_record[
                            "correct"
                        ]
                    ),
                    "no_context_correct": bool(
                        no_context_record[
                            "correct"
                        ]
                    ),
                }
            )

        print(
            f"Paired valid items: "
            f"{len(paired)}"
        )

        if len(paired) != 200:
            raise ValueError(
                f"{model}: expected "
                f"200 paired valid items, "
                f"found {len(paired)}"
            )

        full_acc = (
            sum(
                row[
                    "full_correct"
                ]
                for row in paired
            )
            / len(paired)
        )

        no_context_acc = (
            sum(
                row[
                    "no_context_correct"
                ]
                for row in paired
            )
            / len(paired)
        )

        delta = (
            full_acc
            - no_context_acc
        )

        # HARD / STANDARD
        hard_rows = [
            row
            for row in paired
            if row["hard"]
        ]

        standard_rows = [
            row
            for row in paired
            if not row["hard"]
        ]

        hard_full_acc = (
            sum(
                row[
                    "full_correct"
                ]
                for row in hard_rows
            )
            / len(hard_rows)
        )

        hard_no_context_acc = (
            sum(
                row[
                    "no_context_correct"
                ]
                for row in hard_rows
            )
            / len(hard_rows)
        )

        standard_full_acc = (
            sum(
                row[
                    "full_correct"
                ]
                for row in standard_rows
            )
            / len(standard_rows)
        )

        standard_no_context_acc = (
            sum(
                row[
                    "no_context_correct"
                ]
                for row in standard_rows
            )
            / len(standard_rows)
        )

        # ======================================================
        # ARTICLE-LEVEL DIFFERENCES
        # ======================================================

        article_rows = defaultdict(
            list
        )

        for row in paired:
            article_rows[
                row["article_id"]
            ].append(
                row
            )

        article_differences = []

        improved_articles = 0
        worsened_articles = 0
        unchanged_articles = 0

        for article_id, rows_for_article in (
            article_rows.items()
        ):

            if len(rows_for_article) != 5:
                raise ValueError(
                    f"Article {article_id} "
                    f"does not have 5 "
                    f"paired questions."
                )

            full_article_acc = (
                sum(
                    row[
                        "full_correct"
                    ]
                    for row
                    in rows_for_article
                )
                / 5
            )

            no_context_article_acc = (
                sum(
                    row[
                        "no_context_correct"
                    ]
                    for row
                    in rows_for_article
                )
                / 5
            )

            article_delta = (
                full_article_acc
                - no_context_article_acc
            )

            article_differences.append(
                article_delta
            )

            if article_delta > 0:
                improved_articles += 1
            elif article_delta < 0:
                worsened_articles += 1
            else:
                unchanged_articles += 1

        ci_low, ci_high = (
            cluster_bootstrap_ci(
                article_differences=(
                    article_differences
                ),
                n_bootstrap=(
                    N_BOOTSTRAP
                ),
                seed=(
                    RANDOM_SEED
                ),
            )
        )

        p_value = (
            sign_flip_test(
                article_differences=(
                    article_differences
                ),
                n_permutations=(
                    N_PERMUTATIONS
                ),
                seed=(
                    RANDOM_SEED
                ),
            )
        )

        # ======================================================
        # PRINT
        # ======================================================

        print(
            f"Full context accuracy: "
            f"{full_acc:.3f}"
        )

        print(
            f"No context accuracy: "
            f"{no_context_acc:.3f}"
        )

        print(
            f"Context gain: "
            f"{delta:+.3f} "
            f"({delta * 100:+.1f} pp)"
        )

        print(
            f"95% cluster bootstrap CI: "
            f"[{ci_low:+.3f}, "
            f"{ci_high:+.3f}]"
        )

        print(
            f"Sign-flip p-value: "
            f"{p_value:.6f}"
        )

        print(
            f"Articles | "
            f"context better: "
            f"{improved_articles} | "
            f"context worse: "
            f"{worsened_articles} | "
            f"unchanged: "
            f"{unchanged_articles}"
        )

        print(
            f"HARD | "
            f"full={hard_full_acc:.3f} | "
            f"no_context="
            f"{hard_no_context_acc:.3f} | "
            f"delta="
            f"{hard_full_acc - hard_no_context_acc:+.3f}"
        )

        print(
            f"STANDARD | "
            f"full={standard_full_acc:.3f} | "
            f"no_context="
            f"{standard_no_context_acc:.3f} | "
            f"delta="
            f"{standard_full_acc - standard_no_context_acc:+.3f}"
        )

        rows.append(
            {
                "model": model,
                "model_label": (
                    MODEL_LABELS[
                        model
                    ]
                ),
                "n_items": (
                    len(paired)
                ),
                "n_articles": (
                    len(article_rows)
                ),
                "full_context_accuracy": (
                    full_acc
                ),
                "no_context_accuracy": (
                    no_context_acc
                ),
                "context_gain": (
                    delta
                ),
                "context_gain_pp": (
                    delta * 100
                ),
                "ci_low": (
                    ci_low
                ),
                "ci_high": (
                    ci_high
                ),
                "p_value": (
                    p_value
                ),
                "articles_context_better": (
                    improved_articles
                ),
                "articles_context_worse": (
                    worsened_articles
                ),
                "articles_unchanged": (
                    unchanged_articles
                ),
                "hard_full_accuracy": (
                    hard_full_acc
                ),
                "hard_no_context_accuracy": (
                    hard_no_context_acc
                ),
                "hard_context_gain": (
                    hard_full_acc
                    - hard_no_context_acc
                ),
                "standard_full_accuracy": (
                    standard_full_acc
                ),
                "standard_no_context_accuracy": (
                    standard_no_context_acc
                ),
                "standard_context_gain": (
                    standard_full_acc
                    - standard_no_context_acc
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
        "n_items",
        "n_articles",
        "full_context_accuracy",
        "no_context_accuracy",
        "context_gain",
        "context_gain_pp",
        "ci_low",
        "ci_high",
        "p_value",
        "articles_context_better",
        "articles_context_worse",
        "articles_unchanged",
        "hard_full_accuracy",
        "hard_no_context_accuracy",
        "hard_context_gain",
        "standard_full_accuracy",
        "standard_no_context_accuracy",
        "standard_context_gain",
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