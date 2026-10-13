import csv
import json
import os

import numpy as np
from statsmodels.stats.multitest import multipletests


RESULT_FILES = {
    "qwen": (
        "results/raw_predictions/"
        "quality_full_qwen.jsonl"
    ),
    "granite": (
        "results/raw_predictions/"
        "quality_full_granite.jsonl"
    ),
    "gemma": (
        "results/raw_predictions/"
        "quality_full_gemma.jsonl"
    ),
}

QUALITY_SPLIT_PATH = (
    "data/splits/"
    "quality_full.jsonl"
)

OUTPUT_PATH = (
    "results/statistics/"
    "quality_main_inference.csv"
)

QUESTIONS_PER_ARTICLE = 5

PERMUTATIONS = 100000
BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 42


SHOT_COMPARISONS = [
    {
        "name": "0-shot vs 1-shot random",
        "a": (0, "none", None),
        "b": (1, "random", 42),
    },
    {
        "name": "0-shot vs 3-shot random",
        "a": (0, "none", None),
        "b": (3, "random", 42),
    },
]


SELECTION_COMPARISONS = [
    {
        "name": "1-shot random vs semantic",
        "a": (1, "random", 42),
        "b": (1, "semantic", None),
    },
    {
        "name": "3-shot random vs semantic",
        "a": (3, "random", 42),
        "b": (3, "semantic", None),
    },
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


def build_item_to_article():

    split_records = load_jsonl(
        QUALITY_SPLIT_PATH
    )

    item_to_article = {
        record["item_id"]:
            record["article_id"]
        for record in split_records
    }

    return item_to_article


def build_lookup(records):

    lookup = {}

    for record in records:

        if record["error"] is not None:
            continue

        key = (
            record["item_id"],
            record["shots"],
            record["strategy"],
            record["seed"],
        )

        lookup[key] = bool(
            record["correct"]
        )

    return lookup


def get_article_differences(
    lookup,
    item_to_article,
    condition_a,
    condition_b,
):

    article_a = {}
    article_b = {}

    for item_id, article_id in (
        item_to_article.items()
    ):

        key_a = (
            item_id,
            condition_a[0],
            condition_a[1],
            condition_a[2],
        )

        key_b = (
            item_id,
            condition_b[0],
            condition_b[1],
            condition_b[2],
        )

        if (
            key_a not in lookup
            or key_b not in lookup
        ):
            continue

        article_a.setdefault(
            article_id,
            [],
        ).append(
            int(lookup[key_a])
        )

        article_b.setdefault(
            article_id,
            [],
        ).append(
            int(lookup[key_b])
        )

    common_articles = sorted(
        set(article_a)
        & set(article_b)
    )

    differences = []

    for article_id in common_articles:

        scores_a = article_a[
            article_id
        ]

        scores_b = article_b[
            article_id
        ]

        if (
            len(scores_a)
            != QUESTIONS_PER_ARTICLE
            or
            len(scores_b)
            != QUESTIONS_PER_ARTICLE
        ):
            continue

        accuracy_a = np.mean(
            scores_a
        )

        accuracy_b = np.mean(
            scores_b
        )

        differences.append(
            accuracy_b
            - accuracy_a
        )

    return np.array(
        differences,
        dtype=float,
    )


def sign_flip_permutation_test(
    differences,
    n_permutations,
    seed,
):

    observed = abs(
        np.mean(differences)
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0

    batch_size = 5000
    completed = 0

    while completed < n_permutations:

        current_batch = min(
            batch_size,
            n_permutations
            - completed,
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(
                current_batch,
                len(differences),
            ),
        )

        permuted_means = np.mean(
            signs
            * differences,
            axis=1,
        )

        extreme += np.sum(
            np.abs(
                permuted_means
            )
            >= observed
        )

        completed += (
            current_batch
        )

    p_value = (
        extreme + 1
    ) / (
        n_permutations + 1
    )

    return float(
        p_value
    )


def bootstrap_ci(
    differences,
    n_bootstrap,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    n = len(
        differences
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    lower = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper = np.percentile(
        bootstrap_means,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


def analyze_comparison(
    lookup,
    item_to_article,
    comparison,
):

    differences = (
        get_article_differences(
            lookup=lookup,
            item_to_article=(
                item_to_article
            ),
            condition_a=(
                comparison["a"]
            ),
            condition_b=(
                comparison["b"]
            ),
        )
    )

    if len(differences) == 0:

        raise ValueError(
            "No complete paired "
            "articles found."
        )

    mean_delta = float(
        np.mean(differences)
    )

    median_delta = float(
        np.median(differences)
    )

    p_value = (
        sign_flip_permutation_test(
            differences=(
                differences
            ),
            n_permutations=(
                PERMUTATIONS
            ),
            seed=RANDOM_SEED,
        )
    )

    ci_lower, ci_upper = (
        bootstrap_ci(
            differences=(
                differences
            ),
            n_bootstrap=(
                BOOTSTRAP_SAMPLES
            ),
            seed=RANDOM_SEED,
        )
    )

    improved = int(
        np.sum(
            differences > 0
        )
    )

    worsened = int(
        np.sum(
            differences < 0
        )
    )

    unchanged = int(
        np.sum(
            differences == 0
        )
    )

    return {
        "name":
            comparison["name"],
        "condition_a":
            comparison["a"],
        "condition_b":
            comparison["b"],
        "n_articles":
            len(differences),
        "mean_delta":
            mean_delta,
        "median_delta":
            median_delta,
        "ci_lower":
            ci_lower,
        "ci_upper":
            ci_upper,
        "improved":
            improved,
        "worsened":
            worsened,
        "unchanged":
            unchanged,
        "p_value":
            p_value,
    }


def analyze_family(
    model_name,
    lookup,
    item_to_article,
    family_name,
    comparisons,
):

    results = []

    for comparison in comparisons:

        result = (
            analyze_comparison(
                lookup=lookup,
                item_to_article=(
                    item_to_article
                ),
                comparison=(
                    comparison
                ),
            )
        )

        results.append(
            result
        )

    raw_p_values = [
        result["p_value"]
        for result in results
    ]

    (
        reject,
        adjusted_p_values,
        _,
        _,
    ) = multipletests(
        raw_p_values,
        alpha=0.05,
        method="holm",
    )

    print()
    print(
        family_name.upper()
    )

    print("-" * 100)

    csv_rows = []

    for i, result in enumerate(
        results
    ):

        print(
            result["name"]
        )

        print(
            f"  Complete paired articles: "
            f"{result['n_articles']}"
        )

        print(
            f"  Mean article delta B-A: "
            f"{result['mean_delta']:+.3f}"
        )

        print(
            f"  Median article delta: "
            f"{result['median_delta']:+.3f}"
        )

        print(
            f"  Cluster bootstrap 95% CI: "
            f"["
            f"{result['ci_lower']:+.3f}, "
            f"{result['ci_upper']:+.3f}"
            f"]"
        )

        print(
            f"  Improved articles: "
            f"{result['improved']}"
        )

        print(
            f"  Worsened articles: "
            f"{result['worsened']}"
        )

        print(
            f"  Unchanged articles: "
            f"{result['unchanged']}"
        )

        print(
            f"  Raw permutation p-value: "
            f"{result['p_value']:.6f}"
        )

        print(
            f"  Holm-adjusted p-value: "
            f"{adjusted_p_values[i]:.6f}"
        )

        print(
            f"  Reject H0: "
            f"{bool(reject[i])}"
        )

        print()

        condition_a = (
            result["condition_a"]
        )

        condition_b = (
            result["condition_b"]
        )

        csv_rows.append(
            {
                "model":
                    model_name,
                "family":
                    family_name,
                "comparison":
                    result["name"],
                "a_shots":
                    condition_a[0],
                "a_strategy":
                    condition_a[1],
                "a_seed":
                    condition_a[2],
                "b_shots":
                    condition_b[0],
                "b_strategy":
                    condition_b[1],
                "b_seed":
                    condition_b[2],
                "n_articles":
                    result[
                        "n_articles"
                    ],
                "mean_delta_b_minus_a":
                    result[
                        "mean_delta"
                    ],
                "median_delta_b_minus_a":
                    result[
                        "median_delta"
                    ],
                "effect_ci_low":
                    result[
                        "ci_lower"
                    ],
                "effect_ci_high":
                    result[
                        "ci_upper"
                    ],
                "improved_articles":
                    result[
                        "improved"
                    ],
                "worsened_articles":
                    result[
                        "worsened"
                    ],
                "unchanged_articles":
                    result[
                        "unchanged"
                    ],
                "p_value_raw":
                    result[
                        "p_value"
                    ],
                "p_value_holm":
                    adjusted_p_values[i],
                "reject_h0":
                    bool(
                        reject[i]
                    ),
            }
        )

    return csv_rows


def main():

    print("=" * 100)
    print(
        "QUALITY FULL "
        "CLUSTER-AWARE INFERENCE"
    )
    print("=" * 100)

    item_to_article = (
        build_item_to_article()
    )

    unique_articles = len(
        set(
            item_to_article.values()
        )
    )

    print(
        f"Items in frozen split: "
        f"{len(item_to_article)}"
    )

    print(
        f"Articles in frozen split: "
        f"{unique_articles}"
    )

    all_rows = []

    for model_name, result_path in (
        RESULT_FILES.items()
    ):

        print()
        print("=" * 100)

        print(
            f"MODEL: "
            f"{model_name.upper()}"
        )

        print("=" * 100)

        records = load_jsonl(
            result_path
        )

        lookup = build_lookup(
            records
        )

        shot_rows = analyze_family(
            model_name=model_name,
            lookup=lookup,
            item_to_article=(
                item_to_article
            ),
            family_name=(
                "Shots effect"
            ),
            comparisons=(
                SHOT_COMPARISONS
            ),
        )

        selection_rows = analyze_family(
            model_name=model_name,
            lookup=lookup,
            item_to_article=(
                item_to_article
            ),
            family_name=(
                "Selection effect"
            ),
            comparisons=(
                SELECTION_COMPARISONS
            ),
        )

        all_rows.extend(
            shot_rows
        )

        all_rows.extend(
            selection_rows
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
        "family",
        "comparison",
        "a_shots",
        "a_strategy",
        "a_seed",
        "b_shots",
        "b_strategy",
        "b_seed",
        "n_articles",
        "mean_delta_b_minus_a",
        "median_delta_b_minus_a",
        "effect_ci_low",
        "effect_ci_high",
        "improved_articles",
        "worsened_articles",
        "unchanged_articles",
        "p_value_raw",
        "p_value_holm",
        "reject_h0",
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
            all_rows
        )

    print("=" * 100)
    print("INFERENCE COMPLETE")
    print("=" * 100)

    print(
        f"Rows saved: "
        f"{len(all_rows)}"
    )

    print(
        f"CSV saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()