import csv
import json
import os

import numpy as np

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


RESULT_FILES = {
    "qwen": (
        "results/raw_predictions/"
        "mmlu_pro_full_qwen.jsonl"
    ),
    "granite": (
        "results/raw_predictions/"
        "mmlu_pro_full_granite.jsonl"
    ),
    "gemma": (
        "results/raw_predictions/"
        "mmlu_pro_full_gemma.jsonl"
    ),
}

OUTPUT_PATH = (
    "results/statistics/"
    "mmlu_main_inference.csv"
)


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
    {
        "name": "0-shot vs 5-shot random",
        "a": (0, "none", None),
        "b": (5, "random", 42),
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
    {
        "name": "5-shot random vs semantic",
        "a": (5, "random", 42),
        "b": (5, "semantic", None),
    },
]


BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42


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


def get_paired_results(
    lookup,
    condition_a,
    condition_b,
):

    items_a = {}
    items_b = {}

    for key, correct in lookup.items():

        (
            item_id,
            shots,
            strategy,
            seed,
        ) = key

        condition = (
            shots,
            strategy,
            seed,
        )

        if condition == condition_a:
            items_a[item_id] = correct

        if condition == condition_b:
            items_b[item_id] = correct

    common_items = sorted(
        set(items_a)
        & set(items_b)
    )

    pairs = [
        (
            items_a[item_id],
            items_b[item_id],
        )
        for item_id in common_items
    ]

    return pairs


def bootstrap_delta_ci(
    pairs,
    n_bootstrap=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):

    pair_array = np.array(
        pairs,
        dtype=int,
    )

    differences = (
        pair_array[:, 1]
        - pair_array[:, 0]
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(differences)

    bootstrap_effects = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_effects[i] = (
            differences[
                indices
            ].mean()
        )

    ci_low, ci_high = np.percentile(
        bootstrap_effects,
        [2.5, 97.5],
    )

    return (
        float(ci_low),
        float(ci_high),
    )


def run_mcnemar_test(pairs):

    both_wrong = 0
    a_wrong_b_right = 0
    a_right_b_wrong = 0
    both_right = 0

    for a_correct, b_correct in pairs:

        if (
            not a_correct
            and not b_correct
        ):
            both_wrong += 1

        elif (
            not a_correct
            and b_correct
        ):
            a_wrong_b_right += 1

        elif (
            a_correct
            and not b_correct
        ):
            a_right_b_wrong += 1

        else:
            both_right += 1

    table = [
        [
            both_wrong,
            a_wrong_b_right,
        ],
        [
            a_right_b_wrong,
            both_right,
        ],
    ]

    result = mcnemar(
        table,
        exact=True,
    )

    n = len(pairs)

    correct_a = sum(
        int(a)
        for a, _
        in pairs
    )

    correct_b = sum(
        int(b)
        for _, b
        in pairs
    )

    accuracy_a = (
        correct_a
        / n
    )

    accuracy_b = (
        correct_b
        / n
    )

    effect = (
        accuracy_b
        - accuracy_a
    )

    ci_a_low, ci_a_high = (
        proportion_confint(
            count=correct_a,
            nobs=n,
            alpha=0.05,
            method="wilson",
        )
    )

    ci_b_low, ci_b_high = (
        proportion_confint(
            count=correct_b,
            nobs=n,
            alpha=0.05,
            method="wilson",
        )
    )

    (
        effect_ci_low,
        effect_ci_high,
    ) = bootstrap_delta_ci(
        pairs
    )

    return {
        "n": n,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "ci_a_low": ci_a_low,
        "ci_a_high": ci_a_high,
        "ci_b_low": ci_b_low,
        "ci_b_high": ci_b_high,
        "effect": effect,
        "effect_ci_low": effect_ci_low,
        "effect_ci_high": effect_ci_high,
        "a_wrong_b_right":
            a_wrong_b_right,
        "a_right_b_wrong":
            a_right_b_wrong,
        "p_value":
            result.pvalue,
    }


def analyze_family(
    model_name,
    lookup,
    family_name,
    comparisons,
):

    results = []

    for comparison in comparisons:

        pairs = get_paired_results(
            lookup=lookup,
            condition_a=comparison["a"],
            condition_b=comparison["b"],
        )

        result = run_mcnemar_test(
            pairs
        )

        result["name"] = (
            comparison["name"]
        )

        result["condition_a"] = (
            comparison["a"]
        )

        result["condition_b"] = (
            comparison["b"]
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
        f"{family_name.upper()}"
    )

    print("-" * 100)

    csv_rows = []

    for i, result in enumerate(
        results
    ):

        print(
            f"{result['name']}"
        )

        print(
            f"  n paired: "
            f"{result['n']}"
        )

        print(
            f"  accuracy A: "
            f"{result['accuracy_a']:.3f} "
            f"(95% Wilson CI "
            f"[{result['ci_a_low']:.3f}, "
            f"{result['ci_a_high']:.3f}])"
        )

        print(
            f"  accuracy B: "
            f"{result['accuracy_b']:.3f} "
            f"(95% Wilson CI "
            f"[{result['ci_b_low']:.3f}, "
            f"{result['ci_b_high']:.3f}])"
        )

        print(
            f"  delta B-A: "
            f"{result['effect']:+.3f} "
            f"(95% paired bootstrap CI "
            f"[{result['effect_ci_low']:+.3f}, "
            f"{result['effect_ci_high']:+.3f}])"
        )

        print(
            f"  discordant A-wrong/B-right: "
            f"{result['a_wrong_b_right']}"
        )

        print(
            f"  discordant A-right/B-wrong: "
            f"{result['a_right_b_wrong']}"
        )

        print(
            f"  raw p-value: "
            f"{result['p_value']:.6f}"
        )

        print(
            f"  Holm-adjusted p-value: "
            f"{adjusted_p_values[i]:.6f}"
        )

        print(
            f"  reject H0: "
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
                "model": model_name,
                "family": family_name,
                "comparison": (
                    result["name"]
                ),
                "a_shots": (
                    condition_a[0]
                ),
                "a_strategy": (
                    condition_a[1]
                ),
                "a_seed": (
                    condition_a[2]
                ),
                "b_shots": (
                    condition_b[0]
                ),
                "b_strategy": (
                    condition_b[1]
                ),
                "b_seed": (
                    condition_b[2]
                ),
                "n_paired": (
                    result["n"]
                ),
                "accuracy_a": (
                    result["accuracy_a"]
                ),
                "accuracy_b": (
                    result["accuracy_b"]
                ),
                "ci_a_low": (
                    result["ci_a_low"]
                ),
                "ci_a_high": (
                    result["ci_a_high"]
                ),
                "ci_b_low": (
                    result["ci_b_low"]
                ),
                "ci_b_high": (
                    result["ci_b_high"]
                ),
                "effect_b_minus_a": (
                    result["effect"]
                ),
                "effect_ci_low": (
                    result[
                        "effect_ci_low"
                    ]
                ),
                "effect_ci_high": (
                    result[
                        "effect_ci_high"
                    ]
                ),
                "a_wrong_b_right": (
                    result[
                        "a_wrong_b_right"
                    ]
                ),
                "a_right_b_wrong": (
                    result[
                        "a_right_b_wrong"
                    ]
                ),
                "p_value_raw": (
                    result["p_value"]
                ),
                "p_value_holm": (
                    adjusted_p_values[i]
                ),
                "reject_h0": (
                    bool(reject[i])
                ),
            }
        )

    return csv_rows


def main():

    print("=" * 100)
    print("MMLU-PRO FULL INFERENCE")
    print("=" * 100)

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
        "n_paired",
        "accuracy_a",
        "accuracy_b",
        "ci_a_low",
        "ci_a_high",
        "ci_b_low",
        "ci_b_high",
        "effect_b_minus_a",
        "effect_ci_low",
        "effect_ci_high",
        "a_wrong_b_right",
        "a_right_b_wrong",
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