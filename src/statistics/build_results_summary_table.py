import csv
import os


MMLU_PATH = (
    "results/statistics/"
    "mmlu_main_inference.csv"
)

QUALITY_PATH = (
    "results/statistics/"
    "quality_main_inference.csv"
)

CONTEXT_PATH = (
    "results/statistics/"
    "quality_context_ablation.csv"
)

OUTPUT_PATH = (
    "results/statistics/"
    "results_summary_table.csv"
)


MODEL_LABELS = {
    "qwen": "Qwen",
    "granite": "Granite",
    "gemma": "Gemma",
}


def load_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def interpret_effect(
    effect,
    significant,
):
    if significant:

        if effect > 0:
            return (
                "Significant improvement"
            )

        if effect < 0:
            return (
                "Significant degradation"
            )

    if effect > 0:
        return (
            "Positive but not significant"
        )

    if effect < 0:
        return (
            "Negative but not significant"
        )

    return "No observed difference"


def main():

    rows_out = []

    # ==========================================================
    # MMLU MAIN
    # ==========================================================

    mmlu_rows = load_csv(
        MMLU_PATH
    )

    for row in mmlu_rows:

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
            row["reject_h0"]
            .lower()
            == "true"
        )

        rows_out.append(
            {
                "benchmark":
                    "MMLU-Pro",
                "model":
                    MODEL_LABELS[
                        row["model"]
                    ],
                "family":
                    row["family"],
                "comparison":
                    row["comparison"],
                "effect_pp":
                    effect * 100,
                "ci_low_pp":
                    ci_low * 100,
                "ci_high_pp":
                    ci_high * 100,
                "adjusted_p_value":
                    p_holm,
                "significant":
                    significant,
                "interpretation":
                    interpret_effect(
                        effect,
                        significant,
                    ),
            }
        )

    # ==========================================================
    # QUALITY MAIN
    # ==========================================================

    quality_rows = load_csv(
        QUALITY_PATH
    )

    for row in quality_rows:

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
            row["reject_h0"]
            .lower()
            == "true"
        )

        rows_out.append(
            {
                "benchmark":
                    "QuALITY",
                "model":
                    MODEL_LABELS[
                        row["model"]
                    ],
                "family":
                    row["family"],
                "comparison":
                    row["comparison"],
                "effect_pp":
                    effect * 100,
                "ci_low_pp":
                    ci_low * 100,
                "ci_high_pp":
                    ci_high * 100,
                "adjusted_p_value":
                    p_holm,
                "significant":
                    significant,
                "interpretation":
                    interpret_effect(
                        effect,
                        significant,
                    ),
            }
        )

    # ==========================================================
    # CONTEXT ABLATION
    # ==========================================================

    context_rows = load_csv(
        CONTEXT_PATH
    )

    for row in context_rows:

        effect_pp = float(
            row["context_gain_pp"]
        )

        ci_low_pp = (
            float(
                row["ci_low"]
            )
            * 100
        )

        ci_high_pp = (
            float(
                row["ci_high"]
            )
            * 100
        )

        p_value = float(
            row["p_value"]
        )

        significant = (
            p_value < 0.05
        )

        rows_out.append(
            {
                "benchmark":
                    "QuALITY",
                "model":
                    row["model_label"],
                "family":
                    "Context utilization",
                "comparison":
                    "No context vs full context",
                "effect_pp":
                    effect_pp,
                "ci_low_pp":
                    ci_low_pp,
                "ci_high_pp":
                    ci_high_pp,
                "adjusted_p_value":
                    p_value,
                "significant":
                    significant,
                "interpretation":
                    (
                        "Significant context benefit"
                        if significant
                        else
                        "No significant context benefit"
                    ),
            }
        )

    # ==========================================================
    # SAVE
    # ==========================================================

    os.makedirs(
        "results/statistics",
        exist_ok=True,
    )

    fieldnames = [
        "benchmark",
        "model",
        "family",
        "comparison",
        "effect_pp",
        "ci_low_pp",
        "ci_high_pp",
        "adjusted_p_value",
        "significant",
        "interpretation",
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
            rows_out
        )

    print(
        f"Rows saved: "
        f"{len(rows_out)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()