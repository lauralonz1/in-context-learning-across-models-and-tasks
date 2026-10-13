import csv
import os


INPUT_PATH = (
    "results/statistics/"
    "results_summary_table.csv"
)

OUTPUT_PATH = (
    "results/statistics/"
    "key_findings_table_final.csv"
)


def load_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def format_p(value):

    p = float(value)

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def main():

    rows = load_csv(
        INPUT_PATH
    )

    selected = []

    for row in rows:

        keep = False

        if row["family"] == "Shots effect":
            keep = True

        if (
            row["family"] == "Selection effect"
            and row["significant"] == "True"
        ):
            keep = True

        if row["family"] == "Context utilization":
            keep = True

        if not keep:
            continue

        effect = float(
            row["effect_pp"]
        )

        ci_low = float(
            row["ci_low_pp"]
        )

        ci_high = float(
            row["ci_high_pp"]
        )

        comparison = (
            row["comparison"]
        )

        if row["family"] == "Context utilization":
            comparison = (
                "Full context vs no context"
            )

        selected.append(
            {
                "Benchmark":
                    row["benchmark"],
                "Model":
                    row["model"],
                "Comparison":
                    comparison,
                "Effect (pp)":
                    f"{effect:+.1f}",
                "95% CI":
                    (
                        f"[{ci_low:+.1f}, "
                        f"{ci_high:+.1f}]"
                    ),
                "p-value":
                    format_p(
                        row[
                            "adjusted_p_value"
                        ]
                    ),
                "Significant":
                    (
                        "Yes"
                        if row["significant"]
                        == "True"
                        else "No"
                    ),
                "Interpretation":
                    row["interpretation"],
            }
        )

    fieldnames = [
        "Benchmark",
        "Model",
        "Comparison",
        "Effect (pp)",
        "95% CI",
        "p-value",
        "Significant",
        "Interpretation",
    ]

    os.makedirs(
        "results/statistics",
        exist_ok=True,
    )

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
            selected
        )

    print(
        f"Rows saved: "
        f"{len(selected)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()