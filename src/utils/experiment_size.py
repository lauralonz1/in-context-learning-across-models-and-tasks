from collections import Counter

from src.utils.config import load_config
from src.utils.run_plan import build_run_plan


def calculate_calls(run_plan, n_models, item_counts):
    """
    Calcola quante inference/API calls servono.

    Ogni run viene applicata a tutti gli item
    previsti per quel benchmark.
    """

    total = 0
    per_benchmark = Counter()

    for run in run_plan:

        benchmark = run["benchmark"]

        n_items = item_counts[benchmark]

        calls = n_items * n_models

        total += calls
        per_benchmark[benchmark] += calls

    return total, per_benchmark


def main():

    config = load_config()

    run_plan = build_run_plan(config)

    models = config["models"]

    n_models = len(models)

    pilot_counts = {
        "mmlu_pro": config["pilot"]["mmlu_pro_items"],
        "quality": config["pilot"]["quality_items"],
    }

    print("=" * 70)
    print("EXPERIMENT SIZE")
    print("=" * 70)

    print(f"\nModelli: {n_models}")

    for model in models:
        print(f"- {model}")

    # ---------------------------------------------------------
    # Run per benchmark
    # ---------------------------------------------------------

    runs_per_benchmark = Counter(
        run["benchmark"]
        for run in run_plan
    )

    print("\n--- RUN PER MODEL ---")

    for benchmark, count in runs_per_benchmark.items():
        print(
            f"{benchmark}: {count}"
        )

    print(
        f"Totale per modello: "
        f"{len(run_plan)}"
    )

    print(
        f"Totale sui {n_models} modelli: "
        f"{len(run_plan) * n_models}"
    )

    # ---------------------------------------------------------
    # PILOT
    # ---------------------------------------------------------

    total_calls, per_benchmark = calculate_calls(
        run_plan=run_plan,
        n_models=n_models,
        item_counts=pilot_counts,
    )

    print("\n" + "=" * 70)
    print("PILOT")
    print("=" * 70)

    print(
        f"MMLU-Pro items/run: "
        f"{pilot_counts['mmlu_pro']}"
    )

    print(
        f"QuALITY items/run: "
        f"{pilot_counts['quality']}"
    )

    print("\nInference richieste:")

    for benchmark, calls in per_benchmark.items():

        print(
            f"{benchmark}: {calls}"
        )

    print(
        f"\nTOTAL PILOT CALLS: "
        f"{total_calls}"
    )

    # ---------------------------------------------------------
    # ESEMPI DI FULL STUDY
    # ---------------------------------------------------------

    scenarios = [
        {
            "name": "small",
            "mmlu_pro": 300,
            "quality": 200,
        },
        {
            "name": "medium",
            "mmlu_pro": 500,
            "quality": 300,
        },
        {
            "name": "large",
            "mmlu_pro": 1000,
            "quality": 500,
        },
    ]

    print("\n" + "=" * 70)
    print("POSSIBLE FULL-STUDY SIZES")
    print("=" * 70)

    for scenario in scenarios:

        counts = {
            "mmlu_pro": scenario["mmlu_pro"],
            "quality": scenario["quality"],
        }

        total, _ = calculate_calls(
            run_plan=run_plan,
            n_models=n_models,
            item_counts=counts,
        )

        print(
            f"\n{scenario['name'].upper()}:"
        )

        print(
            f"MMLU-Pro: {scenario['mmlu_pro']} items"
        )

        print(
            f"QuALITY: {scenario['quality']} items"
        )

        print(
            f"Total inference calls: {total}"
        )


if __name__ == "__main__":
    main()