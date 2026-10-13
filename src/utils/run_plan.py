from src.utils.config import load_config


def build_run_plan(config):
    """
    Espande la configurazione sperimentale
    in una lista di run concrete.

    Esempio:
    random con seed [42, 43, 44]
    -> tre run distinte.
    """

    run_plan = []

    random_seeds = config["random_selection"]["seeds"]

    for benchmark_name, benchmark_config in config["benchmarks"].items():

        conditions = benchmark_config["conditions"]

        for condition in conditions:

            shots = condition["shots"]
            strategy = condition["strategy"]

            # Zero-shot:
            # nessun seed di selezione necessario.
            if strategy == "none":

                run_plan.append({
                    "benchmark": benchmark_name,
                    "shots": shots,
                    "strategy": strategy,
                    "seed": None,
                })

            # Random:
            # una run per ogni seed.
            elif strategy == "random":

                for seed in random_seeds:

                    run_plan.append({
                        "benchmark": benchmark_name,
                        "shots": shots,
                        "strategy": strategy,
                        "seed": seed,
                    })

            # Semantic:
            # deterministico -> una sola run.
            elif strategy == "semantic":

                run_plan.append({
                    "benchmark": benchmark_name,
                    "shots": shots,
                    "strategy": strategy,
                    "seed": None,
                })

            else:

                raise ValueError(
                    f"Strategia sconosciuta: {strategy}"
                )

    return run_plan


def main():

    config = load_config()

    run_plan = build_run_plan(config)

    print("=" * 70)
    print("EXPERIMENT RUN PLAN")
    print("=" * 70)

    for i, run in enumerate(run_plan, start=1):

        print(
            f"{i:02d} | "
            f"{run['benchmark']:10s} | "
            f"{run['shots']}-shot | "
            f"{run['strategy']:8s} | "
            f"seed={run['seed']}"
        )

    print("\n" + "=" * 70)
    print(f"TOTAL RUNS: {len(run_plan)}")
    print("=" * 70)


if __name__ == "__main__":
    main()