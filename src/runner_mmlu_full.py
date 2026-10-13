import json
import os

from src.utils.config import load_config
from src.selection.selector import DemoSelector
from src.prompts.mmlu_pro_prompt import build_mmlu_pro_prompt
from src.models.openrouter_client import call_model
from src.evaluation.scoring import score_prediction
from src.evaluation.result_record import build_result_record


OUTPUT_PATH = (
    "results/raw_predictions/"
    "mmlu_pro_full_qwen.jsonl"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
        ]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(
            record,
            f,
            ensure_ascii=False
        )
        f.write("\n")


def main():

    # ==========================================================
    # LOAD CONFIG
    # ==========================================================

    config = load_config(
        "configs/experiment.yaml"
    )

    model_name = "qwen3_30b_a3b_instruct_2507"

    benchmark_config = (
        config["benchmarks"]["mmlu_pro"]
    )

    generation_config = (
        config["generation"]
    )

    random_seeds = [42]

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    demo_pool = load_jsonl(
        benchmark_config["demo_file"]
    )

    full_items = load_jsonl(
        "data/splits/mmlu_pro_full.jsonl"
    )

    selector = DemoSelector()

    # ==========================================================
    # BUILD RUN CONDITIONS
    # ==========================================================

    run_conditions = []

    for condition in benchmark_config["conditions"]:

        shots = condition["shots"]
        strategy = condition["strategy"]

        if strategy == "random":

            for seed in random_seeds:

                run_conditions.append(
                    {
                        "shots": shots,
                        "strategy": strategy,
                        "seed": seed,
                    }
                )

        else:

            run_conditions.append(
                {
                    "shots": shots,
                    "strategy": strategy,
                    "seed": None,
                }
            )

    # ==========================================================
    # PRINT PLAN
    # ==========================================================

    print("\nRUN CONDITIONS")
    print("=" * 70)

    for i, condition in enumerate(
        run_conditions,
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{condition['shots']}-shot | "
            f"{condition['strategy']} | "
            f"seed={condition['seed']}"
        )

    total = (
        len(full_items)
        * len(run_conditions)
    )

    print("\n" + "=" * 70)
    print(
        f"Full items: "
        f"{len(full_items)}"
    )
    print(
        f"Run conditions: "
        f"{len(run_conditions)}"
    )
    print(
        f"Total requests: "
        f"{total}"
    )
    print("=" * 70)

    # ==========================================================
    # PREPARE OUTPUT FOLDER
    # ==========================================================

    os.makedirs(
        "results/raw_predictions",
        exist_ok=True
    )

    # ==========================================================
    # RESUME SUPPORT
    # ==========================================================

    completed_keys = set()

    if os.path.exists(OUTPUT_PATH):

        existing_records = load_jsonl(
            OUTPUT_PATH
        )

        for record in existing_records:

            if record["error"] is not None:
                continue

            key = (
                record["item_id"],
                record["model"],
                record["shots"],
               record["strategy"],
                record["seed"],
            )

            completed_keys.add(key)

        print(
            f"\nExisting completed records: "
            f"{len(completed_keys)}"
        )

    else:

        print(
            "\nNo existing result file found. "
            "Starting fresh."
        )

    # ==========================================================
    # RUN PILOT
    # ==========================================================

    print("\n" + "=" * 70)
    print("MMLU-PRO FULL")
    print("=" * 70)

    current = 0

    for target in full_items:

        for condition in run_conditions:

            current += 1

            shots = condition["shots"]
            strategy = condition["strategy"]
            seed = condition["seed"]

            run_key = (
                target["item_id"],
                model_name,
                shots,
                strategy,
                seed,
            )

            # ==================================================
            # SKIP IF ALREADY DONE
            # ==================================================

            if run_key in completed_keys:

                print(
                    f"\n[{current}/{total}] SKIP | "
                    f"{target['item_id']} | "
                    f"{shots}-shot | "
                    f"{strategy} | "
                    f"seed={seed}"
                )

                continue

            print(
                f"\n[{current}/{total}] "
                f"{target['item_id']} | "
                f"{shots}-shot | "
                f"{strategy} | "
                f"seed={seed}"
            )

            # ==================================================
            # SELECT DEMONSTRATIONS
            # ==================================================

            selector_seed = (
                seed
                if seed is not None
                else 42
            )

            demos = selector.select(
                demo_pool=demo_pool,
                target=target,
                n_shots=shots,
                strategy=strategy,
                seed=selector_seed,
            )

            # ==================================================
            # BUILD PROMPT
            # ==================================================

            prompt = build_mmlu_pro_prompt(
                target=target,
                demonstrations=demos,
            )

            # ==================================================
            # CALL MODEL
            # ==================================================

            try:

                model_result = call_model(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=(
                        generation_config[
                            "temperature"
                        ]
                    ),
                    top_p=(
                        generation_config[
                            "top_p"
                        ]
                    ),
                    max_new_tokens=(
                        generation_config[
                            "max_new_tokens"
                        ]
                    ),
                )

                raw_output = (
                    model_result["raw_output"]
                )

                # ==============================================
                # SCORE
                # ==============================================

                scoring = score_prediction(
                    raw_output=raw_output,
                    gold_answer=(
                        target["gold_answer"]
                    ),
                    valid_letters=list(
                        "ABCDEFGHIJ"
                    ),
                )

                # ==============================================
                # BUILD SUCCESS RECORD
                # ==============================================

                record = build_result_record(
                    item=target,
                    model_name=model_name,
                    shots=shots,
                    strategy=strategy,
                    seed=seed,
                    demo_ids=[
                        demo["item_id"]
                        for demo in demos
                    ],
                    prompt=prompt,
                    raw_output=raw_output,
                    scoring=scoring,
                    latency_seconds=(
                        model_result[
                            "latency_seconds"
                        ]
                    ),
                    prompt_tokens=(
                        model_result[
                            "prompt_tokens"
                        ]
                    ),
                    completion_tokens=(
                        model_result[
                            "completion_tokens"
                        ]
                    ),
                    total_tokens=(
                        model_result[
                            "total_tokens"
                        ]
                    ),
                    provider=(
                        model_result[
                            "provider"
                        ]
                    ),
                    error=None,
                )

            except Exception as e:

                # ==============================================
                # BUILD ERROR RECORD
                # ==============================================

                scoring = score_prediction(
                    raw_output=None,
                    gold_answer=(
                        target["gold_answer"]
                    ),
                    valid_letters=list(
                        "ABCDEFGHIJ"
                    ),
                )

                record = build_result_record(
                    item=target,
                    model_name=model_name,
                    shots=shots,
                    strategy=strategy,
                    seed=seed,
                    demo_ids=[
                        demo["item_id"]
                        for demo in demos
                    ],
                    prompt=prompt,
                    raw_output=None,
                    scoring=scoring,
                    latency_seconds=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    provider=None,
                    error=str(e),
                )

            # ==================================================
            # SAVE RESULT
            # ==========================================================

            append_jsonl(
                OUTPUT_PATH,
                record
            )

            if record["error"] is None:
                completed_keys.add(
                    run_key
                )

            # ==================================================
            # PRINT RESULT
            # ==========================================================

            print(
                f"prediction="
                f"{record['prediction']} | "
                f"gold="
                f"{record['gold_answer']} | "
                f"correct="
                f"{record['correct']} | "
                f"provider="
                f"{record['provider']} | "
                f"error="
                f"{record['error']}"
            )

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print("\n" + "=" * 70)
    print("MMLU-PRO FULL COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()