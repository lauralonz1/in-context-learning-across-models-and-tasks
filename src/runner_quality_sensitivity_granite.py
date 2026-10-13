import json
import os

from transformers import AutoTokenizer

from src.utils.config import load_config
from src.selection.selector import DemoSelector
from src.prompts.quality_prompt import build_quality_prompt
from src.models.openrouter_client import call_model
from src.evaluation.scoring import score_prediction
from src.evaluation.result_record import build_result_record


OUTPUT_PATH = (
    "results/raw_predictions/"
    "quality_sensitivity_granite.jsonl"
)

SPLIT_PATH = (
    "data/splits/"
    "quality_sensitivity.jsonl"
)

TOKENIZER_ID = (
    "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

RUN_CONDITIONS = [
    {
        "shots": 1,
        "strategy": "random",
        "seed": 43,
    },
    {
        "shots": 1,
        "strategy": "random",
        "seed": 44,
    },
    {
        "shots": 3,
        "strategy": "random",
        "seed": 43,
    },
    {
        "shots": 3,
        "strategy": "random",
        "seed": 44,
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


def append_jsonl(path, record):
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        json.dump(
            record,
            f,
            ensure_ascii=False,
        )
        f.write("\n")


def main():

    config = load_config(
        "configs/experiment.yaml"
    )

    model_name = (
        "granite4_micro"
    )

    generation_config = (
        config["generation"]
    )

    max_prompt_tokens = (
        config["quality"][
            "max_prompt_tokens"
        ]
    )

    demo_pool = load_jsonl(
        config["benchmarks"][
            "quality"
        ]["demo_file"]
    )

    sensitivity_items = load_jsonl(
        SPLIT_PATH
    )

    selector = DemoSelector()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            TOKENIZER_ID
        )
    )

    print("\nRUN CONDITIONS")
    print("=" * 70)

    for i, condition in enumerate(
        RUN_CONDITIONS,
        start=1,
    ):

        print(
            f"{i:02d}. "
            f"{condition['shots']}-shot | "
            f"{condition['strategy']} | "
            f"seed={condition['seed']}"
        )

    total = (
        len(sensitivity_items)
        * len(RUN_CONDITIONS)
    )

    print("\n" + "=" * 70)

    print(
        f"Sensitivity items: "
        f"{len(sensitivity_items)}"
    )

    print(
        f"Run conditions: "
        f"{len(RUN_CONDITIONS)}"
    )

    print(
        f"Total requests: "
        f"{total}"
    )

    print(
        f"Prompt budget: "
        f"{max_prompt_tokens}"
    )

    print("=" * 70)

    if len(sensitivity_items) != 200:
        raise ValueError(
            "Expected exactly 200 "
            "sensitivity items."
        )

    if len(RUN_CONDITIONS) != 4:
        raise ValueError(
            "Expected exactly 4 "
            "sensitivity conditions."
        )

    if total != 800:
        raise ValueError(
            "Expected exactly 800 requests."
        )

    # ==========================================================
    # OUTPUT + RESUME
    # ==========================================================

    os.makedirs(
        "results/raw_predictions",
        exist_ok=True,
    )

    completed_keys = set()

    if os.path.exists(
        OUTPUT_PATH
    ):

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

            completed_keys.add(
                key
            )

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
    # RUN
    # ==========================================================

    print("\n" + "=" * 70)
    print("QUALITY SENSITIVITY - QWEN")
    print("=" * 70)

    current = 0

    for target in sensitivity_items:

        for condition in RUN_CONDITIONS:

            current += 1

            shots = (
                condition["shots"]
            )

            strategy = (
                condition["strategy"]
            )

            seed = (
                condition["seed"]
            )

            run_key = (
                target["item_id"],
                model_name,
                shots,
                strategy,
                seed,
            )

            if run_key in completed_keys:

                print(
                    f"\n[{current}/{total}] "
                    f"SKIP | "
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
            # SELECT DEMOS
            # ==================================================

            demos = selector.select(
                demo_pool=demo_pool,
                target=target,
                n_shots=shots,
                strategy=strategy,
                seed=seed,
            )

            # ==================================================
            # BUILD PROMPT
            # ==================================================

            prompt = build_quality_prompt(
                target=target,
                demonstrations=demos,
            )

            # ==================================================
            # TOKEN BUDGET CHECK
            # ==================================================

            prompt_tokens_local = len(
                tokenizer.encode(
                    prompt,
                    add_special_tokens=True,
                )
            )

            print(
                f"prompt_tokens_local="
                f"{prompt_tokens_local}"
            )

            if (
                prompt_tokens_local
                > max_prompt_tokens
            ):

                scoring = score_prediction(
                    raw_output=None,
                    gold_answer=(
                        target[
                            "gold_answer"
                        ]
                    ),
                    valid_letters=list(
                        "ABCD"
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
                    prompt_tokens=(
                        prompt_tokens_local
                    ),
                    completion_tokens=None,
                    total_tokens=None,
                    provider=None,
                    error=(
                        "PROMPT_OVER_BUDGET: "
                        f"{prompt_tokens_local} "
                        f"> {max_prompt_tokens}"
                    ),
                )

                append_jsonl(
                    OUTPUT_PATH,
                    record,
                )

                print(
                    "SKIPPED: "
                    "prompt over budget"
                )

                continue

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
                    model_result[
                        "raw_output"
                    ]
                )

                scoring = score_prediction(
                    raw_output=raw_output,
                    gold_answer=(
                        target[
                            "gold_answer"
                        ]
                    ),
                    valid_letters=list(
                        "ABCD"
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

                scoring = score_prediction(
                    raw_output=None,
                    gold_answer=(
                        target[
                            "gold_answer"
                        ]
                    ),
                    valid_letters=list(
                        "ABCD"
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
                    prompt_tokens=(
                        prompt_tokens_local
                    ),
                    completion_tokens=None,
                    total_tokens=None,
                    provider=None,
                    error=str(e),
                )

            # ==================================================
            # SAVE
            # ==================================================

            append_jsonl(
                OUTPUT_PATH,
                record,
            )

            if record["error"] is None:

                completed_keys.add(
                    run_key
                )

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

    print("\n" + "=" * 70)
    print("QUALITY SENSITIVITY QWEN COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()