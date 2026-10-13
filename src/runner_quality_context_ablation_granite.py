import json
import os

from src.utils.config import load_config
from src.prompts.quality_no_context_prompt import (
    build_quality_no_context_prompt,
)
from src.models.openrouter_client import call_model
from src.evaluation.scoring import score_prediction
from src.evaluation.result_record import build_result_record


OUTPUT_PATH = (
    "results/raw_predictions/"
    "quality_context_ablation_granite.jsonl"
)

SPLIT_PATH = (
    "data/splits/"
    "quality_sensitivity.jsonl"
)


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

    ablation_items = load_jsonl(
        SPLIT_PATH
    )

    print("\n" + "=" * 70)
    print("QUALITY CONTEXT ABLATION - QWEN")
    print("=" * 70)

    print(
        f"Ablation items: "
        f"{len(ablation_items)}"
    )

    print(
        "Condition: "
        "0-shot, no article/context"
    )

    print(
        f"Total requests: "
        f"{len(ablation_items)}"
    )

    print("=" * 70)

    if len(ablation_items) != 200:
        raise ValueError(
            "Expected exactly 200 "
            "ablation items."
        )

    # ==========================================================
    # OUTPUT + RESUME
    # ==========================================================

    os.makedirs(
        "results/raw_predictions",
        exist_ok=True,
    )

    completed_ids = set()

    if os.path.exists(
        OUTPUT_PATH
    ):

        existing_records = load_jsonl(
            OUTPUT_PATH
        )

        for record in existing_records:

            if record["error"] is not None:
                continue

            completed_ids.add(
                record["item_id"]
            )

        print(
            f"\nExisting completed records: "
            f"{len(completed_ids)}"
        )

    else:

        print(
            "\nNo existing result file found. "
            "Starting fresh."
        )

    # ==========================================================
    # RUN
    # ==========================================================

    total = len(
        ablation_items
    )

    for index, target in enumerate(
        ablation_items,
        start=1,
    ):

        if (
            target["item_id"]
            in completed_ids
        ):

            print(
                f"\n[{index}/{total}] "
                f"SKIP | "
                f"{target['item_id']}"
            )

            continue

        print(
            f"\n[{index}/{total}] "
            f"{target['item_id']}"
        )

        prompt = (
            build_quality_no_context_prompt(
                target=target,
            )
        )

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
                shots=0,
                strategy=(
                    "no_context"
                ),
                seed=None,
                demo_ids=[],
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
                shots=0,
                strategy=(
                    "no_context"
                ),
                seed=None,
                demo_ids=[],
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

        append_jsonl(
            OUTPUT_PATH,
            record,
        )

        if record["error"] is None:
            completed_ids.add(
                target["item_id"]
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
    print(
        "QUALITY CONTEXT ABLATION "
        "QWEN COMPLETE"
    )
    print("=" * 70)

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()