import json
import os
import time

from src.utils.config import load_config
from src.prompts.quality_no_context_prompt import (
    build_quality_no_context_prompt,
)
from src.models.openrouter_client import call_model
from src.evaluation.scoring import score_prediction
from src.evaluation.result_record import build_result_record


OUTPUT_PATH = (
    "results/raw_predictions/"
    "quality_context_ablation_gemma.jsonl"
)

SPLIT_PATH = (
    "data/splits/"
    "quality_sensitivity.jsonl"
)

RATE_LIMIT_WAIT_SECONDS = 20
MAX_ATTEMPTS = 3


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


def is_rate_limit_error(error):
    text = str(error).lower()

    indicators = [
        "429",
        "engine_overloaded",
        "rate limit",
        "rate_limit",
        "too many requests",
    ]

    return any(
        indicator in text
        for indicator in indicators
    )


def call_model_with_retry(
    model_name,
    prompt,
    generation_config,
):

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        try:

            return call_model(
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

        except KeyboardInterrupt:
            raise

        except Exception as e:

            if not is_rate_limit_error(e):
                raise

            if attempt == MAX_ATTEMPTS:
                raise

            print(
                f"429 / overload | "
                f"attempt={attempt} | "
                f"waiting "
                f"{RATE_LIMIT_WAIT_SECONDS}s..."
            )

            time.sleep(
                RATE_LIMIT_WAIT_SECONDS
            )


def main():

    config = load_config(
        "configs/experiment.yaml"
    )

    model_name = (
        "gemma3_4b"
    )

    generation_config = (
        config["generation"]
    )

    ablation_items = load_jsonl(
        SPLIT_PATH
    )

    print("\n" + "=" * 70)
    print("QUALITY CONTEXT ABLATION - GEMMA")
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

    print(
        f"429 max attempts: "
        f"{MAX_ATTEMPTS}"
    )

    print(
        f"429 retry wait: "
        f"{RATE_LIMIT_WAIT_SECONDS}s"
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

            model_result = (
                call_model_with_retry(
                    model_name=model_name,
                    prompt=prompt,
                    generation_config=(
                        generation_config
                    ),
                )
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

        except KeyboardInterrupt:

            print(
                "\nInterrupted by user. "
                "Completed records are safe."
            )

            raise

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
        "GEMMA COMPLETE"
    )
    print("=" * 70)

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()