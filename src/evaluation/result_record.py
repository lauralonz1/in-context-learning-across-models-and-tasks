from datetime import datetime, timezone


def build_result_record(
    *,
    item,
    model_name,
    shots,
    strategy,
    seed,
    demo_ids,
    prompt,
    raw_output,
    scoring,
    latency_seconds=None,
    prompt_tokens=None,
    completion_tokens=None,
    total_tokens=None,
    provider=None,
    error=None,
):
    """
    Costruisce il record standard per una singola inference.
    """

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "item_id": item["item_id"],
        "benchmark": item["benchmark"],

        "model": model_name,

        "shots": shots,
        "strategy": strategy,
        "seed": seed,

        "demo_ids": demo_ids,

        "prompt": prompt,

        "gold_answer": scoring["gold_answer"],
        "prediction": scoring["prediction"],
        "raw_output": raw_output,

        "correct": scoring["correct"],
        "parse_failure": scoring["parse_failure"],

        "latency_seconds": latency_seconds,

        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,

        "provider": provider,
        "error": error,
        
    }