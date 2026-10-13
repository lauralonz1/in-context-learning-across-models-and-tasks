from src.evaluation.result_record import build_result_record
from src.evaluation.scoring import score_prediction


def main():

    item = {
        "item_id": "mmlu_pro_test_123",
        "benchmark": "mmlu_pro",
    }

    scoring = score_prediction(
        raw_output="Answer: C",
        gold_answer="C",
    )

    record = build_result_record(
        item=item,
        model_name="qwen3_4b",
        shots=1,
        strategy="semantic",
        seed=None,
        demo_ids=[
            "mmlu_pro_validation_3"
        ],
        prompt="FAKE PROMPT FOR TEST",
        raw_output="Answer: C",
        scoring=scoring,
        latency_seconds=1.23,
        prompt_tokens=500,
        completion_tokens=3,
        total_tokens=503,
        provider="SiliconFlow",
        error=None,
    )

    print("=" * 70)
    print("RESULT RECORD TEST")
    print("=" * 70)

    for key, value in record.items():
        print(f"{key}: {value}")

    assert record["item_id"] == "mmlu_pro_test_123"
    assert record["model"] == "qwen3_4b"
    assert record["correct"] is True
    assert record["prediction"] == "C"
    assert record["parse_failure"] is False
    assert record["total_tokens"] == 503
    assert record["provider"] == "SiliconFlow"

    print("\n" + "=" * 70)
    print("RESULT RECORD TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()