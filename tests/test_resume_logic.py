def build_completed_keys(records):

    completed_keys = set()

    for record in records:

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

    return completed_keys


def main():

    fake_records = [
        {
            "item_id": "mmlu_pro_test_1",
            "model": "qwen3_30b_a3b_instruct_2507",
            "shots": 1,
            "strategy": "random",
            "seed": 42,
            "error": None,
        },
        {
            "item_id": "mmlu_pro_test_2",
            "model": "qwen3_30b_a3b_instruct_2507",
            "shots": 3,
            "strategy": "random",
            "seed": 43,
            "error": "Temporary API error",
        },
    ]

    completed_keys = build_completed_keys(
        fake_records
    )

    success_key = (
        "mmlu_pro_test_1",
        "qwen3_30b_a3b_instruct_2507",
        1,
        "random",
        42,
    )

    failed_key = (
        "mmlu_pro_test_2",
        "qwen3_30b_a3b_instruct_2507",
        3,
        "random",
        43,
    )

    print("=" * 70)
    print("RESUME LOGIC TEST")
    print("=" * 70)

    print(
        "Successful request completed:",
        success_key in completed_keys
    )

    print(
        "Failed request completed:",
        failed_key in completed_keys
    )

    assert success_key in completed_keys

    assert failed_key not in completed_keys

    print("\n" + "=" * 70)
    print("RESUME LOGIC TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()