from src.evaluation.scoring import score_prediction


def main():

    tests = [
        {
            "raw_output": "C",
            "gold": "C",
            "expected_prediction": "C",
            "expected_correct": True,
            "expected_parse_failure": False,
        },
        {
            "raw_output": "Answer: B",
            "gold": "C",
            "expected_prediction": "B",
            "expected_correct": False,
            "expected_parse_failure": False,
        },
        {
            "raw_output": "I don't know.",
            "gold": "I",
            "expected_prediction": None,
            "expected_correct": False,
            "expected_parse_failure": True,
        },
        {
            "raw_output": None,
            "gold": "A",
            "expected_prediction": None,
            "expected_correct": False,
            "expected_parse_failure": True,
        },
    ]

    print("=" * 70)
    print("SCORING TEST")
    print("=" * 70)

    for case in tests:

        result = score_prediction(
            raw_output=case["raw_output"],
            gold_answer=case["gold"],
        )

        print("\nInput:")
        print(case["raw_output"])

        print("Result:")
        print(result)

        assert (
            result["prediction"]
            == case["expected_prediction"]
        )

        assert (
            result["correct"]
            == case["expected_correct"]
        )

        assert (
            result["parse_failure"]
            == case["expected_parse_failure"]
        )

    print("\n" + "=" * 70)
    print("ALL SCORING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()