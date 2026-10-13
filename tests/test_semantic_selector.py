import json

from src.selection.semantic_selector import (
    SemanticSelector,
)


def load_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return [
            json.loads(line)
            for line in f
        ]


def get_ids(examples):

    return [
        ex["item_id"]
        for ex in examples
    ]


def main():

    selector = SemanticSelector()

    # =========================================================
    # MMLU-PRO
    # =========================================================

    mmlu_demo = load_jsonl(
        "data/splits/mmlu_pro_demo.jsonl"
    )

    mmlu_eval = load_jsonl(
        "data/splits/mmlu_pro_eval.jsonl"
    )

    target = mmlu_eval[0]

    print("=" * 70)
    print("MMLU-PRO SEMANTIC SELECTOR TEST")
    print("=" * 70)

    one = selector.select_semantic_demos(
        mmlu_demo,
        target,
        1,
    )

    three = selector.select_semantic_demos(
        mmlu_demo,
        target,
        3,
    )

    five = selector.select_semantic_demos(
        mmlu_demo,
        target,
        5,
    )

    print("\nTarget:")
    print(target["question"])

    print("\n1-shot:")
    print(get_ids(one))

    print("\n3-shot:")

    for demo in three:
        print(
            demo["item_id"],
            "|",
            demo["category"],
            "|",
            demo["question"][:100]
        )

    print("\n5-shot:")
    print(get_ids(five))

    assert get_ids(one) == get_ids(three)[:1]
    assert get_ids(three) == get_ids(five)[:3]

    repeat = selector.select_semantic_demos(
        mmlu_demo,
        target,
        5,
    )

    assert get_ids(five) == get_ids(repeat)

    print("\nMMLU checks: OK")

    # =========================================================
    # QUALITY
    # =========================================================

    quality_demo = load_jsonl(
        "data/splits/quality_demo.jsonl"
    )

    quality_eval = load_jsonl(
        "data/splits/quality_eval.jsonl"
    )

    target = quality_eval[0]

    print("\n" + "=" * 70)
    print("QUALITY SEMANTIC SELECTOR TEST")
    print("=" * 70)

    three = selector.select_semantic_demos(
        quality_demo,
        target,
        3,
    )

    print("\nTarget question:")
    print(target["question"])

    print("\n3-shot:")

    for demo in three:

        print(
            demo["item_id"],
            "| article:",
            demo["article_id"],
            "|",
            demo["question"][:100]
        )

    article_ids = [
        demo["article_id"]
        for demo in three
    ]

    assert len(article_ids) == len(set(article_ids))

    assert target["article_id"] not in article_ids

    print("\nQuALITY checks: OK")

    print("\n" + "=" * 70)
    print("ALL SEMANTIC SELECTOR TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()