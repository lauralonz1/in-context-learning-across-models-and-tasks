import json

from src.selection.random_selector import (
    select_random_demos,
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

    # =========================================================
    # MMLU-PRO
    # =========================================================

    mmlu_demo = load_jsonl(
        "data/splits/mmlu_pro_demo.jsonl"
    )

    mmlu_eval = load_jsonl(
        "data/splits/mmlu_pro_eval.jsonl"
    )

    target_1 = mmlu_eval[0]
    target_2 = mmlu_eval[1]

    print("=" * 70)
    print("MMLU-PRO RANDOM SELECTOR TEST")
    print("=" * 70)

    one = select_random_demos(
        mmlu_demo,
        target_1,
        1,
        seed=42,
    )

    three = select_random_demos(
        mmlu_demo,
        target_1,
        3,
        seed=42,
    )

    five = select_random_demos(
        mmlu_demo,
        target_1,
        5,
        seed=42,
    )

    print("\nTarget 1:")
    print(target_1["item_id"])

    print("\n1-shot:")
    print(get_ids(one))

    print("\n3-shot:")
    print(get_ids(three))

    print("\n5-shot:")
    print(get_ids(five))

    assert get_ids(one) == get_ids(three)[:1]
    assert get_ids(three) == get_ids(five)[:3]

    repeat = select_random_demos(
        mmlu_demo,
        target_1,
        5,
        seed=42,
    )

    assert get_ids(five) == get_ids(repeat)

    other_target = select_random_demos(
        mmlu_demo,
        target_2,
        5,
        seed=42,
    )

    print("\nTarget 2:")
    print(target_2["item_id"])

    print("\n5-shot:")
    print(get_ids(other_target))

    assert get_ids(five) != get_ids(other_target)

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
    print("QUALITY RANDOM SELECTOR TEST")
    print("=" * 70)

    three = select_random_demos(
        quality_demo,
        target,
        3,
        seed=42,
    )

    print("\nTarget:")
    print(
        target["item_id"],
        "| article:",
        target["article_id"]
    )

    print("\n3-shot:")

    for demo in three:

        print(
            demo["item_id"],
            "| article:",
            demo["article_id"]
        )

    article_ids = [
        demo["article_id"]
        for demo in three
    ]

    # Tutti gli articoli demo devono essere diversi.
    assert len(article_ids) == len(set(article_ids))

    # Nessuna demo può avere l'articolo target.
    assert target["article_id"] not in article_ids

    print("\nQuALITY checks: OK")

    print("\n" + "=" * 70)
    print("ALL RANDOM SELECTOR TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()