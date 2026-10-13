import json

from src.selection.selector import DemoSelector


def load_jsonl(path):
    with open(
        path,
        "r",
        encoding="utf-8",
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

    selector = DemoSelector()

    mmlu_demo = load_jsonl(
        "data/splits/mmlu_pro_demo.jsonl"
    )

    mmlu_eval = load_jsonl(
        "data/splits/mmlu_pro_eval.jsonl"
    )

    target = mmlu_eval[0]

    print("=" * 70)
    print("UNIFIED SELECTOR TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # ZERO-SHOT
    # ---------------------------------------------------------

    zero = selector.select(
        demo_pool=mmlu_demo,
        target=target,
        n_shots=0,
        strategy="none",
    )

    print("\n0-shot:")
    print(zero)

    assert zero == []

    # ---------------------------------------------------------
    # RANDOM
    # ---------------------------------------------------------

    random_three = selector.select(
        demo_pool=mmlu_demo,
        target=target,
        n_shots=3,
        strategy="random",
        seed=42,
    )

    print("\n3-shot RANDOM:")
    print(get_ids(random_three))

    assert len(random_three) == 3

    # ---------------------------------------------------------
    # SEMANTIC
    # ---------------------------------------------------------

    semantic_three = selector.select(
        demo_pool=mmlu_demo,
        target=target,
        n_shots=3,
        strategy="semantic",
    )

    print("\n3-shot SEMANTIC:")
    print(get_ids(semantic_three))

    assert len(semantic_three) == 3

    # ---------------------------------------------------------
    # RANDOM VS SEMANTIC
    # ---------------------------------------------------------

    assert (
        get_ids(random_three)
        != get_ids(semantic_three)
    )

    print("\nRandom != Semantic: OK")

    print("\n" + "=" * 70)
    print("ALL UNIFIED SELECTOR TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()