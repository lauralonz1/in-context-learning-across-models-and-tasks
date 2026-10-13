import json

from src.prompts.mmlu_pro_prompt import (
    build_mmlu_pro_prompt,
)

from src.selection.selector import DemoSelector


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


def main():

    selector = DemoSelector()

    demo_pool = load_jsonl(
        "data/splits/mmlu_pro_demo.jsonl"
    )

    smoke_items = load_jsonl(
        "data/splits/mmlu_pro_smoke.jsonl"
    )

    target = smoke_items[0]

    semantic_demo = selector.select(
        demo_pool=demo_pool,
        target=target,
        n_shots=1,
        strategy="semantic",
    )

    print("=" * 70)
    print("ZERO-SHOT PROMPT")
    print("=" * 70)

    zero_prompt = build_mmlu_pro_prompt(
        target=target,
        demonstrations=[],
    )

    print(zero_prompt)

    print("\n" + "=" * 70)
    print("ONE-SHOT SEMANTIC PROMPT")
    print("=" * 70)

    one_prompt = build_mmlu_pro_prompt(
        target=target,
        demonstrations=semantic_demo,
    )

    print(one_prompt)

    
    assert "Correct answer:" not in zero_prompt

   
    assert one_prompt.count(
        "Correct answer:"
    ) == 1

    print("\n" + "=" * 70)
    print("MMLU PROMPT TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()