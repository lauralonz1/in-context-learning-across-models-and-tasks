import json
from collections import Counter, defaultdict
from statistics import mean


RESULT_PATH = (
    "results/raw_predictions/"
    "mmlu_pro_full_granite.jsonl"
)

SPLIT_PATH = (
    "data/splits/"
    "mmlu_pro_full.jsonl"
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


def main():
    records = load_jsonl(
        RESULT_PATH
    )

    split_records = load_jsonl(
        SPLIT_PATH
    )

    item_to_category = {
        record["item_id"]: record["category"]
        for record in split_records
    }

    print("=" * 80)
    print("MMLU-PRO FULL ANALYSIS - GRANITE")
    print("=" * 80)

    print(
        f"Total records: "
        f"{len(records)}"
    )

    # ==========================================================
    # UNIQUE RUN CHECK
    # ==========================================================

    keys = [
        (
            r["item_id"],
            r["model"],
            r["shots"],
            r["strategy"],
            r["seed"],
        )
        for r in records
    ]

    unique_keys = set(keys)

    print(
        f"Unique run keys: "
        f"{len(unique_keys)}"
    )

    print(
        f"Duplicate records: "
        f"{len(keys) - len(unique_keys)}"
    )

    # ==========================================================
    # ERRORS
    # ==========================================================

    error_records = [
        r
        for r in records
        if r["error"] is not None
    ]

    valid_records = [
        r
        for r in records
        if r["error"] is None
    ]

    print(
        f"API/runtime errors: "
        f"{len(error_records)}"
    )

    print(
        f"Valid records: "
        f"{len(valid_records)}"
    )

    # ==========================================================
    # PARSE FAILURES
    # ==========================================================

    parse_failures = [
        r
        for r in valid_records
        if r["parse_failure"]
    ]

    print(
        f"Parse failures: "
        f"{len(parse_failures)}"
    )

    # ==========================================================
    # PROVIDERS
    # ==========================================================

    providers = Counter(
        r["provider"]
        for r in valid_records
        if r["provider"] is not None
    )

    print()
    print("PROVIDERS")
    print("-" * 80)

    for provider, count in sorted(
        providers.items()
    ):
        print(
            f"{provider}: "
            f"{count}"
        )

    # ==========================================================
    # CONDITION ACCURACY
    # ==========================================================

    grouped = defaultdict(list)

    for r in valid_records:
        key = (
            r["shots"],
            r["strategy"],
            r["seed"],
        )

        grouped[key].append(r)

    print()
    print("ACCURACY BY RUN CONDITION")
    print("-" * 80)

    for key in sorted(
        grouped.keys(),
        key=lambda x: (
            x[0],
            x[1],
            -1
            if x[2] is None
            else x[2],
        ),
    ):
        shots, strategy, seed = key

        group = grouped[key]

        correct = sum(
            int(r["correct"])
            for r in group
        )

        total = len(group)

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        print(
            f"{shots}-shot | "
            f"{strategy:<8} | "
            f"seed={str(seed):<4} | "
            f"{correct:>4}/{total:<4} | "
            f"accuracy={accuracy:.3f}"
        )

    # ==========================================================
    # CATEGORY ACCURACY
    # ==========================================================

    category_grouped = defaultdict(
        list
    )

    for r in valid_records:
        category = item_to_category.get(
            r["item_id"]
        )

        if category is not None:
            key = (
                category,
                r["shots"],
                r["strategy"],
                r["seed"],
            )

            category_grouped[
                key
            ].append(r)

    print()
    print("ACCURACY BY CATEGORY AND CONDITION")
    print("-" * 80)

    categories = sorted(
        {
            key[0]
            for key in category_grouped
        }
    )

    for category in categories:
        print()
        print(
            f"[{category}]"
        )

        category_keys = [
            key
            for key in category_grouped
            if key[0] == category
        ]

        for key in sorted(
            category_keys,
            key=lambda x: (
                x[1],
                x[2],
                -1
                if x[3] is None
                else x[3],
            ),
        ):
            (
                _,
                shots,
                strategy,
                seed,
            ) = key

            group = (
                category_grouped[
                    key
                ]
            )

            correct = sum(
                int(r["correct"])
                for r in group
            )

            total = len(group)

            accuracy = (
                correct / total
                if total > 0
                else 0
            )

            print(
                f"  {shots}-shot | "
                f"{strategy:<8} | "
                f"seed={str(seed):<4} | "
                f"{correct:>3}/{total:<3} | "
                f"accuracy="
                f"{accuracy:.3f}"
            )

    # ==========================================================
    # LATENCY
    # ==========================================================

    latencies = [
        r["latency_seconds"]
        for r in valid_records
        if (
            r["latency_seconds"]
            is not None
        )
    ]

    print()
    print("LATENCY")
    print("-" * 80)

    if latencies:
        print(
            f"Mean latency: "
            f"{mean(latencies):.3f} s"
        )

        print(
            f"Min latency: "
            f"{min(latencies):.3f} s"
        )

        print(
            f"Max latency: "
            f"{max(latencies):.3f} s"
        )

    # ==========================================================
    # TOKEN USAGE
    # ==========================================================

    prompt_tokens = [
        r["prompt_tokens"]
        for r in valid_records
        if (
            r["prompt_tokens"]
            is not None
        )
    ]

    completion_tokens = [
        r["completion_tokens"]
        for r in valid_records
        if (
            r["completion_tokens"]
            is not None
        )
    ]

    total_tokens = [
        r["total_tokens"]
        for r in valid_records
        if (
            r["total_tokens"]
            is not None
        )
    ]

    print()
    print("TOKEN USAGE")
    print("-" * 80)

    if prompt_tokens:
        print(
            f"Mean prompt tokens: "
            f"{mean(prompt_tokens):.1f}"
        )

        print(
            f"Total prompt tokens: "
            f"{sum(prompt_tokens)}"
        )

    if completion_tokens:
        print(
            f"Mean completion tokens: "
            f"{mean(completion_tokens):.1f}"
        )

        print(
            f"Total completion tokens: "
            f"{sum(completion_tokens)}"
        )

    if total_tokens:
        print(
            f"Total tokens overall: "
            f"{sum(total_tokens)}"
        )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()