import json


MMLU_SPLIT = (
    "data/splits/mmlu_pro_full.jsonl"
)

QUALITY_SPLIT = (
    "data/splits/quality_full.jsonl"
)


MODELS = [
    "qwen",
    "granite",
    "gemma",
]


MMLU_CONDITIONS = [
    (0, "none", None),
    (1, "random", 42),
    (1, "semantic", None),
    (3, "random", 42),
    (3, "semantic", None),
    (5, "random", 42),
    (5, "semantic", None),
]


QUALITY_CONDITIONS = [
    (0, "none", None),
    (1, "random", 42),
    (1, "semantic", None),
    (3, "random", 42),
    (3, "semantic", None),
]


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


mmlu_items = load_jsonl(
    MMLU_SPLIT
)

quality_items = load_jsonl(
    QUALITY_SPLIT
)


mmlu_n = len(
    mmlu_items
)

quality_n = len(
    quality_items
)


mmlu_requests_per_model = (
    mmlu_n
    * len(MMLU_CONDITIONS)
)

quality_requests_per_model = (
    quality_n
    * len(QUALITY_CONDITIONS)
)


total_mmlu_requests = (
    mmlu_requests_per_model
    * len(MODELS)
)

total_quality_requests = (
    quality_requests_per_model
    * len(MODELS)
)

total_main_requests = (
    total_mmlu_requests
    + total_quality_requests
)


print("=" * 90)
print("FULL EXPERIMENT PLAN CHECK")
print("=" * 90)

print()
print("MMLU-PRO")
print("-" * 90)

print(
    f"Items: "
    f"{mmlu_n}"
)

print(
    f"Conditions per model: "
    f"{len(MMLU_CONDITIONS)}"
)

print()

for model in MODELS:
    print(
        f"{model:<10} -> "
        f"{mmlu_n} x "
        f"{len(MMLU_CONDITIONS)} "
        f"= "
        f"{mmlu_requests_per_model}"
    )

print()

print(
    f"MMLU total requests: "
    f"{total_mmlu_requests}"
)

print()
print("=" * 90)

print()
print("QUALITY")
print("-" * 90)

print(
    f"Items: "
    f"{quality_n}"
)

print(
    f"Conditions per model: "
    f"{len(QUALITY_CONDITIONS)}"
)

print()

for model in MODELS:
    print(
        f"{model:<10} -> "
        f"{quality_n} x "
        f"{len(QUALITY_CONDITIONS)} "
        f"= "
        f"{quality_requests_per_model}"
    )

print()

print(
    f"QUALITY total requests: "
    f"{total_quality_requests}"
)

print()
print("=" * 90)

print()
print(
    f"TOTAL MAIN REQUESTS: "
    f"{total_main_requests}"
)

print()

expected_total = 29625

if total_main_requests == expected_total:
    print(
        "PLAN STATUS: OK"
    )
else:
    print(
        "PLAN STATUS: CHECK REQUIRED"
    )

print("=" * 90)