from collections import defaultdict
from mmlu_pro import load_mmlu_pro_standardized


def normalise_text(text):
    """
    Normalizzazione leggera usata soltanto per confrontare
    domande potenzialmente duplicate.
    """
    return " ".join(text.lower().strip().split())


def main():
    dataset = load_mmlu_pro_standardized()

    test = dataset["test"]
    validation = dataset["validation"]

    print("=" * 70)
    print("MMLU-PRO — DUPLICATE & LEAKAGE CHECK")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Duplicati esatti nel TEST
    # ---------------------------------------------------------

    question_groups = defaultdict(list)

    for ex in test:
        key = normalise_text(ex["question"])
        question_groups[key].append(ex)

    duplicate_groups = {
        question: examples
        for question, examples in question_groups.items()
        if len(examples) > 1
    }

    total_duplicate_extra_rows = sum(
        len(group) - 1
        for group in duplicate_groups.values()
    )

    print("\n--- DUPLICATI NEL TEST ---")
    print(f"Gruppi di domande duplicate: {len(duplicate_groups)}")
    print(f"Righe duplicate extra: {total_duplicate_extra_rows}")

    # ---------------------------------------------------------
    # 2. Controlliamo se i duplicati sono davvero identici
    # ---------------------------------------------------------

    same_everything = 0
    different_options = 0
    different_answer = 0

    for _, group in duplicate_groups.items():
        first = group[0]

        identical = True

        for other in group[1:]:

            if other["options"] != first["options"]:
                different_options += 1
                identical = False

            if other["gold_answer"] != first["gold_answer"]:
                different_answer += 1
                identical = False

        if identical:
            same_everything += 1

    print("\n--- TIPO DI DUPLICATO ---")
    print(f"Gruppi completamente identici: {same_everything}")
    print(f"Con opzioni differenti: {different_options}")
    print(f"Con risposta differente: {different_answer}")

    # ---------------------------------------------------------
    # 3. Mostriamo alcuni esempi
    # ---------------------------------------------------------

    print("\n--- PRIMI 5 GRUPPI DUPLICATI ---")

    for i, (_, group) in enumerate(duplicate_groups.items()):

        if i >= 5:
            break

        print("\n" + "-" * 60)

        for ex in group:
            print(f"ID: {ex['item_id']}")
            print(f"Category: {ex['category']}")
            print(f"Question: {ex['question']}")
            print(f"Gold: {ex['gold_answer']}")
            print(f"Options: {ex['options']}")
            print()

    # ---------------------------------------------------------
    # 4. Leakage validation -> test
    # ---------------------------------------------------------

    test_questions = {
        normalise_text(ex["question"])
        for ex in test
    }

    overlapping_validation = [
        ex
        for ex in validation
        if normalise_text(ex["question"]) in test_questions
    ]

    print("\n" + "=" * 70)
    print("VALIDATION / TEST OVERLAP")
    print("=" * 70)

    print(
        f"Domande validation presenti anche nel test: "
        f"{len(overlapping_validation)}/{len(validation)}"
    )

    if overlapping_validation:

        print("\nEsempi sovrapposti:")

        for ex in overlapping_validation[:10]:
            print(
                f"\n{ex['item_id']} | "
                f"{ex['category']} | "
                f"{ex['question']}"
            )


if __name__ == "__main__":
    main()