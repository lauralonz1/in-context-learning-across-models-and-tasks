from collections import defaultdict
from mmlu_pro import load_mmlu_pro_standardized


def normalise_text(text):
    return " ".join(text.lower().strip().split())


def get_gold_text(example):
    """
    Restituisce il testo dell'opzione corretta.
    """
    return example["options"][example["gold_index"]]


def main():
    dataset = load_mmlu_pro_standardized()
    test = dataset["test"]

    groups = defaultdict(list)

    for ex in test:
        key = normalise_text(ex["question"])
        groups[key].append(ex)

    duplicate_groups = {
        question: examples
        for question, examples in groups.items()
        if len(examples) > 1
    }

    same_gold_text = 0
    different_gold_text = 0

    problematic_groups = []

    for question, group in duplicate_groups.items():

        gold_texts = {
            normalise_text(get_gold_text(ex))
            for ex in group
        }

        if len(gold_texts) == 1:
            same_gold_text += 1
        else:
            different_gold_text += 1
            problematic_groups.append((question, group))

    print("=" * 70)
    print("MMLU-PRO — DUPLICATE GOLD ANSWER CHECK")
    print("=" * 70)

    print(f"\nGruppi duplicati totali: {len(duplicate_groups)}")
    print(f"Stesso testo della risposta corretta: {same_gold_text}")
    print(f"Testo della risposta corretta differente: {different_gold_text}")

    print("\n--- PRIMI 10 GRUPPI CON GOLD TEXT DIVERSO ---")

    for question, group in problematic_groups[:10]:

        print("\n" + "-" * 70)
        print(f"QUESTION: {question}")

        for ex in group:
            print(f"\nID: {ex['item_id']}")
            print(f"Gold letter: {ex['gold_answer']}")
            print(f"Gold index: {ex['gold_index']}")
            print(f"Gold text: {get_gold_text(ex)}")
            print(f"Source: {ex['source']}")


if __name__ == "__main__":
    main()