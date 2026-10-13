from collections import Counter
from mmlu_pro import load_mmlu_pro_standardized


def audit_split(examples, split_name):
    print(f"\n{'=' * 60}")
    print(f"AUDIT SPLIT: {split_name}")
    print(f"{'=' * 60}")

    print(f"\nNumero totale di esempi: {len(examples)}")

    # 1. Missing values
    missing_counts = Counter()

    for ex in examples:
        for key, value in ex.items():
            if value is None or value == "" or value == []:
                missing_counts[key] += 1

    print("\n--- Missing values ---")
    if missing_counts:
        for key, count in missing_counts.items():
            print(f"{key}: {count}")
    else:
        print("Nessun missing value trovato.")

    # 2. Distribuzione categorie
    category_counts = Counter(ex["category"] for ex in examples)

    print("\n--- Categorie ---")
    print(f"Numero categorie: {len(category_counts)}")

    for category, count in category_counts.most_common():
        print(f"{category}: {count}")

    # 3. Distribuzione delle risposte corrette
    answer_counts = Counter(ex["gold_answer"] for ex in examples)

    print("\n--- Distribuzione gold answer ---")
    for answer, count in sorted(answer_counts.items()):
        percentage = count / len(examples) * 100
        print(f"{answer}: {count} ({percentage:.2f}%)")

    # 4. Numero di opzioni
    option_counts = Counter(len(ex["options"]) for ex in examples)

    print("\n--- Numero di opzioni ---")
    for n_options, count in sorted(option_counts.items()):
        percentage = count / len(examples) * 100
        print(f"{n_options} opzioni: {count} ({percentage:.2f}%)")

    # 5. Lunghezza domande
    question_lengths = [len(ex["question"].split()) for ex in examples]

    print("\n--- Lunghezza domande (parole) ---")
    print(f"Min: {min(question_lengths)}")
    print(f"Max: {max(question_lengths)}")
    print(f"Media: {sum(question_lengths) / len(question_lengths):.2f}")

    # 6. Duplicati esatti nelle domande
    questions = [ex["question"] for ex in examples]
    duplicate_count = len(questions) - len(set(questions))

    print("\n--- Duplicati ---")
    print(f"Domande duplicate esatte: {duplicate_count}")

    # 7. CoT disponibile
    cot_available = sum(
        1 for ex in examples
        if ex["cot_content"] is not None and ex["cot_content"].strip() != ""
    )

    print("\n--- CoT content ---")
    print(f"Con CoT disponibile: {cot_available}/{len(examples)}")
    print(f"Percentuale: {cot_available / len(examples) * 100:.2f}%")


def main():
    dataset = load_mmlu_pro_standardized()

    for split_name, examples in dataset.items():
        audit_split(examples, split_name)


if __name__ == "__main__":
    main()