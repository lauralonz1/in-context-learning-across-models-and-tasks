from collections import Counter, defaultdict
import statistics

from quality import load_quality_standardized


def audit_split(examples, split_name):

    print(f"\n{'=' * 70}")
    print(f"AUDIT QUALITY - SPLIT: {split_name}")
    print(f"{'=' * 70}")

    print(f"\nNumero totale domande: {len(examples)}")

    # =========================================================
    # 1. ARTICOLI UNICI
    # =========================================================

    article_ids = [ex["article_id"] for ex in examples]

    unique_articles = set(article_ids)

    print("\n--- ARTICOLI ---")
    print(f"Articoli unici: {len(unique_articles)}")

    # =========================================================
    # 2. DOMANDE PER ARTICOLO
    # =========================================================

    questions_per_article = Counter(article_ids)

    qpa_values = list(questions_per_article.values())

    print("\n--- DOMANDE PER ARTICOLO ---")
    print(f"Min: {min(qpa_values)}")
    print(f"Max: {max(qpa_values)}")
    print(f"Media: {statistics.mean(qpa_values):.2f}")
    print(f"Mediana: {statistics.median(qpa_values):.2f}")

    # =========================================================
    # 3. DISTRIBUZIONE HARD / STANDARD
    # =========================================================

    difficulty_counts = Counter(
        "hard" if ex["hard"] else "standard"
        for ex in examples
    )

    print("\n--- DIFFICOLTA ---")

    for difficulty, count in difficulty_counts.items():

        percentage = count / len(examples) * 100

        print(
            f"{difficulty}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # =========================================================
    # 4. DISTRIBUZIONE RISPOSTE
    # =========================================================

    answer_counts = Counter(
        ex["gold_answer"]
        for ex in examples
    )

    print("\n--- GOLD ANSWER ---")

    for answer, count in sorted(answer_counts.items()):

        percentage = count / len(examples) * 100

        print(
            f"{answer}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # =========================================================
    # 5. NUMERO DI OPZIONI
    # =========================================================

    option_counts = Counter(
        len(ex["options"])
        for ex in examples
    )

    print("\n--- NUMERO OPZIONI ---")

    for n_options, count in sorted(option_counts.items()):

        percentage = count / len(examples) * 100

        print(
            f"{n_options}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # =========================================================
    # 6. LUNGHEZZA ARTICOLI
    # =========================================================
    #
    # IMPORTANTE:
    # calcoliamo la lunghezza sugli articoli UNICI,
    # non una volta per ogni domanda.
    #

    unique_article_text = {}

    for ex in examples:
        unique_article_text[ex["article_id"]] = ex["context"]

    article_lengths_words = [
        len(text.split())
        for text in unique_article_text.values()
    ]

    print("\n--- LUNGHEZZA ARTICOLI (PAROLE) ---")
    print(f"Min: {min(article_lengths_words)}")
    print(f"Max: {max(article_lengths_words)}")
    print(
        f"Media: "
        f"{statistics.mean(article_lengths_words):.2f}"
    )
    print(
        f"Mediana: "
        f"{statistics.median(article_lengths_words):.2f}"
    )

    # =========================================================
    # 7. LUNGHEZZA DOMANDE
    # =========================================================

    question_lengths = [
        len(ex["question"].split())
        for ex in examples
    ]

    print("\n--- LUNGHEZZA DOMANDE (PAROLE) ---")
    print(f"Min: {min(question_lengths)}")
    print(f"Max: {max(question_lengths)}")
    print(
        f"Media: "
        f"{statistics.mean(question_lengths):.2f}"
    )

    # =========================================================
    # 8. DUPLICATI DOMANDA
    # =========================================================

    normalized_questions = [
        " ".join(
            ex["question"]
            .lower()
            .strip()
            .split()
        )
        for ex in examples
    ]

    duplicate_questions = (
        len(normalized_questions)
        - len(set(normalized_questions))
    )

    print("\n--- DUPLICATI DOMANDA ---")
    print(
        f"Domande duplicate esatte/normalizzate: "
        f"{duplicate_questions}"
    )

    # =========================================================
    # 9. MISSING VALUES
    # =========================================================

    missing_counts = Counter()

    for ex in examples:

        for key, value in ex.items():

            if value is None or value == "" or value == []:
                missing_counts[key] += 1

    print("\n--- MISSING VALUES ---")

    if missing_counts:

        for key, count in missing_counts.items():
            print(f"{key}: {count}")

    else:
        print("Nessun missing value trovato.")


def main():

    dataset = load_quality_standardized()

    for split_name, examples in dataset.items():
        audit_split(examples, split_name)


if __name__ == "__main__":
    main()