import json
import os
from collections import Counter

from mmlu_pro import load_mmlu_pro_standardized


OUTPUT_DIR = "data/splits"


def normalise_question(text):
    """
    Normalizzazione usata per identificare la stessa underlying question.
    Non modifica il testo che verrà poi mostrato al modello.
    """
    return " ".join(text.lower().strip().split())


def save_jsonl(path, examples):
    """
    Salva una lista di dizionari in formato JSONL.
    Una riga JSON per ogni esempio.
    """
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")


def print_category_distribution(examples, name):
    counts = Counter(ex["category"] for ex in examples)

    print(f"\n--- Distribuzione categorie: {name} ---")

    for category, count in sorted(counts.items()):
        percentage = count / len(examples) * 100
        print(
            f"{category:20s} "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_mmlu_pro_standardized()

    validation = dataset["validation"]
    test = dataset["test"]

    print("=" * 70)
    print("BUILDING MMLU-PRO SPLITS")
    print("=" * 70)

    # =========================================================
    # 1. DEMONSTRATION POOL
    # =========================================================
    #
    # Usiamo l'intero validation split ufficiale.
    # Sono 70 esempi:
    # 14 categorie x 5 esempi ciascuna.
    #
    # Tutti hanno 10 opzioni e CoT disponibile.
    #

    demo_pool = validation.copy()

    print(f"\nValidation originale: {len(validation)}")
    print(f"Demo pool finale:      {len(demo_pool)}")

    # =========================================================
    # 2. EVALUATION POOL
    # =========================================================
    #
    # Step A:
    # teniamo soltanto item con esattamente 10 opzioni.
    #
    # In questo modo:
    # - stessa chance baseline = 10%
    # - stesso answer space A-J
    # - confronto più pulito
    #

    ten_option_test = [
        ex
        for ex in test
        if len(ex["options"]) == 10
    ]

    print(f"\nTest originale:                {len(test)}")
    print(f"Test con esattamente 10 opzioni: {len(ten_option_test)}")

    # =========================================================
    # Step B:
    # deduplicazione per testo della domanda.
    #
    # Se lo stesso stem appare più volte con differenti
    # set/ordine/formulazione delle opzioni, manteniamo
    # deterministicamente la prima variante presente.
    #
    # Non stiamo dicendo che le altre siano sbagliate:
    # semplicemente evitiamo che la stessa domanda pesi
    # più volte nell'evaluation.
    # =========================================================

    seen_questions = set()
    eval_pool = []

    removed_duplicates = 0

    for ex in ten_option_test:

        key = normalise_question(ex["question"])

        if key in seen_questions:
            removed_duplicates += 1
            continue

        seen_questions.add(key)
        eval_pool.append(ex)

    print(f"Duplicati rimossi:             {removed_duplicates}")
    print(f"Evaluation pool finale:        {len(eval_pool)}")

    # =========================================================
    # 3. CONTROLLO LEAKAGE FINALE
    # =========================================================

    demo_questions = {
        normalise_question(ex["question"])
        for ex in demo_pool
    }

    eval_questions = {
        normalise_question(ex["question"])
        for ex in eval_pool
    }

    overlap = demo_questions.intersection(eval_questions)

    print(f"\nOverlap demo/evaluation: {len(overlap)}")

    if overlap:
        raise ValueError(
            "ERRORE: trovato overlap tra demonstration pool "
            "ed evaluation pool."
        )

    # =========================================================
    # 4. CONTROLLI STRUTTURALI
    # =========================================================

    assert all(
        len(ex["options"]) == 10
        for ex in demo_pool
    )

    assert all(
        len(ex["options"]) == 10
        for ex in eval_pool
    )

    print("\nControlli strutturali: OK")

    # =========================================================
    # 5. DISTRIBUZIONE CATEGORIE
    # =========================================================

    print_category_distribution(
        demo_pool,
        "DEMO POOL"
    )

    print_category_distribution(
        eval_pool,
        "EVALUATION POOL"
    )

    # =========================================================
    # 6. SALVATAGGIO
    # =========================================================

    demo_path = os.path.join(
        OUTPUT_DIR,
        "mmlu_pro_demo.jsonl"
    )

    eval_path = os.path.join(
        OUTPUT_DIR,
        "mmlu_pro_eval.jsonl"
    )

    save_jsonl(
        demo_path,
        demo_pool
    )

    save_jsonl(
        eval_path,
        eval_pool
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(demo_path)
    print(eval_path)


if __name__ == "__main__":
    main()