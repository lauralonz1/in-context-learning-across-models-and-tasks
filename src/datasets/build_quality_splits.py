import json
import os
from collections import Counter

from src.datasets.quality import load_quality_standardized


OUTPUT_DIR = "data/splits"


def save_jsonl(path, examples):
    """
    Salva gli esempi in formato JSONL.
    """
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")


def print_stats(examples, name):
    print(f"\n--- {name} ---")
    print(f"Numero domande: {len(examples)}")

    article_ids = {
        ex["article_id"]
        for ex in examples
    }

    print(f"Articoli unici: {len(article_ids)}")

    difficulty = Counter(
        "hard" if ex["hard"] else "standard"
        for ex in examples
    )

    print("\nDifficolta:")

    for label, count in difficulty.items():
        percentage = count / len(examples) * 100
        print(
            f"{label}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    answers = Counter(
        ex["gold_answer"]
        for ex in examples
    )

    print("\nGold answers:")

    for answer, count in sorted(answers.items()):
        percentage = count / len(examples) * 100
        print(
            f"{answer}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    dataset = load_quality_standardized()

    train = dataset["train"]
    validation = dataset["validation"]

    # =========================================================
    # DEMO / EVALUATION POOL
    # =========================================================

    demo_pool = train.copy()
    eval_pool = validation.copy()

    # =========================================================
    # LEAKAGE CHECK A LIVELLO DI ARTICOLO
    # =========================================================

    demo_articles = {
        ex["article_id"]
        for ex in demo_pool
    }

    eval_articles = {
        ex["article_id"]
        for ex in eval_pool
    }

    overlap = demo_articles.intersection(
        eval_articles
    )

    print("=" * 70)
    print("BUILDING QUALITY SPLITS")
    print("=" * 70)

    print(
        f"\nArticle overlap demo/eval: "
        f"{len(overlap)}"
    )

    if overlap:
        raise ValueError(
            "ERRORE: esistono articoli presenti "
            "sia nel demo pool sia nell'evaluation pool."
        )

    # =========================================================
    # CONTROLLI STRUTTURALI
    # =========================================================

    assert all(
        len(ex["options"]) == 4
        for ex in demo_pool
    )

    assert all(
        len(ex["options"]) == 4
        for ex in eval_pool
    )

    print("Controlli strutturali: OK")

    # =========================================================
    # STATISTICHE
    # =========================================================

    print_stats(
        demo_pool,
        "DEMO POOL"
    )

    print_stats(
        eval_pool,
        "EVALUATION POOL"
    )

    # =========================================================
    # SALVATAGGIO
    # =========================================================

    demo_path = os.path.join(
        OUTPUT_DIR,
        "quality_demo.jsonl"
    )

    eval_path = os.path.join(
        OUTPUT_DIR,
        "quality_eval.jsonl"
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