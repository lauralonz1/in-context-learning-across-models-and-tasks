import statistics

from quality import load_quality_standardized


def word_count(text):
    return len(text.split())


def main():

    dataset = load_quality_standardized()

    train = dataset["train"]
    validation = dataset["validation"]

    print("=" * 70)
    print("QUALITY — PROTOCOL CHECK")
    print("=" * 70)

    # =========================================================
    # 1. ARTICLE OVERLAP TRA TRAIN E VALIDATION
    # =========================================================

    train_articles = {
        ex["article_id"]
        for ex in train
    }

    validation_articles = {
        ex["article_id"]
        for ex in validation
    }

    overlap = train_articles.intersection(
        validation_articles
    )

    print("\n--- ARTICLE OVERLAP ---")

    print(
        f"Train articles:      {len(train_articles)}"
    )

    print(
        f"Validation articles: {len(validation_articles)}"
    )

    print(
        f"Overlap:             {len(overlap)}"
    )

    # =========================================================
    # 2. TESTO UNICO PER ARTICOLO
    # =========================================================

    train_article_text = {}

    for ex in train:
        train_article_text[
            ex["article_id"]
        ] = ex["context"]

    validation_article_text = {}

    for ex in validation:
        validation_article_text[
            ex["article_id"]
        ] = ex["context"]

    train_lengths = [
        word_count(text)
        for text in train_article_text.values()
    ]

    validation_lengths = [
        word_count(text)
        for text in validation_article_text.values()
    ]

    # =========================================================
    # 3. STIMA PROMPT ICL
    # =========================================================
    #
    # Una demonstration completa QuALITY contiene
    # necessariamente almeno:
    #
    # article + question + options + answer
    #
    # Per ora stimiamo usando soprattutto la lunghezza
    # dell'articolo, che domina completamente il prompt.
    # =========================================================

    avg_demo_article = statistics.mean(
        train_lengths
    )

    avg_target_article = statistics.mean(
        validation_lengths
    )

    print("\n--- AVERAGE ARTICLE LENGTH ---")

    print(
        f"Train/demo article: "
        f"{avg_demo_article:.2f} words"
    )

    print(
        f"Validation/target article: "
        f"{avg_target_article:.2f} words"
    )

    print("\n--- APPROXIMATE ICL PROMPT SIZE ---")

    for shots in [0, 1, 3, 5]:

        estimated_words = (
            avg_target_article
            +
            shots * avg_demo_article
        )

        print(
            f"{shots}-shot: "
            f"~{estimated_words:.0f} words "
            f"(article text only)"
        )

    # =========================================================
    # 4. WORST-CASE GREZZO
    # =========================================================

    max_demo_article = max(train_lengths)
    max_target_article = max(validation_lengths)

    print("\n--- ROUGH WORST CASE ---")

    for shots in [0, 1, 3, 5]:

        estimated_words = (
            max_target_article
            +
            shots * max_demo_article
        )

        print(
            f"{shots}-shot: "
            f"~{estimated_words:.0f} words"
        )


if __name__ == "__main__":
    main()