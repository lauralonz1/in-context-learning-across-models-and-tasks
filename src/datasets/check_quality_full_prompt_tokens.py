import statistics

from transformers import AutoTokenizer

from src.datasets.quality import load_quality_standardized
from src.prompts.quality_prompt import build_quality_prompt


MODELS = {
    "qwen": "Qwen/Qwen3-4B",
    "phi": "microsoft/Phi-4-mini-instruct",
    "gemma": "google/gemma-3-4b-it",
}


def get_unique_article_examples(examples):
    """
    Tiene un solo esempio per articolo.
    Utile per evitare di usare 3 demo dallo stesso documento.
    """
    seen = set()
    result = []

    for ex in examples:

        if ex["article_id"] in seen:
            continue

        seen.add(ex["article_id"])
        result.append(ex)

    return result


def count_tokens(tokenizer, text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


def analyse_model(model_name, tokenizer_id, train, validation):

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    demo_candidates = get_unique_article_examples(train)

    token_results = {
        0: [],
        1: [],
        3: [],
        5: [],
    }

    # Per ora usiamo i primi target e le prime demo
    # in modo deterministico.
    targets = validation[:100]

    for target_index, target in enumerate(targets):

        # Evitiamo qualunque eventuale stesso articolo
        available_demos = [
            ex
            for ex in demo_candidates
            if ex["article_id"] != target["article_id"]
        ]

        for shots in [0, 1, 3, 5]:

            demos = available_demos[:shots]

            prompt = build_quality_prompt(
                target=target,
                demonstrations=demos
            )

            n_tokens = count_tokens(
                tokenizer,
                prompt
            )

            token_results[shots].append(
                n_tokens
            )

    print("\n--- FULL PROMPT TOKEN LENGTH ---")

    for shots in [0, 1, 3, 5]:

        values = token_results[shots]

        print(
            f"\n{shots}-shot"
        )

        print(
            f"Min:    {min(values)}"
        )

        print(
            f"Mean:   {statistics.mean(values):.0f}"
        )

        print(
            f"Median: {statistics.median(values):.0f}"
        )

        print(
            f"Max:    {max(values)}"
        )


def main():

    dataset = load_quality_standardized()

    train = dataset["train"]
    validation = dataset["validation"]

    for model_name, tokenizer_id in MODELS.items():

        try:

            analyse_model(
                model_name,
                tokenizer_id,
                train,
                validation
            )

        except Exception as e:

            print("\n" + "=" * 70)
            print(f"MODEL: {model_name}")
            print("=" * 70)

            print(
                f"Tokenizer non disponibile: "
                f"{type(e).__name__}"
            )

            print(str(e))


if __name__ == "__main__":
    main()