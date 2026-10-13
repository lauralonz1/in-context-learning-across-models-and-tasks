import hashlib

from datasets import load_dataset


def load_raw_quality():
    """
    Carica QuALITY da Hugging Face.
    """
    return load_dataset("emozilla/quality")


def make_article_id(article):
    """
    Crea un identificativo deterministico per l'articolo.

    Lo stesso articolo compare in più righe perché contiene
    più domande. L'hash ci permette di riconoscere tutte
    le domande appartenenti allo stesso documento.
    """
    return hashlib.sha1(
        article.encode("utf-8")
    ).hexdigest()[:12]


def index_to_letter(index):
    """
    Converte:
    0 -> A
    1 -> B
    2 -> C
    3 -> D
    """
    return chr(ord("A") + index)


def standardize_example(example, split_name, row_index):
    """
    Converte un esempio QuALITY nel formato standard
    utilizzato dal progetto.

    ATTENZIONE:
    nel dataset originale QuALITY 'answer' è 1-based:
        1 = prima opzione
        2 = seconda opzione
        ...
    Nel nostro formato interno usiamo invece gold_index 0-based.
    """

    gold_index = example["answer"]

    if gold_index < 0 or gold_index >= len(example["options"]):
        raise ValueError(
            f"Answer index non valido alla riga {row_index}: "
            f"{example['answer']}"
        )

    return {
        "item_id": f"quality_{split_name}_{row_index}",
        "benchmark": "quality",
        "split": split_name,

        # QuALITY è context-dependent.
        "context": example["article"],
        "article_id": make_article_id(example["article"]),

        "question": example["question"],
        "options": example["options"],

        "gold_answer": index_to_letter(gold_index),
        "gold_index": gold_index,

        # Non abbiamo categorie tematiche come in MMLU-Pro.
        "category": "hard" if example["hard"] else "standard",

        "hard": example["hard"],
        "source": "QuALITY",

        # Non presente nel dataset.
        "cot_content": None,
    }


def load_quality_standardized():
    """
    Carica e standardizza tutti gli split di QuALITY.
    """

    dataset = load_raw_quality()

    standardized = {}

    for split_name, split_data in dataset.items():

        standardized[split_name] = [
            standardize_example(
                example,
                split_name,
                row_index
            )
            for row_index, example in enumerate(split_data)
        ]

    return standardized


if __name__ == "__main__":

    dataset = load_quality_standardized()

    print("\n=== QUALITY STANDARDIZZATO ===")

    for split_name, examples in dataset.items():

        print(f"\nSplit: {split_name}")
        print(f"Numero esempi: {len(examples)}")

        print("\nPrimo esempio standardizzato:")

        # Evitiamo di stampare tutto l'articolo enorme.
        first = examples[0]

        preview = first.copy()
        preview["context"] = (
            preview["context"][:500]
            + "..."
        )

        print(preview)

        print("\nCampi disponibili:")
        print(list(first.keys()))