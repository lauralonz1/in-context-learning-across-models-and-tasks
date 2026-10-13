from datasets import load_dataset


def load_raw_mmlu_pro():
    """
    Carica MMLU-Pro da Hugging Face.
    """
    return load_dataset("TIGER-Lab/MMLU-Pro")


def standardize_example(example, split_name):
    """
    Converte un esempio MMLU-Pro nel formato standard del progetto.
    """
    return {
        "item_id": f"mmlu_pro_{split_name}_{example['question_id']}",
        "benchmark": "mmlu_pro",
        "split": split_name,
        "question": example["question"],
        "context": None,
        "options": example["options"],
        "gold_answer": example["answer"],
        "gold_index": example["answer_index"],
        "category": example["category"],
        "source": example["src"],
        "cot_content": example["cot_content"],
    }


def load_mmlu_pro_standardized():
    """
    Carica MMLU-Pro e standardizza tutti gli esempi.
    """
    dataset = load_raw_mmlu_pro()

    standardized = {}

    for split_name, split_data in dataset.items():
        standardized[split_name] = [
            standardize_example(example, split_name)
            for example in split_data
        ]

    return standardized


if __name__ == "__main__":
    dataset = load_mmlu_pro_standardized()

    print("\n=== MMLU-PRO STANDARDIZZATO ===")

    for split_name, examples in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Numero esempi: {len(examples)}")

        print("\nPrimo esempio standardizzato:")
        print(examples[0])

        print("\nCampi disponibili:")
        print(list(examples[0].keys()))