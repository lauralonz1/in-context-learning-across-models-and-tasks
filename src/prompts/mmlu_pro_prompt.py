def index_to_letter(index):
    """
    0 -> A
    1 -> B
    ...
    9 -> J
    """
    return chr(ord("A") + index)


def format_options(options):
    """
    Converte le opzioni in:

    A. ...
    B. ...
    ...
    """

    return "\n".join(
        f"{index_to_letter(i)}. {option}"
        for i, option in enumerate(options)
    )


def format_demo(example):
    """
    Formatta una demonstration MMLU-Pro.
    """

    return f"""
### DEMONSTRATION

Question:
{example["question"]}

Options:
{format_options(example["options"])}

Correct answer:
{example["gold_answer"]}
""".strip()


def format_target(example):
    """
    Formatta il target senza includere il gold.
    """

    return f"""
### TARGET QUESTION

Question:
{example["question"]}

Options:
{format_options(example["options"])}

Answer with only the correct letter.
""".strip()


def build_mmlu_pro_prompt(target, demonstrations=None):
    """
    Costruisce il prompt completo MMLU-Pro.
    """

    if demonstrations is None:
        demonstrations = []

    parts = [
        (
            "You are answering a multiple-choice question.\n"
            "Choose the single best answer.\n"
            "Return only the correct option letter."
        )
    ]

    for demo in demonstrations:
        parts.append(
            format_demo(demo)
        )

    parts.append(
        format_target(target)
    )

    return "\n\n".join(parts)