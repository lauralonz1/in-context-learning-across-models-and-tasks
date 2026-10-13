def format_options(options):
    """
    Converte una lista di 4 opzioni in formato:
    A. ...
    B. ...
    C. ...
    D. ...
    """
    letters = ["A", "B", "C", "D"]

    return "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(letters, options)
    )


def format_demo(example):
    """
    Formatta una demonstration completa QuALITY.
    """

    return f"""
### DEMONSTRATION

Article:
{example["context"]}

Question:
{example["question"]}

Options:
{format_options(example["options"])}

Correct answer:
{example["gold_answer"]}
""".strip()


def format_target(example):
    """
    Formatta l'item target.
    La risposta NON viene inclusa.
    """

    return f"""
### TARGET QUESTION

Article:
{example["context"]}

Question:
{example["question"]}

Options:
{format_options(example["options"])}

Answer with only the correct letter: A, B, C, or D.
""".strip()


def build_quality_prompt(target, demonstrations=None):
    """
    Costruisce il prompt completo per QuALITY.

    demonstrations:
        lista di esempi few-shot.
        Se None o lista vuota -> zero-shot.
    """

    if demonstrations is None:
        demonstrations = []

    parts = [
        (
            "You are answering multiple-choice reading comprehension questions.\n"
            "Use the article to answer the question.\n"
            "Return only the correct option letter."
        )
    ]

    for demo in demonstrations:
        parts.append(format_demo(demo))

    parts.append(format_target(target))

    return "\n\n".join(parts)