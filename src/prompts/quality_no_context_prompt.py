def build_quality_no_context_prompt(
    target,
):
    """
    Build a QuALITY multiple-choice prompt
    without the article/context.

    Used only for the context-ablation experiment.
    """

    question = target["question"]

    options = target["options"]

    option_lines = []

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    for letter, option in zip(
        letters,
        options,
    ):
        option_lines.append(
            f"{letter}. {option}"
        )

    options_text = "\n".join(
        option_lines
    )

    prompt = (
        "Answer the following multiple-choice "
        "question.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Options:\n"
        f"{options_text}\n\n"
        "Respond with only the letter of the "
        "correct answer: A, B, C, or D."
    )

    return prompt