from src.evaluation.parser import parse_answer


def main():

    tests = [
        # Risposte ideali
        ("C", "C"),
        (" c ", "C"),
        ("(D)", "D"),
        ("D.", "D"),
        ("[E]", "E"),

        # Risposte che iniziano con lettera + punteggiatura
        (
            "B. The university raises tuition.",
            "B",
        ),
        (
            "G. Encoding.",
            "G",
        ),
        (
            "D) Some explanation",
            "D",
        ),
        (
            "C: final answer",
            "C",
        ),
        (
            "A - first option",
            "A",
        ),

        # Pattern espliciti
        ("Answer: C", "C"),
        ("Answer is F", "F"),
        ("Final answer: J", "J"),
        ("Correct answer is B", "B"),
        ("Option G", "G"),

        # Output leggermente verbosi ma interpretabili
        (
            "After considering the options, the answer is F.",
            "F",
        ),
        (
            "Therefore: H",
            "H",
        ),
        (
            "I choose C",
            "C",
        ),

        # Nessuna risposta valida
        ("I don't know.", None),
        ("I am not sure.", None),
        ("This question is difficult.", None),
        ("", None),
        (None, None),

        # Ambigui
        (
            "I considered A and B.",
            None,
        ),
        (
            "It could be C or D.",
            None,
        ),
    ]

    print("=" * 70)
    print("PARSER TEST")
    print("=" * 70)

    for text, expected in tests:

        result = parse_answer(text)

        print(
            f"{repr(text):45s} "
            f"-> {result} "
            f"(expected {expected})"
        )

        assert result == expected

    print("\n" + "=" * 70)
    print("ALL PARSER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()