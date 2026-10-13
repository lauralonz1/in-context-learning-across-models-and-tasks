import re


VALID_LETTERS = list("ABCDEFGHIJ")


def parse_answer(text, valid_letters=None):
    """
    Estrae una risposta multiple-choice dall'output del modello.

    Il parser è volutamente conservativo:
    meglio classificare una risposta ambigua come parse failure
    piuttosto che inventare una lettera.

    Restituisce:
        "A", "B", ..., "J"

    oppure:
        None
    """

    if valid_letters is None:
        valid_letters = VALID_LETTERS

    if text is None:
        return None

    text = text.strip()

    if not text:
        return None

    upper_text = text.upper()

    # =========================================================
    # 1. CASO IDEALE
    #
    # C
    # =========================================================

    if upper_text in valid_letters:
        return upper_text

    # =========================================================
    # 2. LETTERA CON FORMATTAZIONE MINIMA
    #
    # (C)
    # C.
    # [C]
    # =========================================================

    match = re.fullmatch(
        r"[\(\[]?\s*([A-J])\s*[\)\]]?\.?",
        upper_text,
    )

    if match:

        letter = match.group(1)

        if letter in valid_letters:
            return letter

    # =========================================================
    # 3. RISPOSTA CHE INIZIA CON LETTERA + PUNTEGGIATURA
    #
    # B. The university raises tuition.
    # D) Some explanation
    # C: final answer
    # G - Encoding
    #
    # NON accettiamo "I don't know",
    # perché dopo la lettera serve punteggiatura esplicita.
    # =========================================================

    match = re.match(
        r"^\s*([A-J])\s*[\.\)\]:\-]",
        upper_text,
    )

    if match:

        letter = match.group(1)

        if letter in valid_letters:
            return letter

    # =========================================================
    # 4. PATTERN ESPLICITI
    #
    # Answer: C
    # Answer is C
    # Final answer: C
    # Correct answer is C
    # Option C
    # =========================================================

    patterns = [
        r"\bFINAL\s+ANSWER\s*(?:IS|:|-)?\s*\**\(?([A-J])\)?\b",
        r"\bCORRECT\s+ANSWER\s*(?:IS|:|-)?\s*\**\(?([A-J])\)?\b",
        r"\bANSWER\s*(?:IS|:|-)?\s*\**\(?([A-J])\)?\b",
        r"\bOPTION\s*\**\(?([A-J])\)?\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            upper_text,
        )

        if match:

            letter = match.group(1)

            if letter in valid_letters:
                return letter

    # =========================================================
    # 5. LETTERA FINALE ESPLICITA
    #
    # Therefore: C
    # I choose C
    #
    # Accettiamo soltanto se la lettera compare alla FINE
    # dell'output.
    # =========================================================

    match = re.search(
        r"(?:CHOOSE|SELECT|THEREFORE|THUS)"
        r"\s*:?\s*\**\(?([A-J])\)?\.?\s*$",
        upper_text,
    )

    if match:

        letter = match.group(1)

        if letter in valid_letters:
            return letter

    # =========================================================
    # Nessun pattern sufficientemente affidabile
    # =========================================================

    return None