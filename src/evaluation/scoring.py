from src.evaluation.parser import parse_answer


def score_prediction(
    raw_output,
    gold_answer,
    valid_letters=None,
):
    """
    Valuta una singola risposta del modello.

    Restituisce un dizionario con:
    - raw_output
    - prediction
    - gold_answer
    - correct
    - parse_failure
    """

    prediction = parse_answer(
        raw_output,
        valid_letters=valid_letters,
    )

    parse_failure = prediction is None

    correct = (
        prediction == gold_answer
        if prediction is not None
        else False
    )

    return {
        "raw_output": raw_output,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "correct": correct,
        "parse_failure": parse_failure,
    }