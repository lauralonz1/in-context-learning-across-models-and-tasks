import hashlib
import random


def make_target_seed(base_seed, target_id):
    """
    Crea un seed deterministico specifico per il target.

    NON usiamo hash() di Python perché può cambiare
    tra processi diversi.

    Stesso:
        base_seed + target_id
    -> stesso seed derivato.
    """

    text = f"{base_seed}:{target_id}"

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def remove_same_article_duplicates(examples):
    """
    Per QuALITY mantiene al massimo una domanda
    per ciascun articolo.

    Questo evita che un prompt few-shot contenga,
    ad esempio, tre domande tratte dallo stesso
    identico articolo.
    """

    seen_articles = set()
    result = []

    for example in examples:

        article_id = example.get("article_id")

        if article_id in seen_articles:
            continue

        seen_articles.add(article_id)
        result.append(example)

    return result


def select_random_demos(
    demo_pool,
    target,
    n_shots,
    seed=42,
):
    """
    Seleziona demonstrations casuali ma riproducibili.

    Proprietà:
    - target diversi -> selezioni diverse
    - stesso target + stesso seed -> stessa selezione
    - 1-shot è prefisso del 3-shot
    - 3-shot è prefisso del 5-shot
    - QuALITY: massimo una demo per articolo
    """

    if n_shots == 0:
        return []

    if n_shots < 0:
        raise ValueError(
            "n_shots deve essere >= 0"
        )

    eligible = []

    for example in demo_pool:

        # Mai usare il target stesso come demonstration.
        if example["item_id"] == target["item_id"]:
            continue

        # QuALITY:
        # mai usare lo stesso articolo del target.
        if (
            target["benchmark"] == "quality"
            and example.get("article_id")
            == target.get("article_id")
        ):
            continue

        eligible.append(example)

    # ---------------------------------------------------------
    # RNG specifico per target
    # ---------------------------------------------------------

    target_seed = make_target_seed(
        base_seed=seed,
        target_id=target["item_id"],
    )

    rng = random.Random(target_seed)

    # Creiamo UN ordine random deterministico.
    # Così shot maggiori aggiungono demo
    # senza cambiare quelle precedenti.
    shuffled = eligible.copy()

    rng.shuffle(shuffled)

    # ---------------------------------------------------------
    # QuALITY: una demo per articolo
    # ---------------------------------------------------------

    if target["benchmark"] == "quality":

        shuffled = remove_same_article_duplicates(
            shuffled
        )

    if len(shuffled) < n_shots:
        raise ValueError(
            f"Demo insufficienti: richieste {n_shots}, "
            f"disponibili {len(shuffled)}"
        )

    return shuffled[:n_shots]