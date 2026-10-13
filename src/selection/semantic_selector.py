from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSelector:

    def __init__(
        self,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Carica il modello di embedding una sola volta.
        """

        self.model = SentenceTransformer(
            embedding_model
        )

        self.embedding_model = embedding_model

    def build_text(self, example):
        """
        Testo usato per calcolare la similarità semantica.

        MMLU-Pro:
        question

        QuALITY:
        question

        NOTA:
        non includiamo gold answer o label.
        """

        return example["question"].strip()

    def select_semantic_demos(
        self,
        demo_pool,
        target,
        n_shots,
    ):
        """
        Seleziona le n demonstrations semanticamente
        più simili al target.

        Regole:
        - nessun gold/label usato
        - target escluso
        - QuALITY: massimo una demo per articolo
        - selezione deterministica
        """

        if n_shots == 0:
            return []

        if n_shots < 0:
            raise ValueError(
                "n_shots deve essere >= 0"
            )

        eligible = []

        for example in demo_pool:

            if example["item_id"] == target["item_id"]:
                continue

            if (
                target["benchmark"] == "quality"
                and example.get("article_id")
                == target.get("article_id")
            ):
                continue

            eligible.append(example)

        if len(eligible) < n_shots:
            raise ValueError(
                f"Demo insufficienti: richieste {n_shots}, "
                f"disponibili {len(eligible)}"
            )

        target_text = self.build_text(target)

        demo_texts = [
            self.build_text(example)
            for example in eligible
        ]

        target_embedding = self.model.encode(
            target_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        demo_embeddings = self.model.encode(
            demo_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarities = cos_sim(
            target_embedding,
            demo_embeddings,
        )[0]

        # Ordine decrescente per similarità.
        ranked_indices = similarities.argsort(
            descending=True
        ).tolist()

        selected = []
        seen_articles = set()

        for index in ranked_indices:

            example = eligible[index]

            # QuALITY:
            # massimo una demonstration per articolo.
            if target["benchmark"] == "quality":

                article_id = example["article_id"]

                if article_id in seen_articles:
                    continue

                seen_articles.add(article_id)

            selected.append(example)

            if len(selected) == n_shots:
                break

        if len(selected) < n_shots:
            raise ValueError(
                f"Impossibile selezionare {n_shots} "
                f"demonstrations valide."
            )

        return selected