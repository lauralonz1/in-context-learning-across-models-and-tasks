from src.selection.random_selector import select_random_demos
from src.selection.semantic_selector import SemanticSelector


class DemoSelector:

    def __init__(
        self,
        semantic_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Wrapper comune per tutte le strategie di selezione.
        """

        self.semantic_selector = SemanticSelector(
            embedding_model=semantic_embedding_model
        )

    def select(
        self,
        demo_pool,
        target,
        n_shots,
        strategy,
        seed=42,
    ):
        """
        strategy:
            - "random"
            - "semantic"
            - "none"

        Per zero-shot restituisce sempre [].
        """

        if n_shots == 0:
            return []

        if strategy == "none":
            raise ValueError(
                "strategy='none' è valida solo con n_shots=0"
            )

        if strategy == "random":
            return select_random_demos(
                demo_pool=demo_pool,
                target=target,
                n_shots=n_shots,
                seed=seed,
            )

        if strategy == "semantic":
            return self.semantic_selector.select_semantic_demos(
                demo_pool=demo_pool,
                target=target,
                n_shots=n_shots,
            )

        raise ValueError(
            f"Strategia sconosciuta: {strategy}"
        )