from homoglyph import HomoglyphNormalization


class TextualPerturbation:
    def __init__(self, seed: int = 54):
        from textattack.augmentation import Augmenter
        from textattack.shared import AttackedText
        from textattack.shared.utils import set_seed
        from textattack.transformations.sentence_transformations import BackTranslation

        class FixSeedAugmenter(Augmenter):
            """Fixes the RNG seed before filtering so every augment() call is reproducible.

            Candidates are also sorted: CompositeTransformation collects them in a set, whose
            order changes per process (hash randomization), which would defeat the fixed seed.
            """

            def _filter_transformations(self, transformed_texts, current_text, original_text):
                set_seed(seed)
                transformed_texts = super()._filter_transformations(
                    transformed_texts, current_text, original_text)
                return sorted(transformed_texts, key=lambda t: t.text)

        class EnglishToSpanish(BackTranslation):
            """One-way translation en -> es (no translation back to English)."""

            def __init__(self):
                super().__init__(src_lang="en", target_lang="es",
                                 src_model="Helsinki-NLP/opus-mt-ROMANCE-en",
                                 target_model="Helsinki-NLP/opus-mt-en-ROMANCE",
                                 chained_back_translation=0)

            def _get_transformations(self, current_text, indices_to_modify):
                translated = self.translate([current_text.text], self.target_model,
                                            self.target_tokenizer, self.target_lang)
                return [AttackedText(translated[0])]

        self._FixSeedAugmenter = FixSeedAugmenter
        self._EnglishToSpanish = EnglishToSpanish
        self.replace_homoglyphs = HomoglyphNormalization()

    def __call__(self, caption: str) -> str:
        caption = self.replace_homoglyphs(caption)
        return self.morphing_augment(self.embedding_augment(caption)[0])[0]

    # Synonym swap. Defaults follow Table 5's VillanDiffusion (mignneko) setting;
    # max_mse_dist=None runs without constraints (the latte coffee setting).
    def embedding_augment(self, input_text: str, pct_words_to_swap: float = 1,
                          transformations_per_example: int = 10,
                          max_mse_dist: float | None = 0.05) -> list:
        from textattack.constraints.pre_transformation import RepeatModification
        from textattack.constraints.semantics import WordEmbeddingDistance
        from textattack.transformations import WordSwapEmbedding

        constraints = [] if max_mse_dist is None else [
            RepeatModification(), WordEmbeddingDistance(max_mse_dist=max_mse_dist)]
        augmenter = self._FixSeedAugmenter(
            transformation=WordSwapEmbedding(), constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example)
        return augmenter.augment(input_text)

    # Translate from English to Spanish
    def translation_augment(self, input_text: str, pct_words_to_swap: float = 0.5,
                            transformations_per_example: int = 10) -> list:
        import warnings
        from textattack.constraints.pre_transformation import RepeatModification

        constraints = [RepeatModification()]
        augmenter = self._FixSeedAugmenter(
            transformation=self._EnglishToSpanish(), constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example)
        with warnings.catch_warnings():
            # TextAttack's BackTranslation still calls the deprecated prepare_seq2seq_batch API.
            # The warning message starts with a newline, so match it with a dotall pattern.
            warnings.filterwarnings("ignore", message=r"(?s).*`prepare_seq2seq_batch` is deprecated",
                                    category=FutureWarning)
            return augmenter.augment(input_text)

    # Delete or replace characters ("Random Perturbation"). Defaults follow Table 5's
    # VillanDiffusion (mignneko) setting; Rickrolling passes pct_words_to_swap=0.5,
    # max_mse_dist=0.01, and max_mse_dist=None runs without constraints (latte coffee).
    def morphing_augment(self, input_text: str, pct_words_to_swap: float = 1,
                         transformations_per_example: int = 10,
                         max_mse_dist: float | None = 0.05) -> list:
        from textattack.constraints.pre_transformation import RepeatModification
        from textattack.constraints.semantics import WordEmbeddingDistance
        from textattack.transformations import (CompositeTransformation, WordSwapQWERTY,
                                                WordSwapRandomCharacterDeletion)

        transformation = CompositeTransformation([WordSwapRandomCharacterDeletion(),
                                                  WordSwapQWERTY()])
        constraints = [] if max_mse_dist is None else [
            RepeatModification(), WordEmbeddingDistance(max_mse_dist=max_mse_dist)]
        augmenter = self._FixSeedAugmenter(
            transformation=transformation, constraints=constraints,
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=transformations_per_example)
        return augmenter.augment(input_text)


if __name__ == "__main__":
    perturbation = TextualPerturbation()

    # Perturbation example
    s = "beａutiful car"
    s = perturbation.replace_homoglyphs(s)
    augmented_texts = perturbation.embedding_augment(s)
    augmented_texts = perturbation.morphing_augment(s) # use this for more aggressive attacks
    # augmented_texts = perturbation.translation_augment(s)
    print(augmented_texts)
