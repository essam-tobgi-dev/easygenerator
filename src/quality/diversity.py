"""
Diversity metrics for synthetic review quality assessment.

Measures lexical diversity and semantic similarity to ensure
generated reviews are sufficiently varied.
"""

import re
from collections import Counter
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Optional imports for advanced features
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Semantic similarity disabled.")


class DiversityAnalyzer:
    """Analyzes lexical and semantic diversity of reviews."""

    def __init__(self, config: dict):
        """
        Initialize diversity analyzer.

        Args:
            config: Quality configuration dictionary with diversity thresholds
        """
        self.config = config.get("diversity", {})
        self.max_vocab_overlap = self.config.get("max_vocab_overlap", 0.35)
        self.max_semantic_similarity = self.config.get("max_semantic_similarity", 0.85)
        self.min_unique_words_ratio = self.config.get("min_unique_words_ratio", 0.3)
        self.ngram_overlap_threshold = self.config.get("ngram_overlap_threshold", 0.4)

        self._embedding_model = None
        self._embedding_cache = {}

    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None and EMBEDDINGS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        return self._embedding_model

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenization of text into words."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words

    def get_ngrams(self, words: list[str], n: int = 2) -> list[tuple]:
        """Extract n-grams from word list."""
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

    def calculate_unique_words_ratio(self, text: str) -> float:
        """Calculate ratio of unique words to total words."""
        words = self.tokenize(text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def calculate_vocab_overlap(self, text: str, existing_texts: list[str]) -> float:
        """
        Calculate vocabulary overlap with existing texts.

        Returns the maximum Jaccard similarity with any existing text.
        """
        if not existing_texts:
            return 0.0

        new_words = set(self.tokenize(text))
        if not new_words:
            return 0.0

        max_overlap = 0.0
        for existing in existing_texts:
            existing_words = set(self.tokenize(existing))
            if not existing_words:
                continue

            # Jaccard similarity
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            overlap = intersection / union if union > 0 else 0.0
            max_overlap = max(max_overlap, overlap)

        return max_overlap

    def calculate_ngram_overlap(self, text: str, existing_texts: list[str], n: int = 2) -> float:
        """
        Calculate n-gram overlap with existing texts.

        Detects repeated phrase patterns.
        """
        if not existing_texts:
            return 0.0

        words = self.tokenize(text)
        new_ngrams = set(self.get_ngrams(words, n))
        if not new_ngrams:
            return 0.0

        # Collect all n-grams from existing texts
        existing_ngrams = set()
        for existing in existing_texts:
            existing_words = self.tokenize(existing)
            existing_ngrams.update(self.get_ngrams(existing_words, n))

        if not existing_ngrams:
            return 0.0

        # Calculate overlap
        overlap = len(new_ngrams & existing_ngrams) / len(new_ngrams)
        return overlap

    def calculate_semantic_similarity(
        self,
        text: str,
        existing_texts: list[str],
        use_cache: bool = True
    ) -> float:
        """
        Calculate maximum semantic similarity to existing texts using embeddings.

        Returns 0.0 if embeddings are not available.
        """
        if not EMBEDDINGS_AVAILABLE or not existing_texts:
            return 0.0

        model = self._get_embedding_model()
        if model is None:
            return 0.0

        try:
            # Get embedding for new text
            new_embedding = model.encode([text])[0]

            # Get embeddings for existing texts (with caching)
            max_similarity = 0.0
            for existing in existing_texts:
                if use_cache and existing in self._embedding_cache:
                    existing_embedding = self._embedding_cache[existing]
                else:
                    existing_embedding = model.encode([existing])[0]
                    if use_cache:
                        self._embedding_cache[existing] = existing_embedding

                # Cosine similarity
                similarity = np.dot(new_embedding, existing_embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
                )
                max_similarity = max(max_similarity, similarity)

            return float(max_similarity)

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_global_vocab_diversity(self, all_texts: list[str]) -> float:
        """
        Calculate vocabulary diversity across all texts.

        Returns ratio of unique words to total words across corpus.
        """
        all_words = []
        for text in all_texts:
            all_words.extend(self.tokenize(text))

        if not all_words:
            return 0.0

        return len(set(all_words)) / len(all_words)

    def evaluate(
        self,
        text: str,
        existing_texts: list[str]
    ) -> tuple[float, list[str]]:
        """
        Evaluate diversity of a text against existing corpus.

        Args:
            text: The new text to evaluate
            existing_texts: List of existing texts to compare against

        Returns:
            Tuple of (diversity_score 0-1, list of issues)
        """
        issues = []
        scores = []

        # 1. Unique words ratio (internal diversity)
        unique_ratio = self.calculate_unique_words_ratio(text)
        if unique_ratio < self.min_unique_words_ratio:
            issues.append(f"Low unique word ratio: {unique_ratio:.2f} < {self.min_unique_words_ratio}")
        scores.append(min(1.0, unique_ratio / self.min_unique_words_ratio))

        # 2. Vocabulary overlap
        vocab_overlap = self.calculate_vocab_overlap(text, existing_texts)
        if vocab_overlap > self.max_vocab_overlap:
            issues.append(f"High vocab overlap: {vocab_overlap:.2f} > {self.max_vocab_overlap}")
        scores.append(1.0 - min(1.0, vocab_overlap / self.max_vocab_overlap) if self.max_vocab_overlap > 0 else 1.0)

        # 3. N-gram overlap (phrase repetition)
        ngram_overlap = self.calculate_ngram_overlap(text, existing_texts)
        if ngram_overlap > self.ngram_overlap_threshold:
            issues.append(f"High bigram overlap: {ngram_overlap:.2f} > {self.ngram_overlap_threshold}")
        scores.append(1.0 - min(1.0, ngram_overlap / self.ngram_overlap_threshold) if self.ngram_overlap_threshold > 0 else 1.0)

        # 4. Semantic similarity (if available)
        if EMBEDDINGS_AVAILABLE:
            semantic_sim = self.calculate_semantic_similarity(text, existing_texts)
            if semantic_sim > self.max_semantic_similarity:
                issues.append(f"High semantic similarity: {semantic_sim:.2f} > {self.max_semantic_similarity}")
            scores.append(1.0 - min(1.0, semantic_sim / self.max_semantic_similarity) if self.max_semantic_similarity > 0 else 1.0)

        # Calculate overall diversity score
        diversity_score = sum(scores) / len(scores) if scores else 0.0

        return diversity_score, issues

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
