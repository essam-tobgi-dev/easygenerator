"""
Comparison analysis between synthetic and real review datasets.

Provides statistical analysis, distribution comparisons, and
quality metrics to validate synthetic data realism.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Optional imports for advanced analysis
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class DatasetAnalyzer:
    """Analyzes a single dataset of reviews."""

    def __init__(self, reviews: list[dict]):
        """
        Initialize analyzer with review data.

        Args:
            reviews: List of review dicts with 'review_text' and 'rating' or 'metadata.rating'
        """
        self.reviews = reviews
        self._sentiment_analyzer = None
        self._embedding_model = None

    def _get_sentiment_analyzer(self):
        """Get sentiment analyzer (lazy load)."""
        if self._sentiment_analyzer is None:
            if VADER_AVAILABLE:
                self._sentiment_analyzer = SentimentIntensityAnalyzer()
            else:
                self._sentiment_analyzer = SimpleSentimentAnalyzer()
        return self._sentiment_analyzer

    def _get_text(self, review: dict) -> str:
        """Extract review text from various formats."""
        if "review_text" in review:
            return review["review_text"]
        elif "text" in review:
            return review["text"]
        elif "content" in review:
            return review["content"]
        return ""

    def _get_rating(self, review: dict) -> Optional[int]:
        """Extract rating from various formats."""
        if "rating" in review:
            return review["rating"]
        elif "metadata" in review and "rating" in review["metadata"]:
            return review["metadata"]["rating"]
        elif "star_rating" in review:
            return review["star_rating"]
        return None

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return re.findall(r'\b[a-z]+\b', text.lower())

    def get_basic_stats(self) -> dict:
        """Calculate basic statistics about the dataset."""
        texts = [self._get_text(r) for r in self.reviews]
        ratings = [self._get_rating(r) for r in self.reviews]
        ratings = [r for r in ratings if r is not None]

        word_counts = [len(self.tokenize(t)) for t in texts]
        char_counts = [len(t) for t in texts]

        return {
            "total_reviews": len(self.reviews),
            "word_count": {
                "mean": statistics.mean(word_counts) if word_counts else 0,
                "median": statistics.median(word_counts) if word_counts else 0,
                "stdev": statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0
            },
            "char_count": {
                "mean": statistics.mean(char_counts) if char_counts else 0,
                "median": statistics.median(char_counts) if char_counts else 0,
                "stdev": statistics.stdev(char_counts) if len(char_counts) > 1 else 0
            },
            "rating_distribution": dict(Counter(ratings)),
            "average_rating": statistics.mean(ratings) if ratings else 0
        }

    def get_vocabulary_stats(self) -> dict:
        """Analyze vocabulary usage across the dataset."""
        all_words = []
        unique_per_review = []

        for review in self.reviews:
            words = self.tokenize(self._get_text(review))
            all_words.extend(words)
            unique_per_review.append(len(set(words)) / len(words) if words else 0)

        word_counts = Counter(all_words)
        total_words = len(all_words)
        unique_words = len(word_counts)

        # Calculate vocabulary richness metrics
        hapax_legomena = sum(1 for w, c in word_counts.items() if c == 1)
        type_token_ratio = unique_words / total_words if total_words > 0 else 0

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": type_token_ratio,
            "hapax_legomena": hapax_legomena,
            "hapax_ratio": hapax_legomena / unique_words if unique_words > 0 else 0,
            "avg_unique_per_review": statistics.mean(unique_per_review) if unique_per_review else 0,
            "top_words": word_counts.most_common(20)
        }

    def get_sentiment_distribution(self) -> dict:
        """Analyze sentiment distribution across reviews."""
        analyzer = self._get_sentiment_analyzer()
        sentiments = []

        for review in self.reviews:
            text = self._get_text(review)
            scores = analyzer.polarity_scores(text)
            sentiments.append({
                "compound": scores["compound"],
                "rating": self._get_rating(review)
            })

        compounds = [s["compound"] for s in sentiments]

        # Categorize sentiments
        positive = sum(1 for c in compounds if c > 0.05)
        negative = sum(1 for c in compounds if c < -0.05)
        neutral = len(compounds) - positive - negative

        # Calculate rating-sentiment correlation
        rating_sentiment_pairs = [
            (s["rating"], s["compound"])
            for s in sentiments
            if s["rating"] is not None
        ]

        correlation = 0.0
        if len(rating_sentiment_pairs) > 1:
            ratings = [p[0] for p in rating_sentiment_pairs]
            sents = [p[1] for p in rating_sentiment_pairs]
            # Simple Pearson correlation
            try:
                n = len(ratings)
                sum_xy = sum(r * s for r, s in rating_sentiment_pairs)
                sum_x = sum(ratings)
                sum_y = sum(sents)
                sum_x2 = sum(r ** 2 for r in ratings)
                sum_y2 = sum(s ** 2 for s in sents)

                numerator = n * sum_xy - sum_x * sum_y
                denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

                correlation = numerator / denominator if denominator > 0 else 0
            except Exception:
                correlation = 0.0

        return {
            "positive_ratio": positive / len(compounds) if compounds else 0,
            "negative_ratio": negative / len(compounds) if compounds else 0,
            "neutral_ratio": neutral / len(compounds) if compounds else 0,
            "average_compound": statistics.mean(compounds) if compounds else 0,
            "compound_stdev": statistics.stdev(compounds) if len(compounds) > 1 else 0,
            "rating_sentiment_correlation": correlation
        }

    def get_embedding_stats(self) -> dict:
        """Calculate embedding-based statistics."""
        if not EMBEDDINGS_AVAILABLE:
            return {"available": False, "reason": "sentence-transformers not installed"}

        try:
            if self._embedding_model is None:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            texts = [self._get_text(r) for r in self.reviews]
            embeddings = self._embedding_model.encode(texts)

            # Calculate centroid
            centroid = np.mean(embeddings, axis=0)

            # Calculate distances from centroid
            distances = [np.linalg.norm(e - centroid) for e in embeddings]

            # Calculate pairwise similarities (sample if too large)
            n = len(embeddings)
            if n > 100:
                # Sample indices for pairwise comparison
                sample_size = 100
                indices = np.random.choice(n, sample_size, replace=False)
                sample_embeddings = embeddings[indices]
            else:
                sample_embeddings = embeddings

            similarities = []
            for i in range(len(sample_embeddings)):
                for j in range(i + 1, len(sample_embeddings)):
                    sim = np.dot(sample_embeddings[i], sample_embeddings[j]) / (
                        np.linalg.norm(sample_embeddings[i]) * np.linalg.norm(sample_embeddings[j])
                    )
                    similarities.append(sim)

            return {
                "available": True,
                "embedding_dimension": len(centroid),
                "avg_distance_from_centroid": statistics.mean(distances),
                "distance_stdev": statistics.stdev(distances) if len(distances) > 1 else 0,
                "avg_pairwise_similarity": statistics.mean(similarities) if similarities else 0,
                "similarity_stdev": statistics.stdev(similarities) if len(similarities) > 1 else 0
            }

        except Exception as e:
            return {"available": False, "reason": str(e)}


class SimpleSentimentAnalyzer:
    """Simple fallback sentiment analyzer."""

    POSITIVE_WORDS = {'great', 'excellent', 'amazing', 'good', 'love', 'best', 'awesome', 'perfect', 'easy', 'helpful'}
    NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'poor', 'worst', 'hate', 'slow', 'buggy', 'broken', 'frustrating'}

    def polarity_scores(self, text: str) -> dict:
        words = set(text.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        compound = (pos - neg) / max(pos + neg, 1)
        return {"compound": compound, "pos": pos, "neg": neg}


class DatasetComparator:
    """Compares synthetic and real review datasets."""

    def __init__(
        self,
        real_reviews_path: str = "data/real_reviews/real_reviews.json",
        synthetic_reviews_path: str = "data/synthetic/synthetic_reviews.jsonl"
    ):
        """
        Initialize comparator with dataset paths.

        Args:
            real_reviews_path: Path to real reviews JSON file
            synthetic_reviews_path: Path to synthetic reviews JSONL file
        """
        self.real_path = Path(real_reviews_path)
        self.synthetic_path = Path(synthetic_reviews_path)
        self._real_reviews = None
        self._synthetic_reviews = None

    def load_real_reviews(self) -> list[dict]:
        """Load real reviews from JSON file."""
        if self._real_reviews is None:
            if self.real_path.exists():
                with open(self.real_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Handle both list and dict with 'reviews' key
                    if isinstance(data, list):
                        self._real_reviews = data
                    else:
                        self._real_reviews = data.get("reviews", [])
            else:
                self._real_reviews = []
                logger.warning(f"Real reviews file not found: {self.real_path}")
        return self._real_reviews

    def load_synthetic_reviews(self) -> list[dict]:
        """Load synthetic reviews from JSONL file."""
        if self._synthetic_reviews is None:
            self._synthetic_reviews = []
            if self.synthetic_path.exists():
                with open(self.synthetic_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self._synthetic_reviews.append(json.loads(line))
            else:
                logger.warning(f"Synthetic reviews file not found: {self.synthetic_path}")
        return self._synthetic_reviews

    def compare(self) -> dict:
        """
        Perform comprehensive comparison of datasets.

        Returns detailed comparison metrics.
        """
        real_reviews = self.load_real_reviews()
        synthetic_reviews = self.load_synthetic_reviews()

        real_analyzer = DatasetAnalyzer(real_reviews)
        synthetic_analyzer = DatasetAnalyzer(synthetic_reviews)

        # Get all statistics
        real_stats = {
            "basic": real_analyzer.get_basic_stats(),
            "vocabulary": real_analyzer.get_vocabulary_stats(),
            "sentiment": real_analyzer.get_sentiment_distribution(),
            "embeddings": real_analyzer.get_embedding_stats()
        }

        synthetic_stats = {
            "basic": synthetic_analyzer.get_basic_stats(),
            "vocabulary": synthetic_analyzer.get_vocabulary_stats(),
            "sentiment": synthetic_analyzer.get_sentiment_distribution(),
            "embeddings": synthetic_analyzer.get_embedding_stats()
        }

        # Calculate differences
        differences = self._calculate_differences(real_stats, synthetic_stats)

        # Generate summary scores
        quality_scores = self._calculate_quality_scores(real_stats, synthetic_stats)

        return {
            "real": real_stats,
            "synthetic": synthetic_stats,
            "differences": differences,
            "quality_scores": quality_scores,
            "summary": self._generate_summary(real_stats, synthetic_stats, quality_scores)
        }

    def _calculate_differences(self, real: dict, synthetic: dict) -> dict:
        """Calculate key differences between datasets."""
        differences = {}

        # Basic stats differences
        if real["basic"]["word_count"]["mean"] > 0:
            differences["word_count_diff"] = (
                synthetic["basic"]["word_count"]["mean"] - real["basic"]["word_count"]["mean"]
            ) / real["basic"]["word_count"]["mean"] * 100

        if real["basic"]["average_rating"] > 0:
            differences["avg_rating_diff"] = (
                synthetic["basic"]["average_rating"] - real["basic"]["average_rating"]
            )

        # Vocabulary differences
        differences["type_token_ratio_diff"] = (
            synthetic["vocabulary"]["type_token_ratio"] - real["vocabulary"]["type_token_ratio"]
        )

        # Sentiment differences
        differences["positive_ratio_diff"] = (
            synthetic["sentiment"]["positive_ratio"] - real["sentiment"]["positive_ratio"]
        )
        differences["negative_ratio_diff"] = (
            synthetic["sentiment"]["negative_ratio"] - real["sentiment"]["negative_ratio"]
        )

        # Embedding differences (if available)
        if real["embeddings"].get("available") and synthetic["embeddings"].get("available"):
            differences["embedding_spread_diff"] = (
                synthetic["embeddings"]["distance_stdev"] - real["embeddings"]["distance_stdev"]
            )

        return differences

    def _calculate_quality_scores(self, real: dict, synthetic: dict) -> dict:
        """Calculate quality scores comparing synthetic to real."""
        scores = {}

        # Length similarity (0-1, 1 is perfect match)
        if real["basic"]["word_count"]["mean"] > 0:
            length_ratio = synthetic["basic"]["word_count"]["mean"] / real["basic"]["word_count"]["mean"]
            scores["length_similarity"] = 1.0 - min(1.0, abs(1.0 - length_ratio))
        else:
            scores["length_similarity"] = 0.5

        # Vocabulary richness similarity
        ttr_diff = abs(synthetic["vocabulary"]["type_token_ratio"] - real["vocabulary"]["type_token_ratio"])
        scores["vocabulary_similarity"] = 1.0 - min(1.0, ttr_diff * 5)

        # Sentiment distribution similarity
        pos_diff = abs(synthetic["sentiment"]["positive_ratio"] - real["sentiment"]["positive_ratio"])
        neg_diff = abs(synthetic["sentiment"]["negative_ratio"] - real["sentiment"]["negative_ratio"])
        scores["sentiment_similarity"] = 1.0 - min(1.0, (pos_diff + neg_diff))

        # Rating distribution similarity (if both have ratings)
        real_ratings = real["basic"]["rating_distribution"]
        synth_ratings = synthetic["basic"]["rating_distribution"]
        if real_ratings and synth_ratings:
            all_ratings = set(real_ratings.keys()) | set(synth_ratings.keys())
            total_real = sum(real_ratings.values())
            total_synth = sum(synth_ratings.values())

            rating_diffs = []
            for r in all_ratings:
                real_pct = real_ratings.get(r, 0) / total_real if total_real > 0 else 0
                synth_pct = synth_ratings.get(r, 0) / total_synth if total_synth > 0 else 0
                rating_diffs.append(abs(real_pct - synth_pct))

            scores["rating_distribution_similarity"] = 1.0 - min(1.0, sum(rating_diffs) / 2)
        else:
            scores["rating_distribution_similarity"] = 0.5

        # Overall quality score
        scores["overall"] = statistics.mean([
            scores["length_similarity"],
            scores["vocabulary_similarity"],
            scores["sentiment_similarity"],
            scores["rating_distribution_similarity"]
        ])

        return scores

    def _generate_summary(self, real: dict, synthetic: dict, scores: dict) -> dict:
        """Generate a human-readable summary of the comparison."""
        summary = {
            "dataset_sizes": {
                "real": real["basic"]["total_reviews"],
                "synthetic": synthetic["basic"]["total_reviews"]
            },
            "overall_similarity": scores["overall"],
            "strengths": [],
            "weaknesses": []
        }

        # Identify strengths and weaknesses
        if scores["length_similarity"] >= 0.8:
            summary["strengths"].append("Review length closely matches real data")
        elif scores["length_similarity"] < 0.6:
            summary["weaknesses"].append("Review length differs significantly from real data")

        if scores["vocabulary_similarity"] >= 0.8:
            summary["strengths"].append("Vocabulary richness is consistent with real data")
        elif scores["vocabulary_similarity"] < 0.6:
            summary["weaknesses"].append("Vocabulary usage differs from real data patterns")

        if scores["sentiment_similarity"] >= 0.8:
            summary["strengths"].append("Sentiment distribution matches real data well")
        elif scores["sentiment_similarity"] < 0.6:
            summary["weaknesses"].append("Sentiment distribution is skewed compared to real data")

        if scores["rating_distribution_similarity"] >= 0.8:
            summary["strengths"].append("Rating distribution aligns with real data")
        elif scores["rating_distribution_similarity"] < 0.6:
            summary["weaknesses"].append("Rating distribution differs from real data")

        return summary

    def generate_comparison_table(self) -> str:
        """Generate a formatted comparison table."""
        comparison = self.compare()

        lines = [
            "=" * 70,
            "DATASET COMPARISON REPORT",
            "=" * 70,
            "",
            "BASIC STATISTICS",
            "-" * 40,
            f"{'Metric':<30} {'Real':>15} {'Synthetic':>15}",
            "-" * 40,
        ]

        real_basic = comparison["real"]["basic"]
        synth_basic = comparison["synthetic"]["basic"]

        lines.append(f"{'Total Reviews':<30} {real_basic['total_reviews']:>15} {synth_basic['total_reviews']:>15}")
        lines.append(f"{'Avg Word Count':<30} {real_basic['word_count']['mean']:>15.1f} {synth_basic['word_count']['mean']:>15.1f}")
        lines.append(f"{'Avg Rating':<30} {real_basic['average_rating']:>15.2f} {synth_basic['average_rating']:>15.2f}")

        lines.extend([
            "",
            "VOCABULARY METRICS",
            "-" * 40,
        ])

        real_vocab = comparison["real"]["vocabulary"]
        synth_vocab = comparison["synthetic"]["vocabulary"]

        lines.append(f"{'Unique Words':<30} {real_vocab['unique_words']:>15} {synth_vocab['unique_words']:>15}")
        lines.append(f"{'Type-Token Ratio':<30} {real_vocab['type_token_ratio']:>15.3f} {synth_vocab['type_token_ratio']:>15.3f}")

        lines.extend([
            "",
            "SENTIMENT DISTRIBUTION",
            "-" * 40,
        ])

        real_sent = comparison["real"]["sentiment"]
        synth_sent = comparison["synthetic"]["sentiment"]

        lines.append(f"{'Positive Ratio':<30} {real_sent['positive_ratio']:>15.2%} {synth_sent['positive_ratio']:>15.2%}")
        lines.append(f"{'Negative Ratio':<30} {real_sent['negative_ratio']:>15.2%} {synth_sent['negative_ratio']:>15.2%}")
        lines.append(f"{'Neutral Ratio':<30} {real_sent['neutral_ratio']:>15.2%} {synth_sent['neutral_ratio']:>15.2%}")

        lines.extend([
            "",
            "QUALITY SCORES",
            "-" * 40,
        ])

        for metric, score in comparison["quality_scores"].items():
            lines.append(f"{metric:<30} {score:>15.2%}")

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


def compare_datasets(
    real_path: str = "data/real_reviews/real_reviews.json",
    synthetic_path: str = "data/synthetic/synthetic_reviews.jsonl"
) -> dict:
    """
    High-level function to compare datasets.

    Args:
        real_path: Path to real reviews JSON
        synthetic_path: Path to synthetic reviews JSONL

    Returns:
        Comparison results dictionary
    """
    comparator = DatasetComparator(real_path, synthetic_path)
    return comparator.compare()
