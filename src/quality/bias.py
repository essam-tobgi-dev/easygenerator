"""
Bias detection for synthetic review quality assessment.

Detects sentiment skew, pattern bias, and unrealistic correlations.
"""

import re
from collections import Counter
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Optional VADER for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available. Using simple rule-based sentiment.")


class SimpleSentimentAnalyzer:
    """Simple rule-based sentiment analyzer as fallback."""

    POSITIVE_WORDS = {
        'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'good',
        'love', 'best', 'awesome', 'perfect', 'easy', 'helpful', 'reliable',
        'fast', 'efficient', 'intuitive', 'powerful', 'impressive', 'solid',
        'smooth', 'clean', 'beautiful', 'outstanding', 'superb', 'brilliant'
    }

    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
        'slow', 'buggy', 'broken', 'confusing', 'frustrating', 'difficult',
        'complicated', 'unreliable', 'unstable', 'useless', 'disappointing',
        'annoying', 'clunky', 'painful', 'fails', 'crash', 'error', 'issue'
    }

    def polarity_scores(self, text: str) -> dict:
        """Calculate simple polarity scores."""
        words = text.lower().split()
        word_set = set(words)

        pos_count = len(word_set & self.POSITIVE_WORDS)
        neg_count = len(word_set & self.NEGATIVE_WORDS)
        total = len(words)

        if total == 0:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        pos_ratio = pos_count / total
        neg_ratio = neg_count / total
        neu_ratio = 1.0 - pos_ratio - neg_ratio

        # Compound score from -1 to 1
        compound = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        return {
            "compound": compound,
            "pos": pos_ratio,
            "neg": neg_ratio,
            "neu": max(0, neu_ratio)
        }


class BiasDetector:
    """Detects various forms of bias in generated reviews."""

    def __init__(self, config: dict):
        """
        Initialize bias detector.

        Args:
            config: Quality configuration dictionary with bias thresholds
        """
        self.config = config.get("bias", {})
        self.max_positive_ratio = self.config.get("max_positive_ratio", 0.55)
        self.max_negative_ratio = self.config.get("max_negative_ratio", 0.30)
        self.sentiment_rating_correlation_min = self.config.get("sentiment_rating_correlation_min", 0.5)
        self.max_phrase_repetition = self.config.get("max_phrase_repetition", 3)

        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self._sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self._sentiment_analyzer = SimpleSentimentAnalyzer()

        # Track corpus-level statistics
        self._sentiment_history = []
        self._phrase_counts = Counter()

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of text.

        Returns dict with compound, pos, neg, neu scores.
        """
        return self._sentiment_analyzer.polarity_scores(text)

    def get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound score to label."""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def check_rating_sentiment_alignment(self, text: str, rating: int) -> tuple[bool, float, str]:
        """
        Check if sentiment aligns with star rating.

        Returns:
            Tuple of (is_aligned, alignment_score, issue_description)
        """
        sentiment = self.analyze_sentiment(text)
        compound = sentiment["compound"]

        # Expected sentiment ranges for each rating
        expected_ranges = {
            5: (0.3, 1.0),   # Strongly positive
            4: (0.1, 0.8),   # Positive
            3: (-0.3, 0.3),  # Mixed/neutral
            2: (-0.6, 0.1),  # Negative
            1: (-1.0, -0.2)  # Strongly negative
        }

        min_expected, max_expected = expected_ranges.get(rating, (-1, 1))

        if min_expected <= compound <= max_expected:
            # Calculate how centered the sentiment is in expected range
            range_center = (min_expected + max_expected) / 2
            range_width = max_expected - min_expected
            deviation = abs(compound - range_center) / (range_width / 2) if range_width > 0 else 0
            alignment_score = 1.0 - min(1.0, deviation)
            return True, alignment_score, ""
        else:
            issue = f"Sentiment mismatch: {rating}-star review has sentiment {compound:.2f}, expected {min_expected:.1f} to {max_expected:.1f}"
            # Score based on how far outside the range
            if compound < min_expected:
                deviation = min_expected - compound
            else:
                deviation = compound - max_expected
            alignment_score = max(0, 1.0 - deviation)
            return False, alignment_score, issue

    def extract_phrases(self, text: str, min_length: int = 3, max_length: int = 6) -> list[str]:
        """Extract phrase patterns from text."""
        words = text.lower().split()
        phrases = []

        for length in range(min_length, min(max_length + 1, len(words) + 1)):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i + length])
                # Skip phrases that are mostly stop words
                content_words = [w for w in words[i:i + length] if len(w) > 3]
                if len(content_words) >= 2:
                    phrases.append(phrase)

        return phrases

    def detect_phrase_patterns(self, text: str) -> tuple[list[str], list[str]]:
        """
        Detect repeated phrase patterns.

        Returns:
            Tuple of (repeated_phrases, new_phrases)
        """
        phrases = self.extract_phrases(text)
        repeated = []
        new = []

        for phrase in phrases:
            if self._phrase_counts[phrase] >= self.max_phrase_repetition:
                repeated.append(phrase)
            else:
                new.append(phrase)

        return repeated, new

    def update_phrase_counts(self, text: str):
        """Update phrase frequency counts."""
        phrases = self.extract_phrases(text)
        self._phrase_counts.update(phrases)

    def calculate_corpus_sentiment_distribution(self) -> dict:
        """Calculate sentiment distribution across tracked reviews."""
        if not self._sentiment_history:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        labels = [self.get_sentiment_label(s) for s in self._sentiment_history]
        total = len(labels)

        return {
            "positive": labels.count("positive") / total,
            "neutral": labels.count("neutral") / total,
            "negative": labels.count("negative") / total
        }

    def check_corpus_balance(self) -> tuple[bool, dict, list[str]]:
        """
        Check if corpus sentiment is balanced.

        Returns:
            Tuple of (is_balanced, distribution, issues)
        """
        dist = self.calculate_corpus_sentiment_distribution()
        issues = []

        if dist["positive"] > self.max_positive_ratio:
            issues.append(f"Positive sentiment ratio {dist['positive']:.2f} exceeds {self.max_positive_ratio}")

        if dist["negative"] > self.max_negative_ratio:
            issues.append(f"Negative sentiment ratio {dist['negative']:.2f} exceeds {self.max_negative_ratio}")

        return len(issues) == 0, dist, issues

    def detect_template_patterns(self, text: str) -> list[str]:
        """Detect common template patterns that indicate low quality."""
        patterns = [
            r"^(I |We |Our team |My team )",  # Starting patterns
            r"(would recommend|highly recommend|definitely recommend).*\.$",  # Ending patterns
            r"(pros|cons|overall)[:\s]",  # Structure markers
            r"^\d+\s*(out of|/)\s*\d+",  # Rating formats
            r"(in conclusion|to summarize|all in all)",  # Summary phrases
        ]

        detected = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(pattern)

        return detected

    def evaluate(
        self,
        text: str,
        rating: int,
        update_corpus: bool = True
    ) -> tuple[float, list[str]]:
        """
        Evaluate bias in a single review.

        Args:
            text: Review text to evaluate
            rating: Star rating (1-5)
            update_corpus: Whether to update corpus-level statistics

        Returns:
            Tuple of (bias_score 0-1, list of issues)
        """
        issues = []
        scores = []

        # 1. Check rating-sentiment alignment
        is_aligned, alignment_score, alignment_issue = self.check_rating_sentiment_alignment(text, rating)
        if not is_aligned:
            issues.append(alignment_issue)
        scores.append(alignment_score)

        # 2. Detect phrase pattern repetition
        repeated_phrases, new_phrases = self.detect_phrase_patterns(text)
        if repeated_phrases:
            issues.append(f"Repeated phrases detected: {repeated_phrases[:3]}")
            scores.append(max(0, 1.0 - len(repeated_phrases) * 0.2))
        else:
            scores.append(1.0)

        # 3. Detect template patterns
        templates = self.detect_template_patterns(text)
        if len(templates) > 2:
            issues.append(f"Multiple template patterns detected")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Update corpus statistics
        if update_corpus:
            sentiment = self.analyze_sentiment(text)
            self._sentiment_history.append(sentiment["compound"])
            self.update_phrase_counts(text)

        # 4. Check corpus balance (only warn, don't penalize individual review)
        is_balanced, dist, balance_issues = self.check_corpus_balance()
        if not is_balanced and len(self._sentiment_history) > 10:
            # Only issue warning, don't affect score
            logger.warning(f"Corpus imbalance detected: {balance_issues}")

        # Calculate overall bias score (higher is better = less biased)
        bias_score = sum(scores) / len(scores) if scores else 1.0

        return bias_score, issues

    def reset_corpus_stats(self):
        """Reset corpus-level statistics."""
        self._sentiment_history.clear()
        self._phrase_counts.clear()

    def get_corpus_stats(self) -> dict:
        """Get current corpus statistics."""
        return {
            "total_reviews": len(self._sentiment_history),
            "sentiment_distribution": self.calculate_corpus_sentiment_distribution(),
            "top_phrases": self._phrase_counts.most_common(10),
            "average_sentiment": sum(self._sentiment_history) / len(self._sentiment_history) if self._sentiment_history else 0
        }
