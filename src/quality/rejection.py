"""
Automated rejection and quality evaluation for synthetic reviews.

Combines diversity, bias, and realism metrics into a unified
quality evaluation with configurable thresholds.
"""

import logging
from pathlib import Path
from typing import Optional
import yaml

from .diversity import DiversityAnalyzer
from .bias import BiasDetector
from .realism import RealismValidator

logger = logging.getLogger(__name__)


class QualityMetrics:
    """Container for quality evaluation results."""

    def __init__(
        self,
        diversity_score: float,
        realism_score: float,
        bias_score: float,
        overall_score: float,
        accepted: bool,
        rejection_reasons: list
    ):
        self.diversity_score = diversity_score
        self.realism_score = realism_score
        self.bias_score = bias_score
        self.overall_score = overall_score
        self.accepted = accepted
        self.rejection_reasons = rejection_reasons

    def to_dict(self) -> dict:
        return {
            "diversity_score": self.diversity_score,
            "realism_score": self.realism_score,
            "bias_score": self.bias_score,
            "overall_score": self.overall_score,
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons
        }


class QualityEvaluator:
    """
    Unified quality evaluator for synthetic reviews.

    Combines diversity, bias, and realism checks into a single
    evaluation with configurable weights and thresholds.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the quality evaluator.

        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self.config = self._load_config()

        # Initialize component analyzers
        self.diversity_analyzer = DiversityAnalyzer(self.config)
        self.bias_detector = BiasDetector(self.config)
        self.realism_validator = RealismValidator(self.config)

        # Load scoring configuration
        scoring = self.config.get("scoring", {})
        self.diversity_weight = scoring.get("diversity_weight", 0.30)
        self.realism_weight = scoring.get("realism_weight", 0.35)
        self.bias_weight = scoring.get("bias_weight", 0.35)
        self.pass_threshold = scoring.get("pass_threshold", 0.65)

        # Statistics tracking
        self._evaluation_count = 0
        self._acceptance_count = 0
        self._score_history = []

    def _load_config(self) -> dict:
        """Load quality configuration from YAML."""
        config_path = self.config_dir / "quality.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def evaluate(
        self,
        text: str,
        existing_texts: list[str],
        rating: int
    ) -> QualityMetrics:
        """
        Evaluate the quality of a synthetic review.

        Args:
            text: The review text to evaluate
            existing_texts: List of existing reviews for comparison
            rating: The star rating (1-5) of the review

        Returns:
            QualityMetrics object with scores and acceptance status
        """
        all_issues = []

        # 1. Diversity evaluation
        diversity_score, diversity_issues = self.diversity_analyzer.evaluate(
            text, existing_texts
        )
        all_issues.extend([f"[Diversity] {issue}" for issue in diversity_issues])

        # 2. Bias evaluation
        bias_score, bias_issues = self.bias_detector.evaluate(
            text, rating, update_corpus=True
        )
        all_issues.extend([f"[Bias] {issue}" for issue in bias_issues])

        # 3. Realism evaluation
        realism_score, realism_issues = self.realism_validator.evaluate(
            text, rating
        )
        all_issues.extend([f"[Realism] {issue}" for issue in realism_issues])

        # Calculate weighted overall score
        overall_score = (
            diversity_score * self.diversity_weight +
            realism_score * self.realism_weight +
            bias_score * self.bias_weight
        )

        # Determine acceptance
        accepted = overall_score >= self.pass_threshold and len(all_issues) <= 3

        # Additional rejection rules (hard failures)
        if diversity_score < 0.3:
            accepted = False
            if "[Diversity] Critical: score below minimum" not in all_issues:
                all_issues.append("[Diversity] Critical: score below minimum threshold")

        if realism_score < 0.3:
            accepted = False
            if "[Realism] Critical: score below minimum" not in all_issues:
                all_issues.append("[Realism] Critical: score below minimum threshold")

        # Update statistics
        self._evaluation_count += 1
        if accepted:
            self._acceptance_count += 1
        self._score_history.append(overall_score)

        return QualityMetrics(
            diversity_score=round(diversity_score, 3),
            realism_score=round(realism_score, 3),
            bias_score=round(bias_score, 3),
            overall_score=round(overall_score, 3),
            accepted=accepted,
            rejection_reasons=all_issues if not accepted else []
        )

    def evaluate_batch(
        self,
        reviews: list[dict],
        existing_texts: Optional[list[str]] = None
    ) -> list[QualityMetrics]:
        """
        Evaluate a batch of reviews.

        Args:
            reviews: List of dicts with 'text' and 'rating' keys
            existing_texts: Optional list of existing review texts

        Returns:
            List of QualityMetrics for each review
        """
        if existing_texts is None:
            existing_texts = []

        results = []
        current_texts = list(existing_texts)

        for review in reviews:
            metrics = self.evaluate(
                review["text"],
                current_texts,
                review["rating"]
            )
            results.append(metrics)

            # Add accepted reviews to comparison corpus
            if metrics.accepted:
                current_texts.append(review["text"])

        return results

    def get_statistics(self) -> dict:
        """Get evaluation statistics."""
        if not self._score_history:
            return {
                "total_evaluations": 0,
                "acceptance_rate": 0.0,
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }

        return {
            "total_evaluations": self._evaluation_count,
            "acceptance_rate": self._acceptance_count / self._evaluation_count,
            "average_score": sum(self._score_history) / len(self._score_history),
            "min_score": min(self._score_history),
            "max_score": max(self._score_history),
            "corpus_stats": self.bias_detector.get_corpus_stats()
        }

    def get_detailed_analysis(self, text: str, rating: int) -> dict:
        """
        Get detailed analysis of a single review.

        Returns comprehensive breakdown of all quality metrics.
        """
        # Get individual analyses
        diversity_score, diversity_issues = self.diversity_analyzer.evaluate(text, [])
        bias_score, bias_issues = self.bias_detector.evaluate(text, rating, update_corpus=False)
        realism_report = self.realism_validator.get_analysis_report(text)
        realism_score, realism_issues = self.realism_validator.evaluate(text, rating)

        return {
            "text": text,
            "rating": rating,
            "scores": {
                "diversity": diversity_score,
                "bias": bias_score,
                "realism": realism_score,
                "overall": (
                    diversity_score * self.diversity_weight +
                    realism_score * self.realism_weight +
                    bias_score * self.bias_weight
                )
            },
            "issues": {
                "diversity": diversity_issues,
                "bias": bias_issues,
                "realism": realism_issues
            },
            "realism_details": realism_report,
            "word_count": len(text.split()),
            "thresholds": {
                "pass_threshold": self.pass_threshold,
                "weights": {
                    "diversity": self.diversity_weight,
                    "realism": self.realism_weight,
                    "bias": self.bias_weight
                }
            }
        }

    def reset(self):
        """Reset all statistics and cached data."""
        self._evaluation_count = 0
        self._acceptance_count = 0
        self._score_history.clear()
        self.diversity_analyzer.clear_cache()
        self.bias_detector.reset_corpus_stats()

    def update_thresholds(
        self,
        pass_threshold: Optional[float] = None,
        diversity_weight: Optional[float] = None,
        realism_weight: Optional[float] = None,
        bias_weight: Optional[float] = None
    ):
        """Update evaluation thresholds dynamically."""
        if pass_threshold is not None:
            self.pass_threshold = pass_threshold
        if diversity_weight is not None:
            self.diversity_weight = diversity_weight
        if realism_weight is not None:
            self.realism_weight = realism_weight
        if bias_weight is not None:
            self.bias_weight = bias_weight

        # Normalize weights if they don't sum to 1
        total_weight = self.diversity_weight + self.realism_weight + self.bias_weight
        if abs(total_weight - 1.0) > 0.01:
            self.diversity_weight /= total_weight
            self.realism_weight /= total_weight
            self.bias_weight /= total_weight
            logger.info(f"Normalized weights to sum to 1.0")
