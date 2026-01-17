"""
Quality guardrails for synthetic review generation.

This package provides metrics and evaluation for:
- Diversity: Lexical and semantic variety
- Bias: Sentiment balance and pattern detection
- Realism: Domain authenticity and specificity
- Rejection: Automated quality control with regeneration
"""

from .diversity import DiversityAnalyzer
from .bias import BiasDetector
from .realism import RealismValidator
from .rejection import QualityEvaluator

__all__ = ["DiversityAnalyzer", "BiasDetector", "RealismValidator", "QualityEvaluator"]
