"""
Domain realism validation for synthetic review quality assessment.

Ensures generated reviews contain authentic domain-specific content
and avoid generic or marketing-style language.
"""

import re
from collections import Counter
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RealismValidator:
    """Validates domain authenticity and specificity of reviews."""

    def __init__(self, config: dict):
        """
        Initialize realism validator.

        Args:
            config: Quality configuration dictionary with realism thresholds and domain keywords
        """
        self.realism_config = config.get("realism", {})
        self.min_domain_term_ratio = self.realism_config.get("min_domain_term_ratio", 0.02)
        self.max_marketing_term_ratio = self.realism_config.get("max_marketing_term_ratio", 0.05)
        self.min_specificity_score = self.realism_config.get("min_specificity_score", 0.4)
        self.require_concrete_details = self.realism_config.get("require_concrete_details", True)

        # Load domain keywords
        keywords = config.get("domain_keywords", {})
        self.technical_terms = set(t.lower() for t in keywords.get("technical_terms", []))
        self.feature_terms = set(t.lower() for t in keywords.get("feature_terms", []))
        self.complaint_terms = set(t.lower() for t in keywords.get("complaint_terms", []))
        self.marketing_red_flags = set(t.lower() for t in keywords.get("marketing_red_flags", []))

        # Combine all domain terms
        self.all_domain_terms = self.technical_terms | self.feature_terms | self.complaint_terms

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', text.lower())

    def find_domain_terms(self, text: str) -> dict:
        """
        Find domain-specific terms in text.

        Returns dict with categorized terms found.
        """
        text_lower = text.lower()
        words = self.tokenize(text)
        word_set = set(words)

        found = {
            "technical": [],
            "feature": [],
            "complaint": [],
            "marketing": []
        }

        # Check for exact matches and phrase matches
        for term in self.technical_terms:
            if term in text_lower or term in word_set:
                found["technical"].append(term)

        for term in self.feature_terms:
            if term in text_lower or term in word_set:
                found["feature"].append(term)

        for term in self.complaint_terms:
            if term in text_lower or term in word_set:
                found["complaint"].append(term)

        for term in self.marketing_red_flags:
            if term in text_lower:
                found["marketing"].append(term)

        return found

    def calculate_domain_term_ratio(self, text: str) -> float:
        """Calculate ratio of domain terms to total words."""
        words = self.tokenize(text)
        if not words:
            return 0.0

        found = self.find_domain_terms(text)
        domain_count = len(found["technical"]) + len(found["feature"]) + len(found["complaint"])

        return domain_count / len(words)

    def calculate_marketing_ratio(self, text: str) -> float:
        """Calculate ratio of marketing red flags to total words."""
        words = self.tokenize(text)
        if not words:
            return 0.0

        found = self.find_domain_terms(text)
        return len(found["marketing"]) / len(words)

    def detect_concrete_details(self, text: str) -> dict:
        """
        Detect concrete details in the review.

        Looks for:
        - Specific numbers/metrics
        - Version references
        - Time references
        - Feature names
        - Error messages or specific behaviors
        """
        details = {
            "numbers": [],
            "versions": [],
            "time_refs": [],
            "specific_features": [],
            "behaviors": []
        }

        # Numbers and metrics
        number_patterns = [
            r'\b\d+(?:\.\d+)?(?:\s*%|\s*ms|\s*seconds?|\s*minutes?|\s*hours?|\s*days?|\s*x\b)',
            r'\b\d+(?:\.\d+)?\s*(?:MB|GB|TB|KB)',
            r'\b\d+\s*(?:users?|requests?|calls?|builds?|deployments?)',
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["numbers"].extend(matches)

        # Version references
        version_pattern = r'v?\d+\.\d+(?:\.\d+)?|version\s+\d+'
        details["versions"] = re.findall(version_pattern, text, re.IGNORECASE)

        # Time references
        time_patterns = [
            r'(?:for|over|about|around)\s+(?:\d+\s+)?(?:weeks?|months?|years?)',
            r'(?:since|from)\s+(?:last\s+)?(?:week|month|year)',
            r'(?:recently|lately|last\s+time)',
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["time_refs"].extend(matches)

        # Specific behaviors (error messages, actions)
        behavior_patterns = [
            r'(?:when|if|after|before)\s+(?:I|we|you)\s+\w+',
            r'(?:throws?|returns?|shows?|displays?)\s+(?:an?\s+)?(?:error|warning|message)',
            r'(?:fails?|crashes?|freezes?|hangs?)\s+(?:when|if|on)',
        ]
        for pattern in behavior_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["behaviors"].extend(matches)

        return details

    def calculate_specificity_score(self, text: str) -> float:
        """
        Calculate how specific/concrete the review is.

        Higher score means more specific details.
        """
        details = self.detect_concrete_details(text)
        found_terms = self.find_domain_terms(text)

        # Count different types of specificity indicators
        scores = []

        # Domain terms (up to 0.3)
        domain_count = len(found_terms["technical"]) + len(found_terms["feature"])
        scores.append(min(0.3, domain_count * 0.1))

        # Concrete numbers/metrics (up to 0.3)
        numbers_count = len(details["numbers"]) + len(details["versions"])
        scores.append(min(0.3, numbers_count * 0.15))

        # Behavioral descriptions (up to 0.2)
        behavior_count = len(details["behaviors"])
        scores.append(min(0.2, behavior_count * 0.1))

        # Time references (up to 0.1)
        time_count = len(details["time_refs"])
        scores.append(min(0.1, time_count * 0.1))

        # Length bonus (up to 0.1) - longer reviews tend to be more detailed
        word_count = len(self.tokenize(text))
        if word_count >= 60:
            scores.append(0.1)
        elif word_count >= 40:
            scores.append(0.05)
        else:
            scores.append(0.0)

        return sum(scores)

    def detect_generic_patterns(self, text: str) -> list[str]:
        """Detect generic patterns that indicate low-quality generation."""
        generic_patterns = [
            r"(?:very|really|extremely)\s+(?:good|bad|great|nice|useful)",
            r"(?:I|we)\s+(?:really\s+)?(?:like|love|hate)\s+(?:this|it)",
            r"(?:would|highly)\s+recommend",
            r"(?:overall|in general|all in all)",
            r"(?:nice|good|great)\s+(?:product|tool|service|platform)",
            r"(?:does|works)\s+(?:well|great|fine)",
        ]

        found = []
        for pattern in generic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(pattern)

        return found

    def validate_rating_content_match(self, text: str, rating: int) -> tuple[bool, str]:
        """
        Validate that review content matches the rating.

        For negative ratings, expects complaint terms.
        For positive ratings, expects feature/praise terms.
        """
        found = self.find_domain_terms(text)
        text_lower = text.lower()

        if rating <= 2:
            # Should have some complaint terms or negative language
            negative_indicators = len(found["complaint"])
            negative_words = sum(1 for w in ["issue", "problem", "bug", "slow", "confusing", "frustrating"]
                                if w in text_lower)
            if negative_indicators == 0 and negative_words == 0:
                return False, f"Low rating ({rating}) but no complaint language found"
        elif rating >= 4:
            # Should have positive language but not be purely marketing
            positive_words = sum(1 for w in ["helpful", "easy", "fast", "reliable", "useful", "great"]
                               if w in text_lower)
            if len(found["marketing"]) > positive_words:
                return False, "Too much marketing language relative to genuine praise"

        return True, ""

    def evaluate(self, text: str, rating: int) -> tuple[float, list[str]]:
        """
        Evaluate domain realism of a review.

        Args:
            text: Review text to evaluate
            rating: Star rating (1-5)

        Returns:
            Tuple of (realism_score 0-1, list of issues)
        """
        issues = []
        scores = []

        # 1. Domain term ratio
        domain_ratio = self.calculate_domain_term_ratio(text)
        if domain_ratio < self.min_domain_term_ratio:
            issues.append(f"Low domain term ratio: {domain_ratio:.3f} < {self.min_domain_term_ratio}")
        scores.append(min(1.0, domain_ratio / self.min_domain_term_ratio) if self.min_domain_term_ratio > 0 else 1.0)

        # 2. Marketing language check
        marketing_ratio = self.calculate_marketing_ratio(text)
        if marketing_ratio > self.max_marketing_term_ratio:
            issues.append(f"Too much marketing language: {marketing_ratio:.3f} > {self.max_marketing_term_ratio}")
        scores.append(1.0 - min(1.0, marketing_ratio / self.max_marketing_term_ratio) if self.max_marketing_term_ratio > 0 else 1.0)

        # 3. Specificity score
        specificity = self.calculate_specificity_score(text)
        if specificity < self.min_specificity_score:
            issues.append(f"Low specificity: {specificity:.2f} < {self.min_specificity_score}")
        scores.append(min(1.0, specificity / self.min_specificity_score) if self.min_specificity_score > 0 else 1.0)

        # 4. Generic pattern detection
        generic_patterns = self.detect_generic_patterns(text)
        if len(generic_patterns) > 3:
            issues.append(f"Too many generic patterns: {len(generic_patterns)}")
            scores.append(0.5)
        else:
            scores.append(1.0 - len(generic_patterns) * 0.1)

        # 5. Rating-content match
        if self.require_concrete_details:
            is_match, match_issue = self.validate_rating_content_match(text, rating)
            if not is_match:
                issues.append(match_issue)
                scores.append(0.7)
            else:
                scores.append(1.0)

        # Calculate overall realism score
        realism_score = sum(scores) / len(scores) if scores else 0.0

        return realism_score, issues

    def get_analysis_report(self, text: str) -> dict:
        """Generate detailed analysis report for a review."""
        found_terms = self.find_domain_terms(text)
        details = self.detect_concrete_details(text)

        return {
            "domain_terms": found_terms,
            "concrete_details": details,
            "domain_term_ratio": self.calculate_domain_term_ratio(text),
            "marketing_ratio": self.calculate_marketing_ratio(text),
            "specificity_score": self.calculate_specificity_score(text),
            "generic_patterns": self.detect_generic_patterns(text),
            "word_count": len(self.tokenize(text))
        }
