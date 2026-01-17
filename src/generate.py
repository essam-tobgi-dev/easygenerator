"""
Core generation module for synthetic review data.

Handles LLM interactions, prompt construction, and review generation pipeline.
"""

import json
import random
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
import yaml

try:
    import openai
except ImportError:
    openai = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReviewMetadata:
    """Metadata for a generated review."""
    persona_name: str
    persona_experience: str
    persona_tone: str
    product_name: str
    product_category: str
    rating: int
    model_provider: str
    model_name: str
    generation_time_ms: float
    attempt_number: int
    timestamp: str


@dataclass
class QualityMetrics:
    """Quality metrics for a generated review."""
    diversity_score: float
    realism_score: float
    bias_score: float
    overall_score: float
    accepted: bool
    rejection_reasons: list


@dataclass
class SyntheticReview:
    """A complete synthetic review with all metadata."""
    id: str
    review_text: str
    metadata: ReviewMetadata
    quality_metrics: Optional[QualityMetrics] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "review_text": self.review_text,
            "metadata": asdict(self.metadata)
        }
        if self.quality_metrics:
            # Handle both dataclass and regular class with to_dict method
            if hasattr(self.quality_metrics, 'to_dict'):
                result["quality_metrics"] = self.quality_metrics.to_dict()
            else:
                result["quality_metrics"] = asdict(self.quality_metrics)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SyntheticReview":
        """Create from dictionary."""
        metadata = ReviewMetadata(**data["metadata"])
        quality_metrics = None
        if data.get("quality_metrics"):
            quality_metrics = QualityMetrics(**data["quality_metrics"])
        return cls(
            id=data["id"],
            review_text=data["review_text"],
            metadata=metadata,
            quality_metrics=quality_metrics
        )


class ConfigLoader:
    """Loads and manages YAML configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._personas = None
        self._generation = None
        self._quality = None

    def load_personas(self) -> dict:
        """Load personas configuration."""
        if self._personas is None:
            with open(self.config_dir / "personas.yaml", "r") as f:
                self._personas = yaml.safe_load(f)
        return self._personas

    def load_generation(self) -> dict:
        """Load generation configuration."""
        if self._generation is None:
            with open(self.config_dir / "generation.yaml", "r") as f:
                self._generation = yaml.safe_load(f)
        return self._generation

    def load_quality(self) -> dict:
        """Load quality thresholds configuration."""
        if self._quality is None:
            with open(self.config_dir / "quality.yaml", "r") as f:
                self._quality = yaml.safe_load(f)
        return self._quality

    def get_enabled_models(self) -> list:
        """Get list of enabled models from configuration."""
        gen_config = self.load_generation()
        return [m for m in gen_config.get("models", []) if m.get("enabled", True)]

    def sample_rating(self) -> int:
        """Sample a rating based on configured distribution."""
        gen_config = self.load_generation()
        distribution = gen_config.get("rating_distribution", {})
        ratings = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(ratings, weights=weights, k=1)[0]

    def sample_persona(self) -> dict:
        """Sample a random persona."""
        personas = self.load_personas().get("personas", [])
        return random.choice(personas)

    def sample_product(self) -> dict:
        """Sample a random product."""
        gen_config = self.load_generation()
        products = gen_config.get("products", [])
        return random.choice(products)


class PromptBuilder:
    """Builds prompts for review generation."""

    RATING_SENTIMENTS = {
        5: "extremely positive and enthusiastic",
        4: "positive with minor suggestions",
        3: "balanced with both pros and cons",
        2: "mostly negative with some positives",
        1: "very negative and frustrated"
    }

    TONE_INSTRUCTIONS = {
        "casual": "Use casual, conversational language. Include contractions and informal expressions.",
        "technical": "Use precise technical terminology. Be specific about technical details and metrics.",
        "concise": "Be brief and to the point. Focus on key facts without elaboration.",
        "balanced": "Maintain a professional, balanced tone. Consider multiple perspectives.",
        "strategic": "Focus on business value and ROI. Consider team and organizational impact.",
        "practical": "Be pragmatic and solution-oriented. Focus on real-world usage.",
        "detail_oriented": "Pay attention to specifics. Mention exact features, error messages, or behaviors.",
        "cautious": "Be thorough in evaluation. Consider edge cases and potential issues."
    }

    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader
        self.gen_config = config_loader.load_generation()

    def build_prompt(self, persona: dict, product: dict, rating: int) -> str:
        """Build a generation prompt for a review."""
        min_words = self.gen_config.get("review_length", {}).get("min_words", 40)
        max_words = self.gen_config.get("review_length", {}).get("max_words", 120)

        sentiment = self.RATING_SENTIMENTS.get(rating, "neutral")
        tone_instruction = self.TONE_INSTRUCTIONS.get(persona.get("tone", "casual"), "")

        priorities = ", ".join(persona.get("priorities", []))
        features = ", ".join(product.get("features", []))

        prompt = f"""Generate a realistic software product review with the following specifications:

PRODUCT:
- Name: {product.get('name')}
- Category: {product.get('category')}
- Key Features: {features}
- Description: {product.get('description', '')}

REVIEWER PERSONA:
- Role: {persona.get('name')}
- Experience Level: {persona.get('experience')}
- Priorities: {priorities}
- Background: {persona.get('description', '')}

REVIEW REQUIREMENTS:
- Star Rating: {rating}/5 stars
- Sentiment: {sentiment}
- Length: {min_words}-{max_words} words
- Tone: {tone_instruction}

IMPORTANT GUIDELINES:
1. Write from the first-person perspective of the persona
2. Include specific details about actual product features or behaviors
3. Mention concrete use cases or scenarios from the persona's work
4. For negative aspects, reference specific issues (e.g., error messages, performance metrics, documentation gaps)
5. Avoid generic praise or criticism - be specific
6. Do NOT use marketing language like "game-changing", "revolutionary", "best-in-class"
7. Include at least one technical term relevant to developer tools
8. Make the review sound authentic, as if posted on G2 or Capterra

OUTPUT: Write ONLY the review text, nothing else. No labels, headers, or meta-commentary."""

        return prompt

    def build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are a helpful assistant that generates realistic software product reviews.
Your reviews should sound authentic, as if written by real users with diverse backgrounds and experiences.
Always write from the specified persona's perspective and maintain consistency with their experience level and priorities.
Never break character or include meta-commentary about the review generation process."""


class LLMProvider:
    """Base class for LLM providers."""

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        max_tokens = kwargs.get("max_tokens", 500)
        temperature = kwargs.get("temperature", 0.8)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        if httpx is None:
            raise ImportError("httpx package not installed. Run: pip install httpx")
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Generate text using Ollama API."""
        full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.8),
                    "num_predict": kwargs.get("max_tokens", 500)
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        if anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        import os
        # Load API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        max_tokens = kwargs.get("max_tokens", 500)
        temperature = kwargs.get("temperature", 0.8)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.content[0].text.strip()


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        if genai is None:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        import os
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Generate text using Gemini API."""
        full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.8)
        )

        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        return response.text.strip()


class ReviewGenerator:
    """Main review generation orchestrator."""

    def __init__(self, config_dir: str = "config", output_dir: str = "data/synthetic"):
        self.config = ConfigLoader(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_builder = PromptBuilder(self.config)
        self.providers: dict[str, LLMProvider] = {}
        self._review_count = 0
        self._model_stats = {}

    def _get_provider(self, model_config: dict) -> LLMProvider:
        """Get or create an LLM provider."""
        provider_name = model_config["provider"]
        model_name = model_config["model"]
        key = f"{provider_name}:{model_name}"

        if key not in self.providers:
            if provider_name == "openai":
                self.providers[key] = OpenAIProvider(model=model_name)
            elif provider_name == "anthropic":
                self.providers[key] = AnthropicProvider(model=model_name)
            elif provider_name == "gemini":
                self.providers[key] = GeminiProvider(model=model_name)
            elif provider_name == "ollama":
                self.providers[key] = OllamaProvider(model=model_name)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        return self.providers[key]

    def _init_model_stats(self, model_config: dict):
        """Initialize statistics tracking for a model."""
        key = f"{model_config['provider']}:{model_config['model']}"
        if key not in self._model_stats:
            self._model_stats[key] = {
                "total_attempts": 0,
                "successful": 0,
                "rejected": 0,
                "total_time_ms": 0,
                "average_time_ms": 0,
                "average_quality_score": 0
            }

    def _update_model_stats(self, model_config: dict, time_ms: float, accepted: bool, quality_score: float = 0):
        """Update statistics for a model."""
        key = f"{model_config['provider']}:{model_config['model']}"
        stats = self._model_stats[key]
        stats["total_attempts"] += 1
        stats["total_time_ms"] += time_ms

        if accepted:
            stats["successful"] += 1
            # Running average for quality score
            n = stats["successful"]
            stats["average_quality_score"] = (
                stats["average_quality_score"] * (n - 1) + quality_score
            ) / n
        else:
            stats["rejected"] += 1

        stats["average_time_ms"] = stats["total_time_ms"] / stats["total_attempts"]

    def generate_single(
        self,
        model_config: Optional[dict] = None,
        persona: Optional[dict] = None,
        product: Optional[dict] = None,
        rating: Optional[int] = None
    ) -> tuple[str, ReviewMetadata, float]:
        """Generate a single review without quality evaluation."""
        # Use provided values or sample from config
        if model_config is None:
            models = self.config.get_enabled_models()
            if not models:
                raise ValueError("No enabled models in configuration")
            model_config = random.choice(models)

        if persona is None:
            persona = self.config.sample_persona()
        if product is None:
            product = self.config.sample_product()
        if rating is None:
            rating = self.config.sample_rating()

        self._init_model_stats(model_config)

        # Build prompt
        prompt = self.prompt_builder.build_prompt(persona, product, rating)
        system_prompt = self.prompt_builder.build_system_prompt()

        # Generate
        provider = self._get_provider(model_config)

        start_time = time.time()
        review_text = provider.generate(
            prompt,
            system_prompt,
            max_tokens=model_config.get("max_tokens", 500),
            temperature=model_config.get("temperature", 0.8)
        )
        generation_time = (time.time() - start_time) * 1000

        # Create metadata
        metadata = ReviewMetadata(
            persona_name=persona["name"],
            persona_experience=persona["experience"],
            persona_tone=persona["tone"],
            product_name=product["name"],
            product_category=product["category"],
            rating=rating,
            model_provider=model_config["provider"],
            model_name=model_config["model"],
            generation_time_ms=generation_time,
            attempt_number=1,
            timestamp=datetime.now().isoformat()
        )

        return review_text, metadata, generation_time

    def generate_with_quality(
        self,
        quality_evaluator,
        existing_reviews: list[str],
        model_config: Optional[dict] = None,
        persona: Optional[dict] = None,
        product: Optional[dict] = None,
        rating: Optional[int] = None,
        max_retries: int = 3
    ) -> Optional[SyntheticReview]:
        """Generate a review with quality guardrails and regeneration."""
        # Sample values once for all attempts
        if model_config is None:
            models = self.config.get_enabled_models()
            model_config = random.choice(models)
        if persona is None:
            persona = self.config.sample_persona()
        if product is None:
            product = self.config.sample_product()
        if rating is None:
            rating = self.config.sample_rating()

        for attempt in range(1, max_retries + 1):
            try:
                review_text, metadata, gen_time = self.generate_single(
                    model_config=model_config,
                    persona=persona,
                    product=product,
                    rating=rating
                )
                metadata.attempt_number = attempt

                # Evaluate quality
                quality_metrics = quality_evaluator.evaluate(
                    review_text,
                    existing_reviews,
                    metadata.rating
                )

                # Update stats
                self._update_model_stats(
                    model_config,
                    gen_time,
                    quality_metrics.accepted,
                    quality_metrics.overall_score
                )

                if quality_metrics.accepted:
                    self._review_count += 1
                    review = SyntheticReview(
                        id=f"review_{self._review_count:05d}",
                        review_text=review_text,
                        metadata=metadata,
                        quality_metrics=quality_metrics
                    )
                    return review
                else:
                    logger.debug(
                        f"Review rejected (attempt {attempt}/{max_retries}): "
                        f"{quality_metrics.rejection_reasons}"
                    )

            except Exception as e:
                logger.error(f"Generation error (attempt {attempt}): {e}")
                continue

        logger.warning(f"Failed to generate acceptable review after {max_retries} attempts")
        return None

    def generate_batch(
        self,
        quality_evaluator,
        num_samples: int,
        progress_callback=None
    ) -> Generator[SyntheticReview, None, None]:
        """Generate a batch of reviews with quality control."""
        existing_reviews = []
        gen_config = self.config.load_generation()
        max_retries = gen_config.get("generation", {}).get("max_retries", 3)

        generated = 0
        failed = 0

        while generated < num_samples:
            review = self.generate_with_quality(
                quality_evaluator,
                existing_reviews,
                max_retries=max_retries
            )

            if review:
                existing_reviews.append(review.review_text)
                generated += 1

                if progress_callback:
                    progress_callback(generated, num_samples, failed)

                yield review
            else:
                failed += 1
                if failed > num_samples * 0.5:
                    logger.error("Too many failed generations, stopping")
                    break

    def save_review(self, review: SyntheticReview, filename: str = "synthetic_reviews.jsonl"):
        """Append a review to the output file."""
        output_path = self.output_dir / filename
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(review.to_dict()) + "\n")

    def save_batch(self, reviews: list[SyntheticReview], filename: str = "synthetic_reviews.jsonl"):
        """Save a batch of reviews to the output file."""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for review in reviews:
                f.write(json.dumps(review.to_dict()) + "\n")

    def load_reviews(self, filename: str = "synthetic_reviews.jsonl") -> list[SyntheticReview]:
        """Load reviews from the output file."""
        output_path = self.output_dir / filename
        reviews = []
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        reviews.append(SyntheticReview.from_dict(json.loads(line)))
        return reviews

    def get_model_stats(self) -> dict:
        """Get generation statistics per model."""
        return self._model_stats.copy()

    def clear_output(self, filename: str = "synthetic_reviews.jsonl"):
        """Clear the output file."""
        output_path = self.output_dir / filename
        if output_path.exists():
            output_path.unlink()
        self._review_count = 0


def generate_reviews(
    num_samples: int = 100,
    config_dir: str = "config",
    output_dir: str = "data/synthetic",
    progress_callback=None
) -> list[SyntheticReview]:
    """
    High-level function to generate synthetic reviews.

    Args:
        num_samples: Number of reviews to generate
        config_dir: Path to configuration directory
        output_dir: Path to output directory
        progress_callback: Optional callback(current, total, failed)

    Returns:
        List of generated SyntheticReview objects
    """
    from src.quality.rejection import QualityEvaluator

    generator = ReviewGenerator(config_dir, output_dir)
    quality_evaluator = QualityEvaluator(config_dir)

    reviews = []
    for review in generator.generate_batch(quality_evaluator, num_samples, progress_callback):
        generator.save_review(review)
        reviews.append(review)

    return reviews
