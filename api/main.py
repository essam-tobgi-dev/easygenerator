"""
FastAPI application for Synthetic Review Data Generator.

Provides REST API endpoints for review generation, evaluation, and reporting.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate import ReviewGenerator, SyntheticReview, ConfigLoader
from src.quality.rejection import QualityEvaluator
from src.compare import DatasetComparator
from src.report import ReportGenerator
from src.llm import SUPPORTED_MODELS, get_all_supported_models


# Global state for background tasks
generation_tasks = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    app.state.config_dir = "config"
    app.state.data_dir = "data"
    app.state.output_dir = "data/synthetic"
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Synthetic Review Data Generator API",
    description="Generate high-quality synthetic reviews for SaaS Developer Tools with quality guardrails",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 Router
api_router = APIRouter(prefix="/api/v1")


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerationRequest(BaseModel):
    """Request model for review generation."""
    num_samples: int = Field(default=10, ge=1, le=1000, description="Number of reviews to generate")
    persona_name: Optional[str] = Field(default=None, description="Specific persona to use")
    product_name: Optional[str] = Field(default=None, description="Specific product to use")
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="Specific rating to use")
    model: Optional[str] = Field(default=None, description="Model to use (e.g., 'openai:gpt-4o-mini')")


class SingleReviewRequest(BaseModel):
    """Request model for single review generation."""
    persona_name: Optional[str] = None
    product_name: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    model: Optional[str] = None
    evaluate: bool = Field(default=True, description="Whether to evaluate quality")


class EvaluationRequest(BaseModel):
    """Request model for review evaluation."""
    text: str = Field(..., description="Review text to evaluate")
    rating: int = Field(..., ge=1, le=5, description="Star rating")


class ReviewResponse(BaseModel):
    """Response model for a single review."""
    id: str
    review_text: str
    product_name: str
    persona_name: str
    rating: int
    model: str
    generation_time_ms: float
    quality_metrics: Optional[dict] = None


class GenerationStatus(BaseModel):
    """Response model for generation task status."""
    task_id: str
    status: str
    progress: int
    total: int
    failed: int
    completed_at: Optional[str] = None


class EvaluationResponse(BaseModel):
    """Response model for quality evaluation."""
    diversity_score: float
    realism_score: float
    bias_score: float
    overall_score: float
    accepted: bool
    issues: list[str]


class ComparisonResponse(BaseModel):
    """Response model for dataset comparison."""
    real_count: int
    synthetic_count: int
    quality_scores: dict
    differences: dict
    summary: dict


class ConfigResponse(BaseModel):
    """Response model for configuration."""
    personas: list[dict]
    products: list[dict]
    models: list[dict]
    rating_distribution: dict


# ============================================================================
# Helper Functions
# ============================================================================

def get_generator():
    """Get or create ReviewGenerator instance."""
    return ReviewGenerator(app.state.config_dir, app.state.output_dir)


def get_evaluator():
    """Get or create QualityEvaluator instance."""
    return QualityEvaluator(app.state.config_dir)


def find_persona_by_name(config: ConfigLoader, name: str) -> Optional[dict]:
    """Find persona by name."""
    personas = config.load_personas().get("personas", [])
    for p in personas:
        if p["name"].lower() == name.lower():
            return p
    return None


def find_product_by_name(config: ConfigLoader, name: str) -> Optional[dict]:
    """Find product by name."""
    products = config.load_generation().get("products", [])
    for p in products:
        if p["name"].lower() == name.lower():
            return p
    return None


def find_model_by_key(config: ConfigLoader, key: str) -> Optional[dict]:
    """Find model by provider:model key. Falls back to supported models."""
    # First try config (has user-defined settings)
    models = config.load_generation().get("models", [])
    for m in models:
        if f"{m['provider']}:{m['model']}" == key:
            return m

    # Fall back to supported models with default settings
    if ":" in key:
        provider, model = key.split(":", 1)
        if provider in SUPPORTED_MODELS:
            for model_info in SUPPORTED_MODELS[provider]:
                if model_info["model"] == model:
                    return {
                        "provider": provider,
                        "model": model,
                        "enabled": True,
                        "max_tokens": 500,
                        "temperature": 0.8
                    }
    return None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Synthetic Review Data Generator API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "api_base": "/api/v1"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ----------------------------------------------------------------------------
# Configuration Endpoints
# ----------------------------------------------------------------------------

@api_router.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """Get current configuration."""
    config = ConfigLoader(app.state.config_dir)

    personas = config.load_personas().get("personas", [])
    gen_config = config.load_generation()
    products = gen_config.get("products", [])
    models = gen_config.get("models", [])
    rating_dist = gen_config.get("rating_distribution", {})

    return ConfigResponse(
        personas=personas,
        products=products,
        models=models,
        rating_distribution=rating_dist
    )


@api_router.get("/config/personas")
async def get_personas():
    """Get available personas."""
    config = ConfigLoader(app.state.config_dir)
    return config.load_personas().get("personas", [])


@api_router.get("/config/products")
async def get_products():
    """Get available products."""
    config = ConfigLoader(app.state.config_dir)
    return config.load_generation().get("products", [])


@api_router.get("/config/models")
async def get_models():
    """Get configured models from generation.yaml."""
    config = ConfigLoader(app.state.config_dir)
    return config.load_generation().get("models", [])


@api_router.get("/config/available-models")
async def get_available_models():
    """Get all supported models across all providers."""
    return get_all_supported_models()


@api_router.get("/config/supported-models")
async def get_supported_models():
    """Get supported models grouped by provider."""
    return SUPPORTED_MODELS


# ----------------------------------------------------------------------------
# Generation Endpoints
# ----------------------------------------------------------------------------

@api_router.post("/generate/single", response_model=ReviewResponse)
async def generate_single_review(request: SingleReviewRequest):
    """Generate a single review."""
    generator = get_generator()
    config = ConfigLoader(app.state.config_dir)

    # Resolve optional parameters
    persona = None
    if request.persona_name:
        persona = find_persona_by_name(config, request.persona_name)
        if not persona:
            raise HTTPException(status_code=400, detail=f"Persona not found: {request.persona_name}")

    product = None
    if request.product_name:
        product = find_product_by_name(config, request.product_name)
        if not product:
            raise HTTPException(status_code=400, detail=f"Product not found: {request.product_name}")

    model_config = None
    if request.model:
        model_config = find_model_by_key(config, request.model)
        if not model_config:
            raise HTTPException(status_code=400, detail=f"Model not found: {request.model}")

    try:
        review_text, metadata, gen_time = generator.generate_single(
            model_config=model_config,
            persona=persona,
            product=product,
            rating=request.rating
        )

        quality_metrics = None
        if request.evaluate:
            evaluator = get_evaluator()
            metrics = evaluator.evaluate(review_text, [], metadata.rating)
            quality_metrics = metrics.to_dict()

        return ReviewResponse(
            id=f"review_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            review_text=review_text,
            product_name=metadata.product_name,
            persona_name=metadata.persona_name,
            rating=metadata.rating,
            model=f"{metadata.model_provider}:{metadata.model_name}",
            generation_time_ms=gen_time,
            quality_metrics=quality_metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/generate/batch")
async def start_batch_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """Start batch generation as a background task."""
    import uuid

    task_id = str(uuid.uuid4())[:8]

    generation_tasks[task_id] = {
        "status": "running",
        "progress": 0,
        "total": request.num_samples,
        "failed": 0,
        "reviews": [],
        "completed_at": None
    }

    async def run_generation():
        generator = get_generator()
        evaluator = get_evaluator()
        config = ConfigLoader(app.state.config_dir)

        # Resolve parameters
        persona = None
        if request.persona_name:
            persona = find_persona_by_name(config, request.persona_name)

        product = None
        if request.product_name:
            product = find_product_by_name(config, request.product_name)

        model_config = None
        if request.model:
            model_config = find_model_by_key(config, request.model)

        existing_texts = []
        task = generation_tasks[task_id]

        for i in range(request.num_samples):
            try:
                review = generator.generate_with_quality(
                    evaluator,
                    existing_texts,
                    model_config=model_config,
                    persona=persona,
                    product=product,
                    rating=request.rating
                )

                if review:
                    generator.save_review(review)
                    existing_texts.append(review.review_text)
                    task["reviews"].append(review.to_dict())
                    task["progress"] = i + 1
                else:
                    task["failed"] += 1

            except Exception as e:
                task["failed"] += 1

            # Allow other tasks to run
            await asyncio.sleep(0)

        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()

    background_tasks.add_task(run_generation)

    return {"task_id": task_id, "message": "Generation started"}


@api_router.get("/generate/status/{task_id}", response_model=GenerationStatus)
async def get_generation_status(task_id: str):
    """Get status of a batch generation task."""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = generation_tasks[task_id]
    return GenerationStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        total=task["total"],
        failed=task["failed"],
        completed_at=task.get("completed_at")
    )


@api_router.get("/generate/results/{task_id}")
async def get_generation_results(task_id: str):
    """Get results of a completed batch generation task."""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = generation_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "total_generated": len(task["reviews"]),
        "failed": task["failed"],
        "reviews": task["reviews"]
    }


# ----------------------------------------------------------------------------
# Evaluation Endpoints
# ----------------------------------------------------------------------------

@api_router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_review(request: EvaluationRequest):
    """Evaluate quality of a review."""
    evaluator = get_evaluator()

    metrics = evaluator.evaluate(request.text, [], request.rating)

    return EvaluationResponse(
        diversity_score=metrics.diversity_score,
        realism_score=metrics.realism_score,
        bias_score=metrics.bias_score,
        overall_score=metrics.overall_score,
        accepted=metrics.accepted,
        issues=metrics.rejection_reasons
    )


@api_router.get("/evaluate/detailed")
async def get_detailed_evaluation(
    text: str = Query(..., description="Review text to evaluate"),
    rating: int = Query(..., ge=1, le=5, description="Star rating")
):
    """Get detailed quality analysis of a review."""
    evaluator = get_evaluator()
    return evaluator.get_detailed_analysis(text, rating)


# ----------------------------------------------------------------------------
# Dataset Endpoints
# ----------------------------------------------------------------------------

@api_router.get("/dataset/synthetic")
async def get_synthetic_reviews(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Get synthetic reviews from the dataset."""
    generator = get_generator()
    reviews = generator.load_reviews()

    total = len(reviews)
    paginated = reviews[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "reviews": [r.to_dict() for r in paginated]
    }


@api_router.get("/dataset/real")
async def get_real_reviews():
    """Get real reviews baseline dataset."""
    path = Path(app.state.data_dir) / "real_reviews" / "real_reviews.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Real reviews file not found")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@api_router.delete("/dataset/synthetic")
async def clear_synthetic_dataset():
    """Clear the synthetic reviews dataset."""
    generator = get_generator()
    generator.clear_output()
    return {"message": "Synthetic dataset cleared"}


# ----------------------------------------------------------------------------
# Comparison Endpoints
# ----------------------------------------------------------------------------

@api_router.get("/compare", response_model=ComparisonResponse)
async def compare_datasets():
    """Compare synthetic and real review datasets."""
    comparator = DatasetComparator(
        real_reviews_path=str(Path(app.state.data_dir) / "real_reviews" / "real_reviews.json"),
        synthetic_reviews_path=str(Path(app.state.output_dir) / "synthetic_reviews.jsonl")
    )

    try:
        comparison = comparator.compare()

        return ComparisonResponse(
            real_count=comparison["real"]["basic"]["total_reviews"],
            synthetic_count=comparison["synthetic"]["basic"]["total_reviews"],
            quality_scores=comparison["quality_scores"],
            differences=comparison["differences"],
            summary=comparison["summary"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/compare/table")
async def get_comparison_table():
    """Get comparison as formatted table."""
    comparator = DatasetComparator(
        real_reviews_path=str(Path(app.state.data_dir) / "real_reviews" / "real_reviews.json"),
        synthetic_reviews_path=str(Path(app.state.output_dir) / "synthetic_reviews.jsonl")
    )

    return {"table": comparator.generate_comparison_table()}


# ----------------------------------------------------------------------------
# Report Endpoints
# ----------------------------------------------------------------------------

@api_router.get("/report")
async def get_report(format: str = Query(default="json", enum=["json", "markdown"])):
    """Get quality report."""
    report_generator = ReportGenerator(
        config_dir=app.state.config_dir,
        data_dir=app.state.data_dir,
        reports_dir="reports"
    )

    if format == "json":
        return report_generator.generate_json_report()
    else:
        return {"markdown": report_generator.generate_markdown_report()}


@api_router.post("/report/save")
async def save_report():
    """Generate and save quality report to file."""
    report_generator = ReportGenerator(
        config_dir=app.state.config_dir,
        data_dir=app.state.data_dir,
        reports_dir="reports"
    )

    output_path = report_generator.save_report()
    return {"message": "Report saved", "path": str(output_path)}


@api_router.get("/report/download")
async def download_report():
    """Download the quality report file."""
    report_path = Path("reports") / "quality_report.md"

    if not report_path.exists():
        # Generate report first
        report_generator = ReportGenerator(
            config_dir=app.state.config_dir,
            data_dir=app.state.data_dir,
            reports_dir="reports"
        )
        report_generator.save_report()

    return FileResponse(
        path=str(report_path),
        media_type="text/markdown",
        filename="quality_report.md"
    )


# ----------------------------------------------------------------------------
# Statistics Endpoints
# ----------------------------------------------------------------------------

@api_router.get("/stats")
async def get_statistics():
    """Get overall statistics."""
    generator = get_generator()
    reviews = generator.load_reviews()

    if not reviews:
        return {
            "total_reviews": 0,
            "message": "No synthetic reviews generated yet"
        }

    # Calculate statistics
    ratings = [r.metadata.rating for r in reviews]
    scores = [r.quality_metrics.overall_score for r in reviews if r.quality_metrics]

    personas = {}
    products = {}
    models = {}

    for r in reviews:
        personas[r.metadata.persona_name] = personas.get(r.metadata.persona_name, 0) + 1
        products[r.metadata.product_name] = products.get(r.metadata.product_name, 0) + 1
        model_key = f"{r.metadata.model_provider}:{r.metadata.model_name}"
        models[model_key] = models.get(model_key, 0) + 1

    return {
        "total_reviews": len(reviews),
        "average_rating": sum(ratings) / len(ratings) if ratings else 0,
        "average_quality_score": sum(scores) / len(scores) if scores else 0,
        "rating_distribution": dict(sorted(
            {r: ratings.count(r) for r in set(ratings)}.items()
        )),
        "persona_distribution": personas,
        "product_distribution": products,
        "model_distribution": models
    }


# ============================================================================
# Include Router
# ============================================================================

app.include_router(api_router)


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
