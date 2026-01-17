"""
Gradio GUI for Synthetic Review Data Generator.

Provides a web interface for review generation, evaluation, and analysis.
This GUI directly uses the Python modules without going through FastAPI.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generate import ReviewGenerator, ConfigLoader
from src.quality.rejection import QualityEvaluator
from src.compare import DatasetComparator, DatasetAnalyzer
from src.report import ReportGenerator
from src.llm import SUPPORTED_MODELS, get_all_supported_models


# ============================================================================
# Configuration
# ============================================================================

CONFIG_DIR = "config"
DATA_DIR = "data"
OUTPUT_DIR = "data/synthetic"


# ============================================================================
# Helper Functions
# ============================================================================

def get_persona_choices():
    """Get list of persona names for dropdown."""
    config = ConfigLoader(CONFIG_DIR)
    personas = config.load_personas().get("personas", [])
    return ["Random"] + [p["name"] for p in personas]


def get_product_choices():
    """Get list of product names for dropdown."""
    config = ConfigLoader(CONFIG_DIR)
    products = config.load_generation().get("products", [])
    return ["Random"] + [p["name"] for p in products]


def get_model_choices():
    """Get list of all supported models for dropdown."""
    all_models = get_all_supported_models()
    return ["Random"] + [m["key"] for m in all_models]


def get_model_choices_by_provider():
    """Get models grouped by provider for organized dropdown."""
    choices = ["Random"]
    for provider, models in SUPPORTED_MODELS.items():
        for model_info in models:
            choices.append(f"{provider}:{model_info['model']}")
    return choices


def find_persona_by_name(name: str):
    """Find persona by name."""
    config = ConfigLoader(CONFIG_DIR)
    personas = config.load_personas().get("personas", [])
    for p in personas:
        if p["name"] == name:
            return p
    return None


def find_product_by_name(name: str):
    """Find product by name."""
    config = ConfigLoader(CONFIG_DIR)
    products = config.load_generation().get("products", [])
    for p in products:
        if p["name"] == name:
            return p
    return None


def find_model_by_key(key: str):
    """Find model by provider:model key. Falls back to supported models if not in config."""
    # First try to find in config (has user-defined settings)
    config = ConfigLoader(CONFIG_DIR)
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
# Generation Functions
# ============================================================================

def generate_single_review(persona_choice, product_choice, rating_choice, model_choice, evaluate_quality):
    """Generate a single review."""
    try:
        generator = ReviewGenerator(CONFIG_DIR, OUTPUT_DIR)

        # Parse choices
        persona = None if persona_choice == "Random" else find_persona_by_name(persona_choice)
        product = None if product_choice == "Random" else find_product_by_name(product_choice)
        model_config = None if model_choice == "Random" else find_model_by_key(model_choice)
        rating = None if rating_choice == "Random" else int(rating_choice)

        # Generate
        review_text, metadata, gen_time = generator.generate_single(
            model_config=model_config,
            persona=persona,
            product=product,
            rating=rating
        )

        # Build result
        result = f"""## Generated Review

**Product:** {metadata.product_name}
**Persona:** {metadata.persona_name} ({metadata.persona_experience})
**Rating:** {'⭐' * metadata.rating}
**Model:** {metadata.model_provider}:{metadata.model_name}
**Generation Time:** {gen_time:.0f}ms

---

{review_text}
"""

        # Evaluate if requested
        quality_result = ""
        if evaluate_quality:
            evaluator = QualityEvaluator(CONFIG_DIR)
            metrics = evaluator.evaluate(review_text, [], metadata.rating)

            quality_result = f"""
---

## Quality Evaluation

| Metric | Score |
|--------|-------|
| Diversity | {metrics.diversity_score:.3f} |
| Realism | {metrics.realism_score:.3f} |
| Bias | {metrics.bias_score:.3f} |
| **Overall** | **{metrics.overall_score:.3f}** |

**Status:** {'✅ Accepted' if metrics.accepted else '❌ Rejected'}
"""
            if metrics.rejection_reasons:
                quality_result += "\n**Issues:**\n"
                for reason in metrics.rejection_reasons:
                    quality_result += f"- {reason}\n"

        return result + quality_result

    except Exception as e:
        return f"❌ Error: {str(e)}"


def generate_batch_reviews(num_samples, persona_choice, product_choice, model_choice, progress=gr.Progress()):
    """Generate a batch of reviews with progress tracking."""
    try:
        generator = ReviewGenerator(CONFIG_DIR, OUTPUT_DIR)
        evaluator = QualityEvaluator(CONFIG_DIR)

        # Parse choices
        persona = None if persona_choice == "Random" else find_persona_by_name(persona_choice)
        product = None if product_choice == "Random" else find_product_by_name(product_choice)
        model_config = None if model_choice == "Random" else find_model_by_key(model_choice)

        existing_texts = []
        generated = 0
        failed = 0
        reviews = []

        for i in progress.tqdm(range(num_samples), desc="Generating reviews"):
            review = generator.generate_with_quality(
                evaluator,
                existing_texts,
                model_config=model_config,
                persona=persona,
                product=product
            )

            if review:
                generator.save_review(review)
                existing_texts.append(review.review_text)
                reviews.append(review)
                generated += 1
            else:
                failed += 1

        # Build summary
        result = f"""## Batch Generation Complete

**Generated:** {generated} reviews
**Failed:** {failed} reviews
**Success Rate:** {generated / num_samples * 100:.1f}%

### Sample Reviews

"""
        # Show first 3 reviews
        for review in reviews[:3]:
            result += f"""---

**{review.metadata.product_name}** ({review.metadata.rating}⭐) by {review.metadata.persona_name}

{review.review_text[:200]}{'...' if len(review.review_text) > 200 else ''}

Quality Score: {review.quality_metrics.overall_score:.3f}

"""

        if len(reviews) > 3:
            result += f"\n*...and {len(reviews) - 3} more reviews saved to {OUTPUT_DIR}/synthetic_reviews.jsonl*"

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_custom_review(review_text, rating):
    """Evaluate a custom review text."""
    try:
        evaluator = QualityEvaluator(CONFIG_DIR)
        analysis = evaluator.get_detailed_analysis(review_text, int(rating))

        result = f"""## Quality Analysis

### Scores

| Metric | Score |
|--------|-------|
| Diversity | {analysis['scores']['diversity']:.3f} |
| Realism | {analysis['scores']['realism']:.3f} |
| Bias | {analysis['scores']['bias']:.3f} |
| **Overall** | **{analysis['scores']['overall']:.3f}** |

### Issues Found

"""
        all_issues = analysis['issues']['diversity'] + analysis['issues']['realism'] + analysis['issues']['bias']
        if all_issues:
            for issue in all_issues:
                result += f"- {issue}\n"
        else:
            result += "No issues found! ✅\n"

        result += f"""
### Realism Details

**Domain Terms Found:**
- Technical: {', '.join(analysis['realism_details']['domain_terms']['technical']) or 'None'}
- Feature: {', '.join(analysis['realism_details']['domain_terms']['feature']) or 'None'}
- Complaint: {', '.join(analysis['realism_details']['domain_terms']['complaint']) or 'None'}

**Specificity Score:** {analysis['realism_details']['specificity_score']:.2f}
**Word Count:** {analysis['word_count']}
"""

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Comparison Functions
# ============================================================================

def run_comparison():
    """Run comparison between synthetic and real datasets."""
    try:
        comparator = DatasetComparator(
            real_reviews_path=f"{DATA_DIR}/real_reviews/real_reviews.json",
            synthetic_reviews_path=f"{OUTPUT_DIR}/synthetic_reviews.jsonl"
        )

        comparison = comparator.compare()

        result = f"""## Dataset Comparison

### Dataset Sizes

| Dataset | Reviews |
|---------|---------|
| Real | {comparison['real']['basic']['total_reviews']} |
| Synthetic | {comparison['synthetic']['basic']['total_reviews']} |

### Quality Similarity Scores

| Metric | Score |
|--------|-------|
| Length Similarity | {comparison['quality_scores']['length_similarity']:.1%} |
| Vocabulary Similarity | {comparison['quality_scores']['vocabulary_similarity']:.1%} |
| Sentiment Similarity | {comparison['quality_scores']['sentiment_similarity']:.1%} |
| Rating Distribution | {comparison['quality_scores']['rating_distribution_similarity']:.1%} |
| **Overall Similarity** | **{comparison['quality_scores']['overall']:.1%}** |

### Key Metrics

| Metric | Real | Synthetic |
|--------|------|-----------|
| Avg Word Count | {comparison['real']['basic']['word_count']['mean']:.1f} | {comparison['synthetic']['basic']['word_count']['mean']:.1f} |
| Type-Token Ratio | {comparison['real']['vocabulary']['type_token_ratio']:.3f} | {comparison['synthetic']['vocabulary']['type_token_ratio']:.3f} |
| Positive Sentiment | {comparison['real']['sentiment']['positive_ratio']:.1%} | {comparison['synthetic']['sentiment']['positive_ratio']:.1%} |
| Negative Sentiment | {comparison['real']['sentiment']['negative_ratio']:.1%} | {comparison['synthetic']['sentiment']['negative_ratio']:.1%} |

### Analysis

**Strengths:**
"""
        for s in comparison['summary'].get('strengths', []):
            result += f"- ✅ {s}\n"

        result += "\n**Weaknesses:**\n"
        for w in comparison['summary'].get('weaknesses', []):
            result += f"- ⚠️ {w}\n"

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Report Functions
# ============================================================================

def generate_report():
    """Generate and return the quality report."""
    try:
        report_gen = ReportGenerator(
            config_dir=CONFIG_DIR,
            data_dir=DATA_DIR,
            reports_dir="reports"
        )

        report = report_gen.generate_markdown_report()
        report_gen.save_report()

        return report

    except Exception as e:
        return f"❌ Error: {str(e)}"


def generate_report_pdf_download():
    """Generate PDF report and return file path for download."""
    try:
        report_gen = ReportGenerator(
            config_dir=CONFIG_DIR,
            data_dir=DATA_DIR,
            reports_dir="reports"
        )

        pdf_path = report_gen.save_report_as_pdf()
        return str(pdf_path)

    except ImportError:
        return None
    except Exception as e:
        return None


# ============================================================================
# Dataset Functions
# ============================================================================

def view_synthetic_dataset(limit):
    """View synthetic reviews dataset."""
    try:
        generator = ReviewGenerator(CONFIG_DIR, OUTPUT_DIR)
        reviews = generator.load_reviews()

        if not reviews:
            return "No synthetic reviews generated yet. Use the Generation tab to create some!"

        result = f"## Synthetic Reviews ({len(reviews)} total)\n\n"

        for review in reviews[:int(limit)]:
            result += f"""---

**ID:** {review.id}
**Product:** {review.metadata.product_name} | **Rating:** {'⭐' * review.metadata.rating}
**Persona:** {review.metadata.persona_name}
**Model:** {review.metadata.model_provider}:{review.metadata.model_name}
**Quality Score:** {f"{review.quality_metrics.overall_score:.3f}" if review.quality_metrics else "N/A"}

{review.review_text}

"""

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


def view_real_dataset():
    """View real reviews baseline dataset."""
    try:
        path = Path(DATA_DIR) / "real_reviews" / "real_reviews.json"

        if not path.exists():
            return "Real reviews file not found."

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        reviews = data.get("reviews", [])
        result = f"## Real Reviews Baseline ({len(reviews)} total)\n\n"

        for review in reviews[:20]:
            result += f"""---

**ID:** {review.get('id', 'N/A')}
**Product:** {review.get('product', 'N/A')} | **Rating:** {'⭐' * review.get('rating', 0)}
**Reviewer:** {review.get('reviewer_title', 'N/A')}
**Source:** {review.get('source', 'N/A')}

{review.get('text', '')}

"""

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


def clear_synthetic_dataset():
    """Clear the synthetic dataset."""
    try:
        generator = ReviewGenerator(CONFIG_DIR, OUTPUT_DIR)
        generator.clear_output()
        return "✅ Synthetic dataset cleared successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_statistics():
    """Get dataset statistics."""
    try:
        generator = ReviewGenerator(CONFIG_DIR, OUTPUT_DIR)
        reviews = generator.load_reviews()

        if not reviews:
            return "No synthetic reviews generated yet."

        # Calculate stats
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

        result = f"""## Dataset Statistics

**Total Reviews:** {len(reviews)}
**Average Rating:** {sum(ratings) / len(ratings):.2f}
**Average Quality Score:** {sum(scores) / len(scores):.3f if scores else 'N/A'}

### Rating Distribution

| Rating | Count | Percentage |
|--------|-------|------------|
"""
        for rating in sorted(set(ratings), reverse=True):
            count = ratings.count(rating)
            pct = count / len(ratings) * 100
            result += f"| {'⭐' * rating} | {count} | {pct:.1f}% |\n"

        result += "\n### Persona Distribution\n\n| Persona | Count |\n|---------|-------|\n"
        for persona, count in sorted(personas.items(), key=lambda x: -x[1]):
            result += f"| {persona} | {count} |\n"

        result += "\n### Product Distribution\n\n| Product | Count |\n|---------|-------|\n"
        for product, count in sorted(products.items(), key=lambda x: -x[1]):
            result += f"| {product} | {count} |\n"

        result += "\n### Model Distribution\n\n| Model | Count |\n|-------|-------|\n"
        for model, count in sorted(models.items(), key=lambda x: -x[1]):
            result += f"| {model} | {count} |\n"

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Build Gradio Interface
# ============================================================================

def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Synthetic Review Generator",
        theme=gr.themes.Soft(),
        css="""
            .container { max-width: 1200px; margin: auto; }
            .header { text-align: center; margin-bottom: 20px; }
        """
    ) as demo:

        gr.Markdown("""
        # Synthetic Review Data Generator

        Generate high-quality synthetic reviews for SaaS Developer Tools with configurable personas,
        multiple LLM providers, and quality guardrails.
        """)

        with gr.Tabs():

            # =================================================================
            # Generation Tab
            # =================================================================
            with gr.TabItem("Generation"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Single Review Generation")

                        persona_single = gr.Dropdown(
                            choices=get_persona_choices(),
                            value="Random",
                            label="Persona"
                        )
                        product_single = gr.Dropdown(
                            choices=get_product_choices(),
                            value="Random",
                            label="Product"
                        )
                        rating_single = gr.Dropdown(
                            choices=["Random", "5", "4", "3", "2", "1"],
                            value="Random",
                            label="Rating"
                        )
                        model_single = gr.Dropdown(
                            choices=get_model_choices(),
                            value="Random",
                            label="Model"
                        )
                        evaluate_single = gr.Checkbox(
                            value=True,
                            label="Evaluate Quality"
                        )

                        generate_single_btn = gr.Button("Generate Single Review", variant="primary")

                    with gr.Column(scale=2):
                        single_output = gr.Markdown(label="Generated Review")

                generate_single_btn.click(
                    generate_single_review,
                    inputs=[persona_single, product_single, rating_single, model_single, evaluate_single],
                    outputs=single_output
                )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Generation")

                        num_samples = gr.Slider(
                            minimum=1,
                            maximum=500,
                            value=10,
                            step=1,
                            label="Number of Reviews"
                        )
                        persona_batch = gr.Dropdown(
                            choices=get_persona_choices(),
                            value="Random",
                            label="Persona Filter"
                        )
                        product_batch = gr.Dropdown(
                            choices=get_product_choices(),
                            value="Random",
                            label="Product Filter"
                        )
                        model_batch = gr.Dropdown(
                            choices=get_model_choices(),
                            value="Random",
                            label="Model"
                        )

                        generate_batch_btn = gr.Button("Generate Batch", variant="primary")

                    with gr.Column(scale=2):
                        batch_output = gr.Markdown(label="Batch Results")

                generate_batch_btn.click(
                    generate_batch_reviews,
                    inputs=[num_samples, persona_batch, product_batch, model_batch],
                    outputs=batch_output
                )

            # =================================================================
            # Evaluation Tab
            # =================================================================
            with gr.TabItem("Evaluation"):

                gr.Markdown("### Evaluate Custom Review")
                gr.Markdown("Enter a review to analyze its quality metrics.")

                with gr.Row():
                    with gr.Column(scale=1):
                        eval_text = gr.Textbox(
                            lines=5,
                            placeholder="Enter review text here...",
                            label="Review Text"
                        )
                        eval_rating = gr.Dropdown(
                            choices=["5", "4", "3", "2", "1"],
                            value="4",
                            label="Rating"
                        )
                        evaluate_btn = gr.Button("Evaluate", variant="primary")

                    with gr.Column(scale=1):
                        eval_output = gr.Markdown(label="Evaluation Results")

                evaluate_btn.click(
                    evaluate_custom_review,
                    inputs=[eval_text, eval_rating],
                    outputs=eval_output
                )

            # =================================================================
            # Comparison Tab
            # =================================================================
            with gr.TabItem("Comparison"):

                gr.Markdown("### Compare Synthetic vs Real Reviews")
                gr.Markdown("Analyze how synthetic reviews compare to the real baseline dataset.")

                compare_btn = gr.Button("Run Comparison", variant="primary")
                comparison_output = gr.Markdown()

                compare_btn.click(
                    run_comparison,
                    outputs=comparison_output
                )

            # =================================================================
            # Report Tab
            # =================================================================
            with gr.TabItem("Report"):

                gr.Markdown("### Quality Report")
                gr.Markdown("Generate a comprehensive quality report for the synthetic dataset.")

                with gr.Row():
                    report_btn = gr.Button("Generate Report", variant="primary")
                    pdf_btn = gr.Button("Download as PDF", variant="secondary")

                report_output = gr.Markdown()
                pdf_download = gr.File(label="Download PDF Report", visible=False)

                report_btn.click(
                    generate_report,
                    outputs=report_output
                )

                def generate_and_show_pdf():
                    pdf_path = generate_report_pdf_download()
                    if pdf_path:
                        return gr.update(value=pdf_path, visible=True)
                    return gr.update(value=None, visible=True)

                pdf_btn.click(
                    generate_and_show_pdf,
                    outputs=pdf_download
                )

            # =================================================================
            # Dataset Tab
            # =================================================================
            with gr.TabItem("Dataset"):

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### View Synthetic Reviews")
                        view_limit = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=10,
                            step=5,
                            label="Number to Display"
                        )
                        view_synthetic_btn = gr.Button("View Synthetic")
                        clear_btn = gr.Button("Clear Dataset", variant="stop")

                    with gr.Column():
                        gr.Markdown("### View Real Reviews")
                        view_real_btn = gr.Button("View Real Baseline")

                with gr.Row():
                    gr.Markdown("### Statistics")
                    stats_btn = gr.Button("Get Statistics")

                dataset_output = gr.Markdown()

                view_synthetic_btn.click(
                    view_synthetic_dataset,
                    inputs=[view_limit],
                    outputs=dataset_output
                )

                view_real_btn.click(
                    view_real_dataset,
                    outputs=dataset_output
                )

                clear_btn.click(
                    clear_synthetic_dataset,
                    outputs=dataset_output
                )

                stats_btn.click(
                    get_statistics,
                    outputs=dataset_output
                )

            # =================================================================
            # Configuration Tab
            # =================================================================
            with gr.TabItem("Configuration"):

                gr.Markdown("### Current Configuration")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Personas")

                        config = ConfigLoader(CONFIG_DIR)
                        personas = config.load_personas().get("personas", [])
                        persona_text = ""
                        for p in personas:
                            persona_text += f"**{p['name']}** ({p['experience']})\n"
                            persona_text += f"- Tone: {p['tone']}\n"
                            persona_text += f"- Priorities: {', '.join(p['priorities'])}\n\n"
                        gr.Markdown(persona_text)

                    with gr.Column():
                        gr.Markdown("#### Products")

                        products = config.load_generation().get("products", [])
                        product_text = ""
                        for p in products:
                            product_text += f"**{p['name']}** ({p['category']})\n"
                            product_text += f"- Features: {', '.join(p['features'])}\n\n"
                        gr.Markdown(product_text)

                    with gr.Column():
                        gr.Markdown("#### Available Models")

                        model_text = ""
                        for provider, provider_models in SUPPORTED_MODELS.items():
                            model_text += f"**{provider.upper()}**\n"
                            for model_info in provider_models:
                                model_text += f"- {model_info['model']}: {model_info['description']}\n"
                            model_text += "\n"
                        gr.Markdown(model_text)

                gr.Markdown("""
                ---

                ### Configuration Files

                - `config/personas.yaml` - Reviewer personas
                - `config/generation.yaml` - Generation settings and models
                - `config/quality.yaml` - Quality thresholds and domain keywords

                Edit these files to customize the generation behavior.
                """)

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
