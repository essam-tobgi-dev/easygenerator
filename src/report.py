"""
Quality report generator for synthetic review datasets.

Generates comprehensive markdown reports with statistics,
guardrail metrics, and comparison analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

from .compare import DatasetComparator, DatasetAnalyzer
from .quality.rejection import QualityEvaluator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates quality reports for synthetic review datasets."""

    def __init__(
        self,
        config_dir: str = "config",
        data_dir: str = "data",
        reports_dir: str = "reports"
    ):
        """
        Initialize report generator.

        Args:
            config_dir: Path to configuration directory
            data_dir: Path to data directory
            reports_dir: Path to reports output directory
        """
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.comparator = DatasetComparator(
            real_reviews_path=str(self.data_dir / "real_reviews" / "real_reviews.json"),
            synthetic_reviews_path=str(self.data_dir / "synthetic" / "synthetic_reviews.jsonl")
        )

    def load_synthetic_reviews(self) -> list[dict]:
        """Load synthetic reviews from JSONL file."""
        path = self.data_dir / "synthetic" / "synthetic_reviews.jsonl"
        reviews = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        reviews.append(json.loads(line))
        return reviews

    def calculate_dataset_statistics(self, reviews: list[dict]) -> dict:
        """Calculate detailed statistics for the dataset."""
        if not reviews:
            return {"error": "No reviews found"}

        # Extract quality metrics
        quality_metrics = []
        for r in reviews:
            if "quality_metrics" in r:
                quality_metrics.append(r["quality_metrics"])

        # Extract metadata
        personas = {}
        products = {}
        models = {}
        ratings = {}

        for r in reviews:
            meta = r.get("metadata", {})

            # Count personas
            persona = meta.get("persona_name", "Unknown")
            personas[persona] = personas.get(persona, 0) + 1

            # Count products
            product = meta.get("product_name", "Unknown")
            products[product] = products.get(product, 0) + 1

            # Count models
            model = f"{meta.get('model_provider', 'unknown')}:{meta.get('model_name', 'unknown')}"
            models[model] = models.get(model, 0) + 1

            # Count ratings
            rating = meta.get("rating", 0)
            ratings[rating] = ratings.get(rating, 0) + 1

        # Calculate quality score statistics
        diversity_scores = [m["diversity_score"] for m in quality_metrics if "diversity_score" in m]
        realism_scores = [m["realism_score"] for m in quality_metrics if "realism_score" in m]
        bias_scores = [m["bias_score"] for m in quality_metrics if "bias_score" in m]
        overall_scores = [m["overall_score"] for m in quality_metrics if "overall_score" in m]

        def safe_stats(values):
            if not values:
                return {"mean": 0, "min": 0, "max": 0}
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }

        # Calculate generation times
        gen_times = [r["metadata"]["generation_time_ms"] for r in reviews if "metadata" in r and "generation_time_ms" in r["metadata"]]

        return {
            "total_reviews": len(reviews),
            "quality_scores": {
                "diversity": safe_stats(diversity_scores),
                "realism": safe_stats(realism_scores),
                "bias": safe_stats(bias_scores),
                "overall": safe_stats(overall_scores)
            },
            "distribution": {
                "personas": personas,
                "products": products,
                "models": models,
                "ratings": ratings
            },
            "generation": {
                "avg_time_ms": sum(gen_times) / len(gen_times) if gen_times else 0,
                "total_time_s": sum(gen_times) / 1000 if gen_times else 0
            }
        }

    def calculate_model_performance(self, reviews: list[dict]) -> dict:
        """Calculate per-model performance statistics."""
        model_data = {}

        for r in reviews:
            meta = r.get("metadata", {})
            model_key = f"{meta.get('model_provider', 'unknown')}:{meta.get('model_name', 'unknown')}"

            if model_key not in model_data:
                model_data[model_key] = {
                    "count": 0,
                    "quality_scores": [],
                    "generation_times": [],
                    "attempts": []
                }

            model_data[model_key]["count"] += 1

            if "quality_metrics" in r:
                model_data[model_key]["quality_scores"].append(r["quality_metrics"].get("overall_score", 0))

            if "generation_time_ms" in meta:
                model_data[model_key]["generation_times"].append(meta["generation_time_ms"])

            if "attempt_number" in meta:
                model_data[model_key]["attempts"].append(meta["attempt_number"])

        # Calculate aggregates
        results = {}
        for model, data in model_data.items():
            results[model] = {
                "total_reviews": data["count"],
                "avg_quality_score": sum(data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0,
                "avg_generation_time_ms": sum(data["generation_times"]) / len(data["generation_times"]) if data["generation_times"] else 0,
                "avg_attempts": sum(data["attempts"]) / len(data["attempts"]) if data["attempts"] else 0
            }

        return results

    def identify_failure_modes(self, reviews: list[dict]) -> dict:
        """Identify common failure modes and rejection reasons."""
        rejection_reasons = {}

        for r in reviews:
            if "quality_metrics" in r:
                for reason in r["quality_metrics"].get("rejection_reasons", []):
                    # Extract category from reason
                    if reason.startswith("["):
                        category = reason.split("]")[0] + "]"
                    else:
                        category = "General"

                    if category not in rejection_reasons:
                        rejection_reasons[category] = {"count": 0, "examples": []}

                    rejection_reasons[category]["count"] += 1
                    if len(rejection_reasons[category]["examples"]) < 3:
                        rejection_reasons[category]["examples"].append(reason)

        return rejection_reasons

    def generate_markdown_report(self, reviews: Optional[list[dict]] = None) -> str:
        """
        Generate a comprehensive markdown quality report.

        Args:
            reviews: Optional list of reviews (loads from file if not provided)

        Returns:
            Markdown formatted report string
        """
        if reviews is None:
            reviews = self.load_synthetic_reviews()

        stats = self.calculate_dataset_statistics(reviews)
        model_perf = self.calculate_model_performance(reviews)
        failure_modes = self.identify_failure_modes(reviews)

        # Try to get comparison data
        try:
            comparison = self.comparator.compare()
            has_comparison = True
        except Exception as e:
            logger.warning(f"Could not generate comparison: {e}")
            comparison = None
            has_comparison = False

        # Build report sections
        sections = []

        # Header
        sections.append(f"""# Synthetic Review Quality Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
""")

        # Overview
        sections.append(f"""## 1. Overview

| Metric | Value |
|--------|-------|
| Total Reviews | {stats['total_reviews']} |
| Average Quality Score | {stats['quality_scores']['overall']['mean']:.3f} |
| Total Generation Time | {stats['generation']['total_time_s']:.1f}s |
| Avg Generation Time | {stats['generation']['avg_time_ms']:.0f}ms |

""")

        # Dataset Statistics
        sections.append("""## 2. Dataset Statistics

### Rating Distribution

| Rating | Count | Percentage |
|--------|-------|------------|
""")

        total = stats['total_reviews']
        for rating in sorted(stats['distribution']['ratings'].keys(), reverse=True):
            count = stats['distribution']['ratings'][rating]
            pct = count / total * 100 if total > 0 else 0
            sections.append(f"| {rating} stars | {count} | {pct:.1f}% |\n")

        sections.append("""
### Product Distribution

| Product | Count |
|---------|-------|
""")

        for product, count in sorted(stats['distribution']['products'].items(), key=lambda x: -x[1]):
            sections.append(f"| {product} | {count} |\n")

        sections.append("""
### Persona Distribution

| Persona | Count |
|---------|-------|
""")

        for persona, count in sorted(stats['distribution']['personas'].items(), key=lambda x: -x[1]):
            sections.append(f"| {persona} | {count} |\n")

        # Guardrail Metrics
        sections.append(f"""
## 3. Guardrail Metrics

### Quality Scores

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Diversity | {stats['quality_scores']['diversity']['mean']:.3f} | {stats['quality_scores']['diversity']['min']:.3f} | {stats['quality_scores']['diversity']['max']:.3f} |
| Realism | {stats['quality_scores']['realism']['mean']:.3f} | {stats['quality_scores']['realism']['min']:.3f} | {stats['quality_scores']['realism']['max']:.3f} |
| Bias | {stats['quality_scores']['bias']['mean']:.3f} | {stats['quality_scores']['bias']['min']:.3f} | {stats['quality_scores']['bias']['max']:.3f} |
| **Overall** | **{stats['quality_scores']['overall']['mean']:.3f}** | {stats['quality_scores']['overall']['min']:.3f} | {stats['quality_scores']['overall']['max']:.3f} |

""")

        # Synthetic vs Real Comparison
        if has_comparison and comparison:
            sections.append(f"""## 4. Synthetic vs Real Comparison

### Dataset Sizes

- Real Reviews: {comparison['real']['basic']['total_reviews']}
- Synthetic Reviews: {comparison['synthetic']['basic']['total_reviews']}

### Quality Similarity Scores

| Metric | Score |
|--------|-------|
| Length Similarity | {comparison['quality_scores']['length_similarity']:.2%} |
| Vocabulary Similarity | {comparison['quality_scores']['vocabulary_similarity']:.2%} |
| Sentiment Similarity | {comparison['quality_scores']['sentiment_similarity']:.2%} |
| Rating Distribution | {comparison['quality_scores']['rating_distribution_similarity']:.2%} |
| **Overall Similarity** | **{comparison['quality_scores']['overall']:.2%}** |

### Key Metrics Comparison

| Metric | Real | Synthetic |
|--------|------|-----------|
| Avg Word Count | {comparison['real']['basic']['word_count']['mean']:.1f} | {comparison['synthetic']['basic']['word_count']['mean']:.1f} |
| Type-Token Ratio | {comparison['real']['vocabulary']['type_token_ratio']:.3f} | {comparison['synthetic']['vocabulary']['type_token_ratio']:.3f} |
| Positive Sentiment | {comparison['real']['sentiment']['positive_ratio']:.1%} | {comparison['synthetic']['sentiment']['positive_ratio']:.1%} |
| Negative Sentiment | {comparison['real']['sentiment']['negative_ratio']:.1%} | {comparison['synthetic']['sentiment']['negative_ratio']:.1%} |

""")
            # Strengths and Weaknesses
            if comparison['summary'].get('strengths') or comparison['summary'].get('weaknesses'):
                sections.append("### Analysis\n\n**Strengths:**\n")
                for s in comparison['summary'].get('strengths', []):
                    sections.append(f"- {s}\n")
                sections.append("\n**Weaknesses:**\n")
                for w in comparison['summary'].get('weaknesses', []):
                    sections.append(f"- {w}\n")
                sections.append("\n")

        # Model Performance
        sections.append("""## 5. Model Performance

| Model | Reviews | Avg Quality | Avg Time (ms) | Avg Attempts |
|-------|---------|-------------|---------------|--------------|
""")

        for model, perf in sorted(model_perf.items(), key=lambda x: -x[1]['avg_quality_score']):
            sections.append(
                f"| {model} | {perf['total_reviews']} | {perf['avg_quality_score']:.3f} | "
                f"{perf['avg_generation_time_ms']:.0f} | {perf['avg_attempts']:.1f} |\n"
            )

        # Failure Modes
        if failure_modes:
            sections.append("""
## 6. Failure Modes

| Category | Count | Example |
|----------|-------|---------|
""")
            for category, data in sorted(failure_modes.items(), key=lambda x: -x[1]['count']):
                example = data['examples'][0] if data['examples'] else "N/A"
                # Truncate long examples
                if len(example) > 60:
                    example = example[:57] + "..."
                sections.append(f"| {category} | {data['count']} | {example} |\n")

        # Limitations
        sections.append("""
## 7. Limitations

- **Embedding Analysis:** Requires `sentence-transformers` package for semantic similarity metrics
- **Sentiment Analysis:** Falls back to rule-based analysis if `vaderSentiment` is not installed
- **Local Models:** Ollama models may have variable latency based on hardware
- **Sample Size:** Quality metrics are more reliable with larger datasets (300+ reviews)
- **Domain Specificity:** Quality thresholds are tuned for SaaS developer tools domain

---

*Report generated by Synthetic Review Data Generator v1.0.0*
""")

        return "".join(sections)

    def save_report(self, filename: str = "quality_report.md") -> Path:
        """
        Generate and save the quality report.

        Args:
            filename: Output filename

        Returns:
            Path to saved report
        """
        report = self.generate_markdown_report()
        output_path = self.reports_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Report saved to: {output_path}")
        return output_path

    def save_report_as_pdf(self, filename: str = "quality_report.pdf") -> Path:
        """
        Generate and save the quality report as PDF.

        Args:
            filename: Output filename

        Returns:
            Path to saved PDF report

        Raises:
            ImportError: If fpdf2 is not installed
        """
        if not HAS_FPDF:
            raise ImportError(
                "PDF generation requires fpdf2. Install it with: pip install fpdf2"
            )

        reviews = self.load_synthetic_reviews()
        stats = self.calculate_dataset_statistics(reviews)
        model_perf = self.calculate_model_performance(reviews)
        failure_modes = self.identify_failure_modes(reviews)

        try:
            comparison = self.comparator.compare()
            has_comparison = True
        except Exception as e:
            logger.warning(f"Could not generate comparison: {e}")
            comparison = None
            has_comparison = False

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 10, "Synthetic Review Quality Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(10)

        # Section 1: Overview
        self._pdf_section_header(pdf, "1. Overview")
        self._pdf_table(pdf, [
            ["Metric", "Value"],
            ["Total Reviews", str(stats['total_reviews'])],
            ["Average Quality Score", f"{stats['quality_scores']['overall']['mean']:.3f}"],
            ["Total Generation Time", f"{stats['generation']['total_time_s']:.1f}s"],
            ["Avg Generation Time", f"{stats['generation']['avg_time_ms']:.0f}ms"],
        ])
        pdf.ln(5)

        # Section 2: Rating Distribution
        self._pdf_section_header(pdf, "2. Dataset Statistics")
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Rating Distribution", ln=True)

        total = stats['total_reviews']
        rating_rows = [["Rating", "Count", "Percentage"]]
        for rating in sorted(stats['distribution']['ratings'].keys(), reverse=True):
            count = stats['distribution']['ratings'][rating]
            pct = count / total * 100 if total > 0 else 0
            rating_rows.append([f"{rating} stars", str(count), f"{pct:.1f}%"])
        self._pdf_table(pdf, rating_rows)
        pdf.ln(5)

        # Product Distribution
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Product Distribution", ln=True)
        product_rows = [["Product", "Count"]]
        for product, count in sorted(stats['distribution']['products'].items(), key=lambda x: -x[1])[:10]:
            product_rows.append([product[:40], str(count)])
        self._pdf_table(pdf, product_rows)
        pdf.ln(5)

        # Section 3: Quality Scores
        self._pdf_section_header(pdf, "3. Guardrail Metrics")
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Quality Scores", ln=True)
        self._pdf_table(pdf, [
            ["Metric", "Mean", "Min", "Max"],
            ["Diversity", f"{stats['quality_scores']['diversity']['mean']:.3f}",
             f"{stats['quality_scores']['diversity']['min']:.3f}",
             f"{stats['quality_scores']['diversity']['max']:.3f}"],
            ["Realism", f"{stats['quality_scores']['realism']['mean']:.3f}",
             f"{stats['quality_scores']['realism']['min']:.3f}",
             f"{stats['quality_scores']['realism']['max']:.3f}"],
            ["Bias", f"{stats['quality_scores']['bias']['mean']:.3f}",
             f"{stats['quality_scores']['bias']['min']:.3f}",
             f"{stats['quality_scores']['bias']['max']:.3f}"],
            ["Overall", f"{stats['quality_scores']['overall']['mean']:.3f}",
             f"{stats['quality_scores']['overall']['min']:.3f}",
             f"{stats['quality_scores']['overall']['max']:.3f}"],
        ])
        pdf.ln(5)

        # Section 4: Comparison (if available)
        if has_comparison and comparison:
            self._pdf_section_header(pdf, "4. Synthetic vs Real Comparison")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Real Reviews: {comparison['real']['basic']['total_reviews']}", ln=True)
            pdf.cell(0, 6, f"Synthetic Reviews: {comparison['synthetic']['basic']['total_reviews']}", ln=True)
            pdf.ln(3)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Quality Similarity Scores", ln=True)
            self._pdf_table(pdf, [
                ["Metric", "Score"],
                ["Length Similarity", f"{comparison['quality_scores']['length_similarity']:.1%}"],
                ["Vocabulary Similarity", f"{comparison['quality_scores']['vocabulary_similarity']:.1%}"],
                ["Sentiment Similarity", f"{comparison['quality_scores']['sentiment_similarity']:.1%}"],
                ["Rating Distribution", f"{comparison['quality_scores']['rating_distribution_similarity']:.1%}"],
                ["Overall Similarity", f"{comparison['quality_scores']['overall']:.1%}"],
            ])
            pdf.ln(5)

        # Section 5: Model Performance
        self._pdf_section_header(pdf, "5. Model Performance")
        model_rows = [["Model", "Reviews", "Avg Quality", "Avg Time (ms)", "Avg Attempts"]]
        for model, perf in sorted(model_perf.items(), key=lambda x: -x[1]['avg_quality_score']):
            model_rows.append([
                model[:30],
                str(perf['total_reviews']),
                f"{perf['avg_quality_score']:.3f}",
                f"{perf['avg_generation_time_ms']:.0f}",
                f"{perf['avg_attempts']:.1f}"
            ])
        self._pdf_table(pdf, model_rows)
        pdf.ln(5)

        # Section 6: Failure Modes
        if failure_modes:
            self._pdf_section_header(pdf, "6. Failure Modes")
            failure_rows = [["Category", "Count"]]
            for category, data in sorted(failure_modes.items(), key=lambda x: -x[1]['count'])[:10]:
                failure_rows.append([category[:30], str(data['count'])])
            self._pdf_table(pdf, failure_rows)

        # Footer
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "Report generated by Synthetic Review Data Generator v1.0.0", ln=True, align="C")

        output_path = self.reports_dir / filename
        pdf.output(str(output_path))
        logger.info(f"PDF report saved to: {output_path}")
        return output_path

    def _pdf_section_header(self, pdf: "FPDF", title: str) -> None:
        """Add a section header to the PDF."""
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, title, ln=True, fill=True)
        pdf.ln(3)

    def _pdf_table(self, pdf: "FPDF", data: list[list[str]]) -> None:
        """Add a table to the PDF."""
        if not data:
            return

        pdf.set_font("Helvetica", "", 9)
        col_count = len(data[0])
        page_width = pdf.w - 2 * pdf.l_margin
        col_width = page_width / col_count

        for i, row in enumerate(data):
            if i == 0:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(220, 220, 220)
            else:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_fill_color(255, 255, 255)

            for cell in row:
                pdf.cell(col_width, 7, str(cell)[:35], border=1, fill=True)
            pdf.ln()

    def generate_json_report(self) -> dict:
        """Generate report data as JSON-serializable dict."""
        reviews = self.load_synthetic_reviews()
        stats = self.calculate_dataset_statistics(reviews)
        model_perf = self.calculate_model_performance(reviews)

        try:
            comparison = self.comparator.compare()
        except Exception:
            comparison = None

        return {
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "model_performance": model_perf,
            "comparison": comparison
        }


def generate_report(
    config_dir: str = "config",
    data_dir: str = "data",
    reports_dir: str = "reports",
    output_filename: str = "quality_report.md"
) -> str:
    """
    High-level function to generate a quality report.

    Args:
        config_dir: Path to configuration directory
        data_dir: Path to data directory
        reports_dir: Path to reports output directory
        output_filename: Name of the output file

    Returns:
        Path to the generated report
    """
    generator = ReportGenerator(config_dir, data_dir, reports_dir)
    return str(generator.save_report(output_filename))


def generate_report_pdf(
    config_dir: str = "config",
    data_dir: str = "data",
    reports_dir: str = "reports",
    output_filename: str = "quality_report.pdf"
) -> str:
    """
    High-level function to generate a PDF quality report.

    Args:
        config_dir: Path to configuration directory
        data_dir: Path to data directory
        reports_dir: Path to reports output directory
        output_filename: Name of the output PDF file

    Returns:
        Path to the generated PDF report

    Raises:
        ImportError: If fpdf2 is not installed
    """
    generator = ReportGenerator(config_dir, data_dir, reports_dir)
    return str(generator.save_report_as_pdf(output_filename))
