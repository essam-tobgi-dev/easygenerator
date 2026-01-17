"""
Command-line interface for Synthetic Review Data Generator.

Provides commands for generation, evaluation, and reporting.
"""

import argparse
import json
import sys
from pathlib import Path


def progress_callback(current: int, total: int, failed: int):
    """Display progress bar in terminal."""
    percent = current / total * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({percent:.1f}%) | Failed: {failed}")
    sys.stdout.flush()
    if current == total:
        print()  # New line at completion


def cmd_generate(args):
    """Execute the generate command."""
    from .generate import ReviewGenerator
    from .quality.rejection import QualityEvaluator

    print(f"Initializing generator with config from: {args.config}")

    generator = ReviewGenerator(args.config, args.output)
    quality_evaluator = QualityEvaluator(args.config)

    # Override samples if specified
    num_samples = args.max_samples if args.max_samples else None
    if num_samples is None:
        import yaml
        with open(Path(args.config) / "generation.yaml", "r") as f:
            gen_config = yaml.safe_load(f)
            num_samples = gen_config.get("samples", 100)

    print(f"Generating {num_samples} synthetic reviews...")

    if args.clear:
        generator.clear_output()
        print("Cleared existing output file.")

    reviews = []
    try:
        for review in generator.generate_batch(
            quality_evaluator,
            num_samples,
            progress_callback=progress_callback if not args.quiet else None
        ):
            generator.save_review(review)
            reviews.append(review)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"\nError during generation: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()

    print(f"\nGenerated {len(reviews)} reviews.")

    # Show model statistics
    if not args.quiet:
        stats = generator.get_model_stats()
        print("\nModel Statistics:")
        for model, model_stats in stats.items():
            print(f"  {model}:")
            print(f"    - Successful: {model_stats['successful']}")
            print(f"    - Rejected: {model_stats['rejected']}")
            print(f"    - Avg Time: {model_stats['average_time_ms']:.0f}ms")
            print(f"    - Avg Quality: {model_stats['average_quality_score']:.3f}")


def cmd_evaluate(args):
    """Execute the evaluate command."""
    from .quality.rejection import QualityEvaluator
    import json

    print(f"Evaluating dataset: {args.dataset}")

    evaluator = QualityEvaluator(args.config)

    # Load reviews
    reviews = []
    dataset_path = Path(args.dataset)
    if dataset_path.suffix == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            reviews = data if isinstance(data, list) else data.get("reviews", [])

    print(f"Loaded {len(reviews)} reviews.")

    # Evaluate each review
    existing_texts = []
    results = []

    for i, review in enumerate(reviews):
        text = review.get("review_text", review.get("text", ""))
        rating = review.get("rating", review.get("metadata", {}).get("rating", 3))

        metrics = evaluator.evaluate(text, existing_texts, rating)
        results.append({
            "id": review.get("id", i),
            "accepted": metrics.accepted,
            "overall_score": metrics.overall_score,
            "issues": metrics.rejection_reasons
        })

        if metrics.accepted:
            existing_texts.append(text)

        if not args.quiet and (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(reviews)} reviews...")

    # Summary
    stats = evaluator.get_statistics()
    print(f"\nEvaluation Summary:")
    print(f"  Total Reviews: {stats['total_evaluations']}")
    print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
    print(f"  Average Score: {stats['average_score']:.3f}")
    print(f"  Score Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")

    # Save results if output specified
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def cmd_report(args):
    """Execute the report command."""
    from .report import ReportGenerator

    print("Generating quality report...")

    generator = ReportGenerator(
        config_dir=args.config,
        data_dir=args.data,
        reports_dir=args.output
    )

    if args.format == "json":
        report = generator.generate_json_report()
        output_path = Path(args.output) / "quality_report.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to: {output_path}")
    else:
        output_path = generator.save_report()
        print(f"Markdown report saved to: {output_path}")


def cmd_compare(args):
    """Execute the compare command."""
    from .compare import DatasetComparator

    print("Comparing datasets...")

    comparator = DatasetComparator(
        real_reviews_path=args.real,
        synthetic_reviews_path=args.synthetic
    )

    if args.table:
        print(comparator.generate_comparison_table())
    else:
        comparison = comparator.compare()
        print(json.dumps(comparison, indent=2, default=str))


def cmd_single(args):
    """Generate a single review for testing."""
    from .generate import ReviewGenerator

    generator = ReviewGenerator(args.config, args.output)

    print("Generating single review...")

    try:
        text, metadata, time_ms = generator.generate_single()

        print(f"\n{'='*60}")
        print(f"Product: {metadata.product_name}")
        print(f"Persona: {metadata.persona_name}")
        print(f"Rating: {'*' * metadata.rating}")
        print(f"Model: {metadata.model_provider}:{metadata.model_name}")
        print(f"Time: {time_ms:.0f}ms")
        print(f"{'='*60}")
        print(f"\n{text}\n")

        if args.evaluate:
            from .quality.rejection import QualityEvaluator
            evaluator = QualityEvaluator(args.config)
            metrics = evaluator.evaluate(text, [], metadata.rating)

            print(f"{'='*60}")
            print("Quality Evaluation:")
            print(f"  Diversity: {metrics.diversity_score:.3f}")
            print(f"  Realism: {metrics.realism_score:.3f}")
            print(f"  Bias: {metrics.bias_score:.3f}")
            print(f"  Overall: {metrics.overall_score:.3f}")
            print(f"  Accepted: {metrics.accepted}")
            if metrics.rejection_reasons:
                print("  Issues:")
                for reason in metrics.rejection_reasons:
                    print(f"    - {reason}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="syntheticgen",
        description="Synthetic Review Data Generator with Quality Guardrails"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic reviews"
    )
    gen_parser.add_argument(
        "--config", "-c",
        default="config",
        help="Path to configuration directory (default: config)"
    )
    gen_parser.add_argument(
        "--output", "-o",
        default="data/synthetic",
        help="Path to output directory (default: data/synthetic)"
    )
    gen_parser.add_argument(
        "--max-samples", "-n",
        type=int,
        help="Maximum number of samples to generate"
    )
    gen_parser.add_argument(
        "--model",
        help="Specific model to use (e.g., openai:gpt-4o-mini)"
    )
    gen_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing output before generating"
    )
    gen_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a dataset's quality"
    )
    eval_parser.add_argument(
        "--dataset", "-d",
        default="data/synthetic/synthetic_reviews.jsonl",
        help="Path to dataset file"
    )
    eval_parser.add_argument(
        "--config", "-c",
        default="config",
        help="Path to configuration directory"
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Path to save evaluation results (JSON)"
    )
    eval_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate quality report"
    )
    report_parser.add_argument(
        "--config", "-c",
        default="config",
        help="Path to configuration directory"
    )
    report_parser.add_argument(
        "--data", "-d",
        default="data",
        help="Path to data directory"
    )
    report_parser.add_argument(
        "--output", "-o",
        default="reports",
        help="Path to reports output directory"
    )
    report_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    report_parser.set_defaults(func=cmd_report)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare synthetic and real datasets"
    )
    compare_parser.add_argument(
        "--real", "-r",
        default="data/real_reviews/real_reviews.json",
        help="Path to real reviews file"
    )
    compare_parser.add_argument(
        "--synthetic", "-s",
        default="data/synthetic/synthetic_reviews.jsonl",
        help="Path to synthetic reviews file"
    )
    compare_parser.add_argument(
        "--table", "-t",
        action="store_true",
        help="Output as formatted table"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Single command (for testing)
    single_parser = subparsers.add_parser(
        "single",
        help="Generate a single review for testing"
    )
    single_parser.add_argument(
        "--config", "-c",
        default="config",
        help="Path to configuration directory"
    )
    single_parser.add_argument(
        "--output", "-o",
        default="data/synthetic",
        help="Path to output directory"
    )
    single_parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Evaluate the generated review"
    )
    single_parser.set_defaults(func=cmd_single)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
