# Synthetic Review Data Generator

A comprehensive tool for generating high-quality synthetic reviews for SaaS Developer Tools with configurable personas, multiple LLM providers, and quality guardrails.

## Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic (Claude), Google Gemini, and Ollama (local models)
- **LangChain Integration**: Unified LLM abstraction layer with consistent interface across providers
- **OpenLIT Observability**: Token tracking and monitoring for all LLM calls
- **Configurable Personas**: 8 distinct developer personas with unique tones and priorities
- **Quality Guardrails**: Diversity, bias, and realism checks with automatic rejection/regeneration
- **Multiple Interfaces**: CLI, FastAPI REST API, and Gradio Web GUI
- **Comparison Analysis**: Compare synthetic reviews against real baseline data
- **Comprehensive Reporting**: Markdown quality reports with statistics

## Project Structure

```
easygenerator/
├── config/
│   ├── personas.yaml       # Reviewer personas configuration
│   ├── generation.yaml     # Generation settings and models
│   └── quality.yaml        # Quality thresholds and domain keywords
├── data/
│   ├── real_reviews/
│   │   └── real_reviews.json    # Baseline real reviews (40 samples)
│   └── synthetic/
│       └── synthetic_reviews.jsonl  # Generated synthetic reviews
├── src/
│   ├── __init__.py
│   ├── generate.py         # Core generation module
│   ├── compare.py          # Dataset comparison analysis
│   ├── report.py           # Quality report generator
│   ├── cli.py              # Command-line interface
│   ├── llm/                # LLM provider abstraction layer
│   │   ├── __init__.py
│   │   ├── base.py         # Base provider class and interfaces
│   │   ├── openai_provider.py    # OpenAI GPT models
│   │   ├── anthropic_provider.py # Anthropic Claude models
│   │   ├── gemini_provider.py    # Google Gemini models
│   │   └── ollama_provider.py    # Local Ollama models
│   └── quality/
│       ├── __init__.py
│       ├── diversity.py    # Lexical and semantic diversity metrics
│       ├── bias.py         # Sentiment and pattern bias detection
│       ├── realism.py      # Domain authenticity validation
│       └── rejection.py    # Quality evaluation and rejection logic
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI application
├── gui/
│   ├── __init__.py
│   └── app.py              # Gradio web interface
├── reports/
│   └── quality_report.md   # Generated quality reports
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd easygenerator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys for your chosen providers:
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic (Claude)
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-key"

# On Windows use 'set' instead of 'export'
```

5. (Optional) Install Ollama for local models:
- Follow instructions at https://ollama.ai
- Pull models: `ollama pull mistral` or `ollama pull llama3.2`

## Usage

### Command Line Interface

```bash
# Generate reviews
python -m src.cli generate --max-samples 100

# Generate a single review for testing
python -m src.cli single --evaluate

# Evaluate existing dataset
python -m src.cli evaluate --dataset data/synthetic/synthetic_reviews.jsonl

# Generate quality report
python -m src.cli report

# Compare synthetic vs real datasets
python -m src.cli compare --table
```

### FastAPI REST API

Start the API server:
```bash
python -m api.main
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints (all under `/api/v1`):
- `GET /docs` - Interactive API documentation
- `POST /api/v1/generate/single` - Generate single review
- `POST /api/v1/generate/batch` - Start batch generation
- `GET /api/v1/generate/status/{task_id}` - Check batch status
- `POST /api/v1/evaluate` - Evaluate review quality
- `GET /api/v1/compare` - Compare datasets
- `GET /api/v1/report` - Get quality report
- `GET /api/v1/stats` - Get statistics

### Gradio Web Interface

Start the GUI:
```bash
python -m gui.app
# Then open http://localhost:7860 in your browser
```

The GUI provides:
- Single and batch review generation
- Quality evaluation
- Dataset comparison
- Report generation
- Dataset management

## Configuration

### Personas (config/personas.yaml)

Configure reviewer personas with:
- Experience level
- Priorities (what they care about)
- Tone (casual, technical, concise, etc.)

### Generation (config/generation.yaml)

Configure:
- Number of samples
- Rating distribution
- Review length limits
- Products and their features
- LLM models and providers

### Quality (config/quality.yaml)

Configure quality thresholds:
- Diversity: vocabulary overlap, semantic similarity
- Bias: sentiment ratios, phrase repetition limits
- Realism: domain term ratios, specificity requirements

## Quality Guardrails

### Diversity Metrics
- Jaccard overlap on unigrams/bigrams
- Semantic similarity using sentence embeddings
- Unique word ratio checks

### Bias Detection
- Sentiment skew analysis
- Rating-sentiment correlation
- Phrase pattern detection

### Realism Validation
- Domain keyword density
- Marketing language detection
- Specificity scoring

## Hardware Requirements

- **CPU-only**: Works but slower for embeddings
- **GPU recommended**: For faster sentence-transformers
- **Memory**: 4GB+ recommended for embedding models
- **Ollama**: Requires additional 4-8GB for local models

## Supported Models

### OpenAI
- GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo
- o1, o1-mini, o1-preview (reasoning models)

### Anthropic (Claude)
- Claude Opus 4, Claude Sonnet 4
- Claude 3.5 Sonnet, Claude 3.5 Haiku
- Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku

### Google Gemini
- Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash

### Ollama (Local)
- Llama 3.2, Llama 3.1, Llama 3, Mistral, Mixtral
- Gemma 2, Phi-3, Qwen 2.5, CodeLlama, and more

## Engineering Trade-offs

| Area | Decision | Trade-off |
|------|----------|-----------|
| LLM Abstraction | LangChain | Unified interface, slight overhead |
| Observability | OpenLIT | Token tracking, minimal performance impact |
| Embeddings | sentence-transformers | CPU-heavy, high quality |
| Local Model | Ollama | No API costs, requires local resources |
| Regeneration | Max 3 retries | Time vs quality |
| Sentiment | VADER + fallback | Less nuanced, faster |

## API vs GUI

- **FastAPI**: For programmatic access, integration, batch processing
- **Gradio**: For interactive exploration, demos, quick testing
- Gradio directly uses Python modules (not FastAPI)

## Sample Output

```json
{
  "id": "review_00001",
  "review_text": "We've been using CloudBuildX for about 8 months...",
  "metadata": {
    "persona_name": "DevOps Engineer",
    "product_name": "CloudBuildX",
    "rating": 4,
    "model_provider": "anthropic",
    "model_name": "claude-3-5-sonnet-20241022",
    "generation_time_ms": 1234.5
  },
  "quality_metrics": {
    "diversity_score": 0.82,
    "realism_score": 0.76,
    "bias_score": 0.91,
    "overall_score": 0.83,
    "accepted": true
  }
}
```

## License

MIT License
