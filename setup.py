"""
Setup script for Synthetic Review Data Generator.
"""

from setuptools import setup, find_packages

setup(
    name="syntheticgen",
    version="1.0.0",
    description="Synthetic Review Data Generator with Quality Guardrails",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
        "httpx>=0.25.0",
        "openai>=1.0.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.25.0",
        "gradio>=4.0.0",
    ],
    extras_require={
        "full": [
            "vaderSentiment>=3.3.2",
            "sentence-transformers>=2.2.0",
            "numpy>=1.24.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "syntheticgen=src.cli:main",
        ],
    },
)
