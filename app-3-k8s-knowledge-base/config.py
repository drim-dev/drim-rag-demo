"""Shared configuration for the K8s Knowledge Base RAG app."""

import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
LLM_MODEL = "claude-sonnet-4-20250514"

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "k8s_rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

POSTGRES_CONNECTION_STRING = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Cost per token (Claude Sonnet pricing)
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000

# Semantic cache similarity threshold
CACHE_SIMILARITY_THRESHOLD = 0.95

# Index collection names
DOCS_INDEX = "docs_index"
CODE_INDEX = "code_index"
API_SPECS_INDEX = "api_specs_index"
