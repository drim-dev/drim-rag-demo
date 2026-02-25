"""End-to-end RAG pipeline: cache check → routing → search → generation → metrics."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import text

from config import (
    ANTHROPIC_API_KEY,
    API_SPECS_INDEX,
    CACHE_SIMILARITY_THRESHOLD,
    CODE_INDEX,
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
    DOCS_INDEX,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_MODEL,
    MIN_RELEVANCE_SCORE,
    OLLAMA_HOST,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from db import Metric, SemanticCache, get_session, init_db
from router import route_query

SYNTHESIS_PROMPT = PromptTemplate(
    "You are a Kubernetes expert assistant. Answer the user's question using ONLY "
    "the provided context from multiple sources. Cite your sources using labels "
    "[docs], [code], or [api] after each claim.\n\n"
    "If the context is insufficient, say so honestly — do not make up information.\n\n"
    "Context:\n"
    "-----\n"
    "{context_str}\n"
    "-----\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

INDEX_NAME_MAP = {
    "docs": DOCS_INDEX,
    "code": CODE_INDEX,
    "api": API_SPECS_INDEX,
}

SOURCE_LABELS = {
    "docs": "📄 Документация",
    "code": "💻 Исходный код",
    "api": "🔌 API спецификация",
}


def _get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def _get_vector_store(collection_name: str) -> PGVectorStore:
    return PGVectorStore.from_params(
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        password=POSTGRES_PASSWORD,
        port=str(POSTGRES_PORT),
        user=POSTGRES_USER,
        table_name=collection_name,
        embed_dim=EMBEDDING_DIM,
    )


def _check_cache(query_embedding: list[float]) -> dict | None:
    """Check semantic cache for a similar query. Returns cached response or None."""
    session = get_session()
    try:
        result = session.execute(
            text(
                "SELECT id, original_query, response_text, sources_json, route_label, "
                "1 - (query_embedding <=> CAST(:embedding AS vector)) as similarity "
                "FROM semantic_cache "
                "ORDER BY query_embedding <=> CAST(:embedding AS vector) "
                "LIMIT 1"
            ),
            {"embedding": str(query_embedding)},
        ).fetchone()

        if result and result.similarity >= CACHE_SIMILARITY_THRESHOLD:
            return {
                "response_text": result.response_text,
                "sources_json": result.sources_json,
                "route_label": result.route_label,
                "similarity": result.similarity,
            }
        return None
    finally:
        session.close()


def _store_in_cache(
    query_embedding: list[float],
    original_query: str,
    response_text: str,
    sources_json: str,
    route_label: str,
):
    session = get_session()
    try:
        session.add(
            SemanticCache(
                query_embedding=query_embedding,
                original_query=original_query,
                response_text=response_text,
                sources_json=sources_json,
                route_label=route_label,
            )
        )
        session.commit()
    finally:
        session.close()


def _search_index(index_name: str, query: str, embed_model, top_k: int = 5) -> list[dict]:
    """Search a single pgvector index, returning scored results with metadata."""
    try:
        vector_store = _get_vector_store(index_name)
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        return [
            {
                "text": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
                "source_type": node.node.metadata.get("source_type", "unknown"),
            }
            for node in nodes
            if node.score is not None and node.score >= MIN_RELEVANCE_SCORE
        ]
    except Exception as e:
        print(f"Search error on {index_name}: {e}")
        return []


def _search_multiple_indices(
    indices: list[str], query: str, embed_model, top_k: int = 5
) -> list[dict]:
    """Search multiple indices in parallel and merge results."""
    all_results = []

    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        futures = {
            executor.submit(
                _search_index, INDEX_NAME_MAP[idx], query, embed_model, top_k
            ): idx
            for idx in indices
            if idx in INDEX_NAME_MAP
        }
        for future in futures:
            results = future.result()
            all_results.extend(results)

    all_results.sort(key=lambda x: x["score"] or 0, reverse=True)
    return all_results


def _build_context(results: list[dict]) -> str:
    """Build context string from search results with source type labels."""
    parts = []
    for i, r in enumerate(results, 1):
        source_type = r["source_type"]
        label = SOURCE_LABELS.get(source_type, source_type)
        meta_parts = []
        for key in ("page_title", "section", "symbol_name", "package", "path"):
            if key in r["metadata"] and r["metadata"][key]:
                meta_parts.append(f"{key}={r['metadata'][key]}")
        meta_str = ", ".join(meta_parts)
        parts.append(f"[Source {i} — {label}] ({meta_str})\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def _record_metrics(
    query_id: str,
    latency_embedding_ms: float,
    latency_routing_ms: float,
    latency_search_ms: float,
    latency_generation_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
):
    total = latency_embedding_ms + latency_routing_ms + latency_search_ms + latency_generation_ms
    cost = input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN

    session = get_session()
    try:
        session.add(
            Metric(
                query_id=query_id,
                latency_embedding_ms=latency_embedding_ms,
                latency_routing_ms=latency_routing_ms,
                latency_search_ms=latency_search_ms,
                latency_generation_ms=latency_generation_ms,
                total_latency_ms=total,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=cost,
                cache_hit=cache_hit,
            )
        )
        session.commit()
    finally:
        session.close()


def query_pipeline(query: str, top_k: int = 5, use_cache: bool = True) -> dict:
    """Run the full RAG pipeline: embed → cache check → route → search → generate.

    Returns dict with: query_id, response_text, sources, route, metrics, cache_hit
    """
    init_db()
    query_id = str(uuid.uuid4())[:8]
    embed_model = _get_embed_model()

    # Embed the query
    t0 = time.time()
    query_embedding = embed_model.get_query_embedding(query)
    latency_embedding_ms = (time.time() - t0) * 1000

    # Check semantic cache
    if use_cache:
        cached = _check_cache(query_embedding)
        if cached:
            _record_metrics(
                query_id=query_id,
                latency_embedding_ms=latency_embedding_ms,
                latency_routing_ms=0,
                latency_search_ms=0,
                latency_generation_ms=0,
                input_tokens=0,
                output_tokens=0,
                cache_hit=True,
            )
            return {
                "query_id": query_id,
                "response_text": cached["response_text"],
                "sources": json.loads(cached["sources_json"]) if cached["sources_json"] else [],
                "route": {"label": cached["route_label"], "confidence": 1.0, "indices": []},
                "metrics": {
                    "latency_embedding_ms": latency_embedding_ms,
                    "latency_routing_ms": 0,
                    "latency_search_ms": 0,
                    "latency_generation_ms": 0,
                    "total_latency_ms": latency_embedding_ms,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost": 0,
                },
                "cache_hit": True,
            }

    # Route query
    route = route_query(query)
    latency_routing_ms = route["latency_ms"]

    # Search indices
    t0 = time.time()
    results = _search_multiple_indices(route["indices"], query, embed_model, top_k)
    latency_search_ms = (time.time() - t0) * 1000

    context = _build_context(results[:top_k])

    # Generate response
    llm = Anthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=2048)

    t0 = time.time()
    prompt_text = SYNTHESIS_PROMPT.format(context_str=context, query_str=query)
    response = llm.complete(prompt_text)
    latency_generation_ms = (time.time() - t0) * 1000

    response_text = response.text

    input_tokens = getattr(getattr(response.raw, "usage", None), "input_tokens", 0) if hasattr(response, "raw") and response.raw else 0
    output_tokens = getattr(getattr(response.raw, "usage", None), "output_tokens", 0) if hasattr(response, "raw") and response.raw else 0

    sources_for_display = [
        {
            "text": r["text"][:300],
            "score": r["score"],
            "source_type": r["source_type"],
            "metadata": r["metadata"],
        }
        for r in results[:top_k]
    ]

    # Store in cache
    if use_cache:
        _store_in_cache(
            query_embedding=query_embedding,
            original_query=query,
            response_text=response_text,
            sources_json=json.dumps(sources_for_display, default=str),
            route_label=route["label"],
        )

    _record_metrics(
        query_id=query_id,
        latency_embedding_ms=latency_embedding_ms,
        latency_routing_ms=latency_routing_ms,
        latency_search_ms=latency_search_ms,
        latency_generation_ms=latency_generation_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    return {
        "query_id": query_id,
        "response_text": response_text,
        "sources": sources_for_display,
        "route": {
            "label": route["label"],
            "confidence": route["confidence"],
            "indices": route["indices"],
        },
        "metrics": {
            "latency_embedding_ms": latency_embedding_ms,
            "latency_routing_ms": latency_routing_ms,
            "latency_search_ms": latency_search_ms,
            "latency_generation_ms": latency_generation_ms,
            "total_latency_ms": (
                latency_embedding_ms + latency_routing_ms
                + latency_search_ms + latency_generation_ms
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": (
                input_tokens * COST_PER_INPUT_TOKEN
                + output_tokens * COST_PER_OUTPUT_TOKEN
            ),
        },
        "cache_hit": False,
    }


def query_pipeline_streaming(query: str, top_k: int = 5, use_cache: bool = True):
    """Streaming version of the pipeline. Yields (token, metadata_or_none) tuples.

    The last yield has metadata dict as second element.
    """
    init_db()
    query_id = str(uuid.uuid4())[:8]
    embed_model = _get_embed_model()

    t0 = time.time()
    query_embedding = embed_model.get_query_embedding(query)
    latency_embedding_ms = (time.time() - t0) * 1000

    if use_cache:
        cached = _check_cache(query_embedding)
        if cached:
            _record_metrics(
                query_id=query_id,
                latency_embedding_ms=latency_embedding_ms,
                latency_routing_ms=0, latency_search_ms=0,
                latency_generation_ms=0,
                input_tokens=0, output_tokens=0, cache_hit=True,
            )
            yield cached["response_text"], {
                "query_id": query_id,
                "sources": json.loads(cached["sources_json"]) if cached["sources_json"] else [],
                "route": {"label": cached["route_label"], "confidence": 1.0, "indices": []},
                "metrics": {
                    "latency_embedding_ms": latency_embedding_ms,
                    "latency_routing_ms": 0,
                    "latency_search_ms": 0,
                    "latency_generation_ms": 0,
                    "total_latency_ms": latency_embedding_ms,
                    "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0,
                },
                "cache_hit": True,
            }
            return

    route = route_query(query)
    latency_routing_ms = route["latency_ms"]

    t0 = time.time()
    results = _search_multiple_indices(route["indices"], query, embed_model, top_k)
    latency_search_ms = (time.time() - t0) * 1000

    context = _build_context(results[:top_k])

    llm = Anthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=2048)
    prompt_text = SYNTHESIS_PROMPT.format(context_str=context, query_str=query)

    t0 = time.time()
    response = llm.complete(prompt_text)
    latency_generation_ms = (time.time() - t0) * 1000
    full_response = response.text
    yield full_response, None

    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "raw") and response.raw:
        usage = getattr(response.raw, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

    sources_for_display = [
        {
            "text": r["text"][:300],
            "score": r["score"],
            "source_type": r["source_type"],
            "metadata": r["metadata"],
        }
        for r in results[:top_k]
    ]

    if use_cache:
        _store_in_cache(
            query_embedding=query_embedding,
            original_query=query,
            response_text=full_response,
            sources_json=json.dumps(sources_for_display, default=str),
            route_label=route["label"],
        )

    _record_metrics(
        query_id=query_id,
        latency_embedding_ms=latency_embedding_ms,
        latency_routing_ms=latency_routing_ms,
        latency_search_ms=latency_search_ms,
        latency_generation_ms=latency_generation_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    yield "", {
        "query_id": query_id,
        "sources": sources_for_display,
        "route": {
            "label": route["label"],
            "confidence": route["confidence"],
            "indices": route["indices"],
        },
        "metrics": {
            "latency_embedding_ms": latency_embedding_ms,
            "latency_routing_ms": latency_routing_ms,
            "latency_search_ms": latency_search_ms,
            "latency_generation_ms": latency_generation_ms,
            "total_latency_ms": (
                latency_embedding_ms + latency_routing_ms
                + latency_search_ms + latency_generation_ms
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": (
                input_tokens * COST_PER_INPUT_TOKEN
                + output_tokens * COST_PER_OUTPUT_TOKEN
            ),
        },
        "cache_hit": False,
    }
