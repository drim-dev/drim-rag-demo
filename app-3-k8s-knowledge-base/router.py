"""LLM-based query router that classifies questions to appropriate indices."""

import json
import time

from llama_index.llms.anthropic import Anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL

ROUTING_PROMPT = """You are a query router for a Kubernetes knowledge base with three indices:

1. **docs** — Kubernetes documentation: concepts, tasks, tutorials (What is X? How to configure Y?)
2. **code** — Kubernetes Go source code: functions, types, interfaces (How is X implemented? Show the scheduler code)
3. **api** — Kubernetes API spec: endpoints, request/response schemas (What fields does Pod spec have? API for creating Deployments)

Given a user query, decide which indices to search. Return JSON with:
- "indices": list of index names to search (one or more of: "docs", "code", "api")
- "label": short classification label (e.g., "docs", "code", "api", "docs+code", "docs+api", "all")
- "confidence": float 0-1

Examples:
- "What is a Pod?" → {"indices": ["docs"], "label": "docs", "confidence": 0.95}
- "How is the scheduler loop implemented?" → {"indices": ["code"], "label": "code", "confidence": 0.9}
- "What fields does the Pod spec have?" → {"indices": ["api"], "label": "api", "confidence": 0.9}
- "How does kubectl apply work?" → {"indices": ["docs", "code", "api"], "label": "all", "confidence": 0.85}
- "Difference between Deployment and StatefulSet" → {"indices": ["docs"], "label": "docs", "confidence": 0.9}
- "Show the ReplicaSet controller code" → {"indices": ["code"], "label": "code", "confidence": 0.95}
- "How to set resource limits via API?" → {"indices": ["docs", "api"], "label": "docs+api", "confidence": 0.85}

Return ONLY valid JSON, no other text.

Query: {query}"""


def route_query(query: str) -> dict:
    """Classify a query and return routing decision with timing info.

    Returns dict with keys: indices, label, confidence, latency_ms
    """
    llm = Anthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=200)

    start = time.time()
    response = llm.complete(ROUTING_PROMPT.replace("{query}", query))
    latency_ms = (time.time() - start) * 1000

    fallback = {
        "indices": ["docs"],
        "label": "docs",
        "confidence": 0.5,
        "latency_ms": latency_ms,
    }

    try:
        result = json.loads(response.text.strip())
    except json.JSONDecodeError:
        return fallback

    if not isinstance(result, dict) or "indices" not in result or not result["indices"]:
        return fallback

    result.setdefault("label", "docs")
    result.setdefault("confidence", 0.5)
    result["latency_ms"] = latency_ms
    return result
