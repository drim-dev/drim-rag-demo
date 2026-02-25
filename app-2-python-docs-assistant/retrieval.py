"""Retrieval methods: vector, BM25, hybrid RRF, HyDE, reranking, parent-child."""

import os
import pickle
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

COLLECTION_NAME = "python_docs"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "embed-v4.0"
BM25_PATH = os.path.join(os.path.dirname(__file__), "data", "bm25_index.pkl")


@dataclass
class RetrievedChunk:
    text: str
    score: float
    module: str
    doc_type: str
    section_path: str
    source_file: str
    chunk_id: str
    parent_id: str
    retrieval_method: str


@dataclass
class RetrievalConfig:
    use_vector: bool = True
    use_bm25: bool = False
    use_hybrid: bool = False
    use_hyde: bool = False
    use_reranking: bool = False
    use_parent_child: bool = False
    filter_module: str | None = None
    filter_doc_type: str | None = None
    top_k: int = 5


class Retriever:
    def __init__(self):
        self._qdrant = QdrantClient(url=QDRANT_URL)
        self._embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)
        self._bm25_data: dict | None = None

    def _load_bm25(self) -> dict:
        if self._bm25_data is None:
            with open(BM25_PATH, "rb") as f:
                self._bm25_data = pickle.load(f)
        return self._bm25_data

    def _build_qdrant_filter(self, config: RetrievalConfig) -> Filter | None:
        conditions = []
        if config.filter_module:
            conditions.append(
                FieldCondition(key="module", match=MatchValue(value=config.filter_module))
            )
        if config.filter_doc_type:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=config.filter_doc_type))
            )
        if conditions:
            return Filter(must=conditions)
        return None

    def _payload_to_chunk(self, payload: dict, score: float, method: str) -> RetrievedChunk:
        return RetrievedChunk(
            text=payload["text"],
            score=score,
            module=payload.get("module", ""),
            doc_type=payload.get("doc_type", ""),
            section_path=payload.get("section_path", ""),
            source_file=payload.get("source_file", ""),
            chunk_id=payload.get("chunk_id", ""),
            parent_id=payload.get("parent_id", ""),
            retrieval_method=method,
        )

    # ----- Vector search -----

    def vector_search(self, query: str, config: RetrievalConfig) -> list[RetrievedChunk]:
        query_vector = self._embeddings.embed_query(query)
        results = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=self._build_qdrant_filter(config),
            limit=config.top_k,
            with_payload=True,
        )
        return [
            self._payload_to_chunk(point.payload, point.score, "vector")
            for point in results.points
        ]

    # ----- BM25 search -----

    def bm25_search(self, query: str, config: RetrievalConfig) -> list[RetrievedChunk]:
        data = self._load_bm25()
        bm25 = data["bm25"]
        chunks = data["chunks"]

        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)

        scored_chunks = list(zip(range(len(chunks)), scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_chunks:
            if score <= 0:
                break
            chunk = chunks[idx]
            meta = chunk["metadata"]

            if config.filter_module and meta.get("module") != config.filter_module:
                continue
            if config.filter_doc_type and meta.get("doc_type") != config.filter_doc_type:
                continue

            results.append(RetrievedChunk(
                text=chunk["text"],
                score=float(score),
                module=meta.get("module", ""),
                doc_type=meta.get("doc_type", ""),
                section_path=meta.get("section_path", ""),
                source_file=meta.get("source_file", ""),
                chunk_id=meta.get("chunk_id", ""),
                parent_id=meta.get("parent_id", ""),
                retrieval_method="bm25",
            ))
            if len(results) >= config.top_k:
                break

        return results

    # ----- Hybrid search (Reciprocal Rank Fusion) -----

    def hybrid_search(self, query: str, config: RetrievalConfig) -> list[RetrievedChunk]:
        """Combine vector + BM25 results using Reciprocal Rank Fusion."""
        RRF_K = 60

        expanded_config = RetrievalConfig(
            top_k=config.top_k * 2,
            filter_module=config.filter_module,
            filter_doc_type=config.filter_doc_type,
        )

        vector_results = self.vector_search(query, expanded_config)
        bm25_results = self.bm25_search(query, expanded_config)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results):
            key = chunk.chunk_id or chunk.text[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
            chunk_map[key] = chunk

        for rank, chunk in enumerate(bm25_results):
            key = chunk.chunk_id or chunk.text[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in sorted_keys[: config.top_k]:
            chunk = chunk_map[key]
            chunk.score = rrf_scores[key]
            chunk.retrieval_method = "hybrid-rrf"
            results.append(chunk)

        return results

    # ----- HyDE (Hypothetical Document Embeddings) -----

    def hyde_search(self, query: str, config: RetrievalConfig) -> list[RetrievedChunk]:
        """Generate a hypothetical answer, embed it, and use that for vector search."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        hypothetical = llm.invoke(
            f"Write a short technical paragraph that would answer this Python documentation "
            f"question. Write as if you are the Python docs:\n\n{query}"
        ).content

        hyde_vector = self._embeddings.embed_query(str(hypothetical))
        results = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=hyde_vector,
            query_filter=self._build_qdrant_filter(config),
            limit=config.top_k,
            with_payload=True,
        )
        return [
            self._payload_to_chunk(point.payload, point.score, "hyde")
            for point in results.points
        ]

    # ----- Reranking -----

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rerank chunks using Cohere rerank API."""
        if not chunks:
            return chunks

        reranker = CohereRerank(model="rerank-v3.5", top_n=top_k)
        from langchain_core.documents import Document

        docs = [Document(page_content=c.text) for c in chunks]
        reranked = reranker.compress_documents(docs, query)

        result = []
        for rdoc in reranked:
            original_idx = next(
                i for i, c in enumerate(chunks) if c.text == rdoc.page_content
            )
            chunk = chunks[original_idx]
            chunk.score = rdoc.metadata.get("relevance_score", chunk.score)
            chunk.retrieval_method += "+reranked"
            result.append(chunk)

        return result

    # ----- Parent-child retrieval -----

    def expand_to_parents(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Replace child chunks with their parent chunks for broader context."""
        parent_ids = {c.parent_id for c in chunks if c.parent_id != c.chunk_id}
        if not parent_ids:
            return chunks

        parent_texts: dict[str, RetrievedChunk] = {}
        for pid in parent_ids:
            results = self._qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=self._embeddings.embed_query(""),  # Dummy — we scroll by ID
                limit=1,
                with_payload=True,
            )
            # Use scroll to find by ID instead
            scroll_results = self._qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="chunk_id", match=MatchValue(value=pid))]
                ),
                limit=1,
                with_payload=True,
            )
            points, _ = scroll_results
            if points:
                p = points[0]
                parent_texts[pid] = self._payload_to_chunk(
                    p.payload, 1.0, "parent"
                )

        result = []
        seen_parents: set[str] = set()
        for chunk in chunks:
            if chunk.parent_id != chunk.chunk_id and chunk.parent_id in parent_texts:
                if chunk.parent_id not in seen_parents:
                    parent = parent_texts[chunk.parent_id]
                    parent.score = chunk.score
                    parent.retrieval_method = chunk.retrieval_method + "+parent"
                    result.append(parent)
                    seen_parents.add(chunk.parent_id)
            else:
                result.append(chunk)

        return result

    # ----- Main retrieve method -----

    def retrieve(self, query: str, config: RetrievalConfig) -> list[RetrievedChunk]:
        """Run the configured retrieval pipeline."""
        if config.use_hybrid:
            chunks = self.hybrid_search(query, config)
        elif config.use_hyde:
            chunks = self.hyde_search(query, config)
        elif config.use_bm25 and not config.use_vector:
            chunks = self.bm25_search(query, config)
        elif config.use_vector:
            chunks = self.vector_search(query, config)
        else:
            chunks = self.vector_search(query, config)

        if config.use_reranking and chunks:
            chunks = self.rerank(query, chunks, config.top_k)

        if config.use_parent_child and chunks:
            chunks = self.expand_to_parents(chunks)

        return chunks[: config.top_k]

    def get_available_modules(self) -> list[str]:
        """Get list of unique module names from the index."""
        scroll_results, _ = self._qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=["module"],
        )
        modules = sorted({p.payload["module"] for p in scroll_results if p.payload.get("module")})
        return modules

    def get_available_doc_types(self) -> list[str]:
        """Get list of unique doc types from the index."""
        scroll_results, _ = self._qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=["doc_type"],
        )
        types = sorted({p.payload["doc_type"] for p in scroll_results if p.payload.get("doc_type")})
        return types
