"""Indexing pipeline: parse RST docs, chunk, embed, store in Qdrant + BM25."""

import argparse
import os
import pickle
import uuid

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "python-docs")
BM25_PATH = os.path.join(os.path.dirname(__file__), "data", "bm25_index.pkl")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "data", "chunks.pkl")

COLLECTION_NAME = "python_docs"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIM = 1536


# ---------------------------------------------------------------------------
# RST parsing
# ---------------------------------------------------------------------------

RST_HEADER_CHARS = "=#-~^`"


def _is_header_underline(line: str) -> bool:
    """Check if a line is an RST section underline (e.g., '=====' or '-----')."""
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    return all(c == stripped[0] for c in stripped) and stripped[0] in RST_HEADER_CHARS


def _detect_doc_type(filepath: str) -> str:
    if "/tutorial/" in filepath:
        return "tutorial"
    return "api-reference"


def _detect_module_name(filepath: str) -> str:
    """Extract module name from file path (e.g., 'library/asyncio.rst' -> 'asyncio')."""
    basename = os.path.basename(filepath)
    name = basename.replace(".rst", "")
    if "/tutorial/" in filepath:
        return f"tutorial/{name}"
    return name


def parse_rst_sections(filepath: str) -> list[Document]:
    """Parse an RST file into sections, preserving header hierarchy."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    module_name = _detect_module_name(filepath)
    doc_type = _detect_doc_type(filepath)
    sections: list[Document] = []
    header_stack: list[str] = []
    current_text_lines: list[str] = []
    # Track header char hierarchy within this file
    header_levels: list[str] = []

    def _flush_section():
        text = "\n".join(current_text_lines).strip()
        if text and len(text) > 50:
            section_path = " > ".join(header_stack) if header_stack else module_name
            sections.append(Document(
                page_content=text,
                metadata={
                    "module": module_name,
                    "doc_type": doc_type,
                    "section_path": section_path,
                    "source_file": os.path.basename(filepath),
                },
            ))

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect RST header: text line followed by underline of same length
        if (
            i + 1 < len(lines)
            and line.strip()
            and _is_header_underline(lines[i + 1])
            and len(lines[i + 1].strip()) >= len(line.strip())
        ):
            _flush_section()
            current_text_lines = []

            header_text = line.strip()
            underline_char = lines[i + 1].strip()[0]

            if underline_char not in header_levels:
                header_levels.append(underline_char)
            level = header_levels.index(underline_char)

            header_stack = header_stack[:level] + [header_text]
            # Include the header in the section text for context
            current_text_lines.append(header_text)
            i += 2
            continue

        # Also handle overline + title + underline pattern
        if (
            _is_header_underline(line)
            and i + 2 < len(lines)
            and lines[i + 1].strip()
            and _is_header_underline(lines[i + 2])
        ):
            _flush_section()
            current_text_lines = []

            header_text = lines[i + 1].strip()
            underline_char = line.strip()[0]

            if underline_char not in header_levels:
                header_levels.append(underline_char)
            level = header_levels.index(underline_char)

            header_stack = header_stack[:level] + [header_text]
            current_text_lines.append(header_text)
            i += 3
            continue

        current_text_lines.append(line.rstrip())
        i += 1

    _flush_section()
    return sections


def load_all_documents() -> list[Document]:
    """Load and parse all RST files from the data directory."""
    all_docs: list[Document] = []

    for root, _, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            if not fname.endswith(".rst"):
                continue
            filepath = os.path.join(root, fname)
            sections = parse_rst_sections(filepath)
            all_docs.extend(sections)

    return all_docs


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_fixed(documents: list[Document]) -> list[Document]:
    """Recursive character splitter — baseline strategy."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for text in splits:
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "chunk_strategy": "fixed"},
            ))
    return chunks


def chunk_header_based(documents: list[Document]) -> list[Document]:
    """Use RST sections directly — each parsed section becomes a chunk.

    Oversized sections are split further while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        if len(doc.page_content) <= 1500:
            chunks.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "chunk_strategy": "header-based"},
            ))
        else:
            splits = splitter.split_text(doc.page_content)
            for i, text in enumerate(splits):
                chunks.append(Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "header-based",
                        "section_path": f"{doc.metadata['section_path']} (part {i + 1})",
                    },
                ))
    return chunks


def chunk_semantic(documents: list[Document]) -> list[Document]:
    """Group text by semantic similarity using LangChain SemanticChunker."""
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    chunks = []
    for doc in documents:
        if len(doc.page_content) < 100:
            continue
        try:
            splits = chunker.split_text(doc.page_content)
        except Exception:
            # Fall back to fixed splitting if semantic chunking fails on short text
            splits = [doc.page_content]
        for text in splits:
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "chunk_strategy": "semantic"},
            ))
    return chunks


STRATEGIES = {
    "fixed": chunk_fixed,
    "header-based": chunk_header_based,
    "semantic": chunk_semantic,
}


# ---------------------------------------------------------------------------
# Parent-child mapping
# ---------------------------------------------------------------------------

def build_parent_child_mapping(chunks: list[Document]) -> tuple[list[Document], dict[str, str]]:
    """Assign IDs and build parent-child relationships.

    Parent = the original section document. Children = the chunks derived from it.
    For header-based strategy, the section IS the chunk (self-parent).
    For other strategies, we group by (source_file, section_path).
    """
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    # Group chunks by their source section
    section_groups: dict[str, list[Document]] = {}
    for chunk in chunks:
        key = f"{chunk.metadata['source_file']}::{chunk.metadata['section_path']}"
        section_groups.setdefault(key, []).append(chunk)

    parent_map: dict[str, str] = {}
    for group in section_groups.values():
        if len(group) == 1:
            parent_map[group[0].metadata["chunk_id"]] = group[0].metadata["chunk_id"]
        else:
            parent_id = group[0].metadata["chunk_id"]
            # First chunk in a section acts as parent
            for chunk in group:
                parent_map[chunk.metadata["chunk_id"]] = parent_id

    return chunks, parent_map


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------

def embed_and_store(chunks: list[Document], parent_map: dict[str, str]) -> None:
    """Embed chunks with Cohere and store in Qdrant."""
    embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)

    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    BATCH_SIZE = 64
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c.page_content for c in batch]
        vectors = embeddings.embed_documents(texts)

        points = []
        for chunk, vector in zip(batch, vectors):
            chunk_id = chunk.metadata["chunk_id"]
            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "module": chunk.metadata["module"],
                    "doc_type": chunk.metadata["doc_type"],
                    "section_path": chunk.metadata["section_path"],
                    "source_file": chunk.metadata["source_file"],
                    "chunk_strategy": chunk.metadata["chunk_strategy"],
                    "chunk_id": chunk_id,
                    "parent_id": parent_map.get(chunk_id, chunk_id),
                },
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Indexed {min(start + BATCH_SIZE, total)}/{total} chunks")


def build_bm25_index(chunks: list[Document]) -> None:
    """Build BM25 index and persist it alongside chunk data."""
    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    chunk_data = []
    for c in chunks:
        chunk_data.append({
            "text": c.page_content,
            "metadata": c.metadata,
        })

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunk_data}, f)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_data, f)

    print(f"  BM25 index saved to {BM25_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_indexing(strategy: str) -> dict:
    """Run the full indexing pipeline. Returns stats dict."""
    print(f"\n1. Loading RST documents from {DATA_DIR}...")
    documents = load_all_documents()
    print(f"   Found {len(documents)} sections")

    if not documents:
        print("ERROR: No documents found. Run download_data.py first.")
        return {"sections": 0, "chunks": 0}

    print(f"\n2. Chunking with '{strategy}' strategy...")
    chunk_fn = STRATEGIES[strategy]
    chunks = chunk_fn(documents)
    print(f"   Created {len(chunks)} chunks")

    print("\n3. Building parent-child mapping...")
    chunks, parent_map = build_parent_child_mapping(chunks)

    print("\n4. Embedding and storing in Qdrant...")
    embed_and_store(chunks, parent_map)

    print("\n5. Building BM25 index...")
    build_bm25_index(chunks)

    doc_types = {}
    for c in chunks:
        dt = c.metadata["doc_type"]
        doc_types[dt] = doc_types.get(dt, 0) + 1

    print(f"\nDone! Indexed {len(chunks)} chunks:")
    for dt, count in sorted(doc_types.items()):
        print(f"  {dt}: {count}")

    return {
        "sections": len(documents),
        "chunks": len(chunks),
        "by_doc_type": doc_types,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Python documentation for RAG")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="header-based",
        help="Chunking strategy (default: header-based)",
    )
    args = parser.parse_args()
    run_indexing(args.strategy)
