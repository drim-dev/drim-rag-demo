"""Multi-source indexing pipeline for Kubernetes documentation, code, and API specs.

Usage:
    python index.py --source docs          # Index only docs
    python index.py --source code          # Index only code
    python index.py --source api           # Index only API specs
    python index.py --source all           # Index everything
    python index.py --source all --force   # Force reindex (ignore hashes)
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import tree_sitter_go as tsgo
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from tree_sitter import Language, Parser

from config import (
    API_SPECS_INDEX,
    CODE_INDEX,
    DOCS_INDEX,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from db import ContentHash, get_session, init_db

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "k8s-docs"
CODE_DIR = BASE_DIR / "data" / "k8s-code"
API_DIR = BASE_DIR / "data" / "k8s-api-specs"

GO_LANGUAGE = Language(tsgo.language())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_changed_files(source_type: str, files: list[Path], force: bool) -> list[Path]:
    """Return only files whose content has changed since last indexing."""
    if force:
        return files

    session = get_session()
    try:
        existing = {
            h.file_path: h.content_hash
            for h in session.query(ContentHash)
            .filter(ContentHash.source_type == source_type)
            .all()
        }

        changed = []
        for f in files:
            current_hash = file_hash(f)
            if str(f) not in existing or existing[str(f)] != current_hash:
                changed.append(f)

        return changed
    finally:
        session.close()


def update_file_hashes(source_type: str, files: list[Path]):
    session = get_session()
    try:
        for f in files:
            h = file_hash(f)
            existing = (
                session.query(ContentHash)
                .filter(ContentHash.file_path == str(f))
                .first()
            )
            if existing:
                existing.content_hash = h
            else:
                session.add(
                    ContentHash(
                        source_type=source_type,
                        file_path=str(f),
                        content_hash=h,
                    )
                )
        session.commit()
    finally:
        session.close()


def get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )


def get_vector_store(collection_name: str) -> PGVectorStore:
    return PGVectorStore.from_params(
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        password=POSTGRES_PASSWORD,
        port=str(POSTGRES_PORT),
        user=POSTGRES_USER,
        table_name=collection_name,
        embed_dim=EMBEDDING_DIM,
    )


def build_index(documents: list[Document], collection_name: str) -> int:
    """Build a pgvector index from documents. Returns number of nodes created."""
    embed_model = get_embed_model()
    vector_store = get_vector_store(collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[node_parser],
        show_progress=True,
    )
    return len(documents)


# ---------------------------------------------------------------------------
# Docs indexing
# ---------------------------------------------------------------------------

def categorize_doc(filename: str) -> str:
    if "concepts__" in filename:
        return "concept"
    elif "tasks__" in filename:
        return "task"
    elif "tutorials__" in filename:
        return "tutorial"
    return "other"


def extract_title(content: str) -> str:
    """Extract page title from markdown frontmatter or first heading."""
    title_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()
    heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return "Untitled"


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].strip()
    return content


def split_by_headers(content: str) -> list[tuple[str, str]]:
    """Split markdown by headers, returning (section_path, section_text) pairs."""
    lines = content.split("\n")
    sections = []
    current_section = ""
    current_header = "Introduction"

    for line in lines:
        header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if header_match:
            if current_section.strip():
                sections.append((current_header, current_section.strip()))
            current_header = header_match.group(2).strip()
            current_section = line + "\n"
        else:
            current_section += line + "\n"

    if current_section.strip():
        sections.append((current_header, current_section.strip()))

    return sections


def index_docs(force: bool = False) -> dict:
    if not DOCS_DIR.exists():
        print("No docs found. Run: python scripts/download_docs.py")
        return {"files": 0, "chunks": 0}

    files = sorted(DOCS_DIR.glob("*.md"))
    changed = get_changed_files("docs", files, force)

    if not changed:
        print("No docs changes detected. Use --force to reindex.")
        return {"files": 0, "chunks": 0}

    print(f"Indexing {len(changed)} doc files...")
    documents = []

    for f in changed:
        raw = f.read_text()
        title = extract_title(raw)
        body = strip_frontmatter(raw)
        category = categorize_doc(f.name)
        sections = split_by_headers(body)

        for section_header, section_text in sections:
            if len(section_text.strip()) < 50:
                continue
            documents.append(
                Document(
                    text=section_text,
                    metadata={
                        "source_type": "docs",
                        "file_name": f.name,
                        "page_title": title,
                        "section": section_header,
                        "category": category,
                    },
                )
            )

    chunk_count = build_index(documents, DOCS_INDEX)
    update_file_hashes("docs", changed)
    print(f"Indexed {chunk_count} doc chunks from {len(changed)} files")
    return {"files": len(changed), "chunks": chunk_count}


# ---------------------------------------------------------------------------
# Code indexing
# ---------------------------------------------------------------------------

def parse_go_symbols(content: str, file_path: str) -> list[Document]:
    """Extract functions, types, and interfaces from Go source using tree-sitter."""
    parser = Parser(GO_LANGUAGE)
    tree = parser.parse(content.encode())

    package_match = re.search(r"^package\s+(\w+)", content, re.MULTILINE)
    package_name = package_match.group(1) if package_match else "unknown"

    documents = []

    def extract_nodes(node, depth=0):
        if node.type in (
            "function_declaration",
            "method_declaration",
            "type_declaration",
            "type_spec",
        ):
            text = content[node.start_byte:node.end_byte]
            if len(text.strip()) < 20:
                return

            symbol_name = ""
            symbol_type = "unknown"

            if node.type in ("function_declaration", "method_declaration"):
                symbol_type = "func"
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbol_name = content[name_node.start_byte:name_node.end_byte]
            elif node.type in ("type_declaration", "type_spec"):
                symbol_type = "type"
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbol_name = content[name_node.start_byte:name_node.end_byte]
                for child in node.children:
                    if child.type == "interface_type":
                        symbol_type = "interface"
                        break

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "code",
                        "file_path": file_path,
                        "package": package_name,
                        "symbol_name": symbol_name,
                        "symbol_type": symbol_type,
                    },
                )
            )

        for child in node.children:
            extract_nodes(child, depth + 1)

    extract_nodes(tree.root_node)
    return documents


def index_code(force: bool = False) -> dict:
    if not CODE_DIR.exists():
        print("No code found. Run: python scripts/download_code.py")
        return {"files": 0, "chunks": 0}

    files = sorted(CODE_DIR.glob("*.go"))
    changed = get_changed_files("code", files, force)

    if not changed:
        print("No code changes detected. Use --force to reindex.")
        return {"files": 0, "chunks": 0}

    print(f"Indexing {len(changed)} Go source files...")
    documents = []

    for f in changed:
        content = f.read_text()
        original_path = f.stem.replace("__", "/")
        symbols = parse_go_symbols(content, original_path)
        documents.extend(symbols)

    if not documents:
        print("No symbols extracted from code files")
        return {"files": len(changed), "chunks": 0}

    chunk_count = build_index(documents, CODE_INDEX)
    update_file_hashes("code", changed)
    print(f"Indexed {chunk_count} code symbols from {len(changed)} files")
    return {"files": len(changed), "chunks": chunk_count}


# ---------------------------------------------------------------------------
# API specs indexing
# ---------------------------------------------------------------------------

def parse_openapi_spec(spec_path: Path) -> list[Document]:
    """Parse OpenAPI spec into one document per endpoint."""
    with open(spec_path) as f:
        spec = json.load(f)

    documents = []
    paths = spec.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.startswith("x-") or method == "parameters":
                continue

            summary = details.get("summary", "")
            description = details.get("description", "")
            tags = details.get("tags", [])
            operation_id = details.get("operationId", "")

            response_summary = ""
            responses = details.get("responses", {})
            if "200" in responses:
                resp = responses["200"]
                response_summary = resp.get("description", "")

            text = (
                f"API Endpoint: {method.upper()} {path}\n"
                f"Operation: {operation_id}\n"
                f"Summary: {summary}\n"
                f"Description: {description}\n"
            )
            if response_summary:
                text += f"Success Response: {response_summary}\n"

            api_group = tags[0] if tags else "core"
            kind = operation_id.replace("read", "").replace("list", "").replace(
                "create", ""
            ).replace("delete", "").replace("patch", "").replace(
                "replace", ""
            ).replace("Namespaced", "").strip() if operation_id else "unknown"

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "api",
                        "api_group": api_group,
                        "resource_kind": kind,
                        "http_method": method.upper(),
                        "path": path,
                    },
                )
            )

    return documents


def index_api_specs(force: bool = False) -> dict:
    spec_path = API_DIR / "swagger.json"
    if not spec_path.exists():
        print("No API spec found. Run: python scripts/download_api_specs.py")
        return {"files": 0, "chunks": 0}

    changed = get_changed_files("api", [spec_path], force)
    if not changed:
        print("No API spec changes detected. Use --force to reindex.")
        return {"files": 0, "chunks": 0}

    print("Indexing Kubernetes API spec...")
    documents = parse_openapi_spec(spec_path)

    chunk_count = build_index(documents, API_SPECS_INDEX)
    update_file_hashes("api", changed)
    print(f"Indexed {chunk_count} API endpoint chunks")
    return {"files": 1, "chunks": chunk_count}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Index K8s knowledge base sources")
    parser.add_argument(
        "--source",
        choices=["docs", "code", "api", "all"],
        required=True,
        help="Which source to index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex, ignoring content hashes",
    )
    args = parser.parse_args()

    init_db()

    start = time.time()
    results = {}

    if args.source in ("docs", "all"):
        results["docs"] = index_docs(args.force)
    if args.source in ("code", "all"):
        results["code"] = index_code(args.force)
    if args.source in ("api", "all"):
        results["api"] = index_api_specs(args.force)

    elapsed = time.time() - start
    print(f"\nIndexing complete in {elapsed:.1f}s")
    for source, stats in results.items():
        print(f"  {source}: {stats['files']} files, {stats['chunks']} chunks")


if __name__ == "__main__":
    main()
