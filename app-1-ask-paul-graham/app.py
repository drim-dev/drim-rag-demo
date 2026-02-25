"""Ask Paul Graham — RAG practice app with Streamlit UI."""

import os
import time
from pathlib import Path

import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

BASE_DIR = Path(__file__).parent
ESSAYS_DIR = BASE_DIR / "data" / "essays"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "paul_graham"

QA_PROMPT_TEMPLATE = PromptTemplate(
    "You are an AI assistant that answers questions about Paul Graham's essays. "
    "Use ONLY the context provided below to answer. "
    "If the context does not contain enough information to answer, say so honestly — "
    "do not make up information.\n\n"
    "When answering, cite which essay(s) your answer comes from.\n\n"
    "Context:\n"
    "-----\n"
    "{context_str}\n"
    "-----\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def count_essays() -> int:
    if not ESSAYS_DIR.exists():
        return 0
    return len(list(ESSAYS_DIR.glob("*.txt")))


def build_index(chunk_size: int, chunk_overlap: int) -> dict:
    """Build a Qdrant-backed vector index from downloaded essays.

    Returns stats dict with chunk_count, embedding_dim, and elapsed_seconds.
    """
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    documents = SimpleDirectoryReader(str(ESSAYS_DIR)).load_data()

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    client = qdrant_client.QdrantClient(url=QDRANT_URL)

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    vector_store = QdrantVectorStore(
        collection_name=QDRANT_COLLECTION,
        client=client,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    start = time.time()
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    elapsed = time.time() - start

    embedding_dim = 1536  # text-embedding-3-small default

    return {
        "index": index,
        "chunk_count": len(nodes),
        "embedding_dim": embedding_dim,
        "elapsed_seconds": elapsed,
    }


def load_existing_index() -> VectorStoreIndex | None:
    """Load index from existing Qdrant collection."""
    try:
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        client = qdrant_client.QdrantClient(url=QDRANT_URL)

        if not client.collection_exists(QDRANT_COLLECTION):
            return None

        collection_info = client.get_collection(QDRANT_COLLECTION)
        if collection_info.points_count == 0:
            return None

        vector_store = QdrantVectorStore(
            collection_name=QDRANT_COLLECTION,
            client=client,
        )
        return VectorStoreIndex.from_vector_store(vector_store)
    except Exception:
        return None


def query_index(index: VectorStoreIndex, question: str, top_k: int) -> dict:
    """Query the index and return answer + source chunks."""
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=QA_PROMPT_TEMPLATE,
        streaming=True,
    )

    response = query_engine.query(question)

    source_nodes = []
    for node in response.source_nodes:
        text = node.node.get_content()
        title = node.node.metadata.get("file_name", "Unknown")
        title = title.replace(".txt", "").replace("-", " ").title()
        source_nodes.append({
            "title": title,
            "score": node.score,
            "text": text,
        })

    return {
        "response": response,
        "sources": source_nodes,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ask Paul Graham",
    page_icon=":books:",
    layout="wide",
)

st.title("Ask Paul Graham")
st.caption("RAG-приложение для поиска по эссе Пола Грэма")

# -- Sidebar --

st.sidebar.header("Данные и индекс")

essay_count = count_essays()
st.sidebar.metric("Эссе на диске", essay_count)

if st.sidebar.button("Скачать эссе"):
    with st.sidebar:
        with st.spinner("Скачивание эссе..."):
            from download_data import download_essays
            download_essays()
        st.success("Готово!")
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Настройки индекса")

chunk_size = st.sidebar.selectbox("Chunk size", [256, 512, 1024], index=1)
overlap_pct = st.sidebar.selectbox("Chunk overlap", ["0%", "10%", "20%"], index=1)
overlap_map = {"0%": 0, "10%": 0.1, "20%": 0.2}
chunk_overlap = int(chunk_size * overlap_map[overlap_pct])

if st.sidebar.button("Построить индекс", disabled=(essay_count == 0)):
    if essay_count == 0:
        st.sidebar.error("Сначала скачайте эссе.")
    else:
        with st.sidebar:
            with st.spinner("Построение индекса..."):
                stats = build_index(chunk_size, chunk_overlap)
                st.session_state["index"] = stats["index"]
                st.session_state["index_stats"] = {
                    "chunk_count": stats["chunk_count"],
                    "embedding_dim": stats["embedding_dim"],
                    "elapsed_seconds": stats["elapsed_seconds"],
                }
        st.sidebar.success("Индекс построен!")

if "index_stats" in st.session_state:
    s = st.session_state["index_stats"]
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Чанков", s["chunk_count"])
    col2.metric("Размерность", s["embedding_dim"])
    st.sidebar.caption(f"Построен за {s['elapsed_seconds']:.1f} сек.")

# Try to load existing index if none in session
if "index" not in st.session_state:
    existing = load_existing_index()
    if existing:
        st.session_state["index"] = existing

st.sidebar.divider()
st.sidebar.subheader("Настройки поиска")
top_k = st.sidebar.slider("Top-K (количество фрагментов)", 1, 10, 5)

# -- Main area --

question = st.text_input(
    "Задайте вопрос Полу Грэму",
    placeholder="What is the most important thing for a startup?",
)

ask_disabled = "index" not in st.session_state or not question.strip()

if st.button("Спросить", disabled=ask_disabled, type="primary"):
    if "index" not in st.session_state:
        st.error("Сначала постройте индекс (кнопка в боковой панели).")
    else:
        result = query_index(st.session_state["index"], question.strip(), top_k)

        st.subheader("Ответ")
        answer_placeholder = st.empty()
        full_answer = ""
        for text in result["response"].response_gen:
            full_answer += text
            answer_placeholder.markdown(full_answer)

        if result["sources"]:
            with st.expander(f"Найденные фрагменты ({len(result['sources'])})"):
                for i, src in enumerate(result["sources"], 1):
                    score_pct = (src["score"] or 0) * 100
                    st.markdown(f"**{i}. {src['title']}** — релевантность: {score_pct:.1f}%")
                    st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                    if i < len(result["sources"]):
                        st.divider()

elif "index" not in st.session_state:
    st.info("Постройте индекс в боковой панели, чтобы начать задавать вопросы.")
