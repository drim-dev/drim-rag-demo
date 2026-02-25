"""Gradio UI for Python Docs Assistant — indexing, search, and comparison."""

import os

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from download_data import download_docs, DATA_DIR
from index import STRATEGIES, run_indexing
from retrieval import Retriever, RetrievalConfig, RetrievedChunk

load_dotenv()

EXAMPLE_QUERIES = [
    "Как читать файл построчно?",
    "В чём разница между os.path и pathlib?",
    "asyncio.gather timeout",
    "GIL",
    "Как использовать dataclasses?",
    "logging basicConfig формат",
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[RetrievedChunk]):
    """Stream an answer from GPT-4o using retrieved context."""
    if not chunks:
        yield "Не найдено релевантных документов. Попробуйте другой запрос."
        return

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Источник {i}: {chunk.module} — {chunk.section_path}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "Ты — помощник по документации Python. Отвечай на русском языке, "
        "опираясь ТОЛЬКО на предоставленный контекст из официальной документации Python. "
        "Если в контексте нет ответа, честно скажи об этом. "
        "Приводи примеры кода, когда это уместно."
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    accumulated = ""
    for token in llm.stream([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
    ]):
        accumulated += token.content
        yield accumulated


# ---------------------------------------------------------------------------
# Tab 1: Indexing
# ---------------------------------------------------------------------------

def handle_download():
    download_docs()
    file_count = 0
    total_size = 0
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".rst"):
                file_count += 1
                total_size += os.path.getsize(os.path.join(root, f))
    return f"Скачано {file_count} файлов, общий размер: {total_size / 1024:.1f} КБ"


def handle_index(strategy: str):
    stats = run_indexing(strategy)
    if stats["chunks"] == 0:
        return "Ошибка: документы не найдены. Сначала скачайте документацию."
    lines = [f"Секций: {stats['sections']}", f"Чанков: {stats['chunks']}"]
    for dt, count in stats.get("by_doc_type", {}).items():
        lines.append(f"  {dt}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2: Search
# ---------------------------------------------------------------------------

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_modules():
    try:
        modules = get_retriever().get_available_modules()
        return gr.update(choices=modules, value=None)
    except Exception:
        return gr.update(choices=[], value=None)


def get_doc_types():
    try:
        types = get_retriever().get_available_doc_types()
        return gr.update(choices=types, value=None)
    except Exception:
        return gr.update(choices=[], value=None)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"**{i}. {c.module}** — {c.section_path}")
    return "\n".join(lines) if lines else "Нет источников"


def format_chunks_detail(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"### Чанк {i}\n"
            f"**Метод:** {c.retrieval_method} | "
            f"**Релевантность:** {c.score:.4f} | "
            f"**Модуль:** {c.module} | "
            f"**Тип:** {c.doc_type}\n"
            f"**Секция:** {c.section_path}\n\n"
            f"```\n{c.text[:800]}{'...' if len(c.text) > 800 else ''}\n```"
        )
    return "\n\n---\n\n".join(parts) if parts else "Нет результатов"


def handle_search(
    query, use_vector, use_bm25, use_hybrid, use_hyde, use_reranking,
    use_parent_child, filter_module, filter_doc_type, top_k,
):
    if not query.strip():
        yield "", "Введите запрос", "Нет результатов"
        return

    config = RetrievalConfig(
        use_vector=use_vector,
        use_bm25=use_bm25,
        use_hybrid=use_hybrid,
        use_hyde=use_hyde,
        use_reranking=use_reranking,
        use_parent_child=use_parent_child,
        filter_module=filter_module if filter_module else None,
        filter_doc_type=filter_doc_type if filter_doc_type else None,
        top_k=int(top_k),
    )

    retriever = get_retriever()
    chunks = retriever.retrieve(query, config)

    sources = format_sources(chunks)
    details = format_chunks_detail(chunks)

    for answer_so_far in generate_answer(query, chunks):
        yield answer_so_far, sources, details


# ---------------------------------------------------------------------------
# Tab 3: Comparison
# ---------------------------------------------------------------------------

def handle_compare(query):
    if not query.strip():
        return "Введите запрос", "", "", "", ""

    retriever = get_retriever()

    naive_config = RetrievalConfig(use_vector=True, top_k=5)
    naive_chunks = retriever.retrieve(query, naive_config)

    advanced_config = RetrievalConfig(
        use_hybrid=True,
        use_reranking=True,
        use_parent_child=True,
        top_k=5,
    )
    advanced_chunks = retriever.retrieve(query, advanced_config)

    naive_answer = ""
    for part in generate_answer(query, naive_chunks):
        naive_answer = part
    advanced_answer = ""
    for part in generate_answer(query, advanced_chunks):
        advanced_answer = part

    naive_texts = {c.text[:200] for c in naive_chunks}
    advanced_texts = {c.text[:200] for c in advanced_chunks}
    overlap = len(naive_texts & advanced_texts)

    naive_avg = sum(c.score for c in naive_chunks) / len(naive_chunks) if naive_chunks else 0
    advanced_avg = sum(c.score for c in advanced_chunks) / len(advanced_chunks) if advanced_chunks else 0

    summary = (
        f"| Метрика | Naive (vector) | Advanced (hybrid+rerank) |\n"
        f"|---------|----------------|-------------------------|\n"
        f"| Чанков | {len(naive_chunks)} | {len(advanced_chunks)} |\n"
        f"| Средняя релевантность | {naive_avg:.4f} | {advanced_avg:.4f} |\n"
        f"| Длина ответа | {len(naive_answer)} символов | {len(advanced_answer)} символов |\n"
        f"| Пересечение чанков | {overlap} из {max(len(naive_chunks), len(advanced_chunks))} |"
    )

    return (
        naive_answer,
        format_chunks_detail(naive_chunks),
        advanced_answer,
        format_chunks_detail(advanced_chunks),
        summary,
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Python Docs Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🐍 Python Docs Assistant\nRAG-ассистент по документации Python")

        # --- Tab 1: Indexing ---
        with gr.Tab("Индексация"):
            gr.Markdown("### Шаг 1: Скачайте документацию")
            download_btn = gr.Button("Скачать документацию Python", variant="primary")
            download_status = gr.Textbox(label="Статус загрузки", interactive=False)
            download_btn.click(fn=handle_download, outputs=download_status)

            gr.Markdown("### Шаг 2: Постройте индекс")
            strategy_selector = gr.Radio(
                choices=list(STRATEGIES.keys()),
                value="header-based",
                label="Стратегия чанкинга",
            )
            index_btn = gr.Button("Построить индекс", variant="primary")
            index_status = gr.Textbox(label="Статистика индекса", interactive=False, lines=5)
            index_btn.click(fn=handle_index, inputs=strategy_selector, outputs=index_status)

        # --- Tab 2: Search ---
        with gr.Tab("Поиск"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Ваш вопрос",
                        placeholder="Например: Как читать файл построчно?",
                        lines=2,
                    )
                    examples = gr.Examples(
                        examples=[[q] for q in EXAMPLE_QUERIES],
                        inputs=query_input,
                        label="Примеры запросов",
                    )
                with gr.Column(scale=2):
                    gr.Markdown("**Методы поиска:**")
                    use_vector = gr.Checkbox(label="Vector search", value=True)
                    use_bm25 = gr.Checkbox(label="BM25", value=False)
                    use_hybrid = gr.Checkbox(label="Hybrid (RRF)", value=False)
                    use_hyde = gr.Checkbox(label="HyDE", value=False)
                    use_reranking = gr.Checkbox(label="Реранкинг (Cohere)", value=False)
                    use_parent_child = gr.Checkbox(label="Parent-child retrieval", value=False)

                    gr.Markdown("**Фильтры:**")
                    filter_module = gr.Dropdown(
                        choices=[],
                        label="Модуль",
                        allow_custom_value=True,
                        interactive=True,
                    )
                    filter_doc_type = gr.Dropdown(
                        choices=[],
                        label="Тип документа",
                        allow_custom_value=True,
                        interactive=True,
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Top-K результатов",
                    )

            search_btn = gr.Button("Спросить", variant="primary", size="lg")

            answer_output = gr.Markdown(label="Ответ")
            with gr.Accordion("Источники", open=True):
                sources_output = gr.Markdown()
            with gr.Accordion("Найденные чанки", open=False):
                chunks_output = gr.Markdown()

            search_btn.click(
                fn=handle_search,
                inputs=[
                    query_input, use_vector, use_bm25, use_hybrid, use_hyde,
                    use_reranking, use_parent_child, filter_module, filter_doc_type, top_k,
                ],
                outputs=[answer_output, sources_output, chunks_output],
            )

            # Populate filter dropdowns on tab load
            app.load(fn=get_modules, outputs=filter_module)
            app.load(fn=get_doc_types, outputs=filter_doc_type)

        # --- Tab 3: Comparison ---
        with gr.Tab("Сравнение"):
            gr.Markdown(
                "### Сравнение naive vs advanced retrieval\n"
                "Один и тот же запрос проходит через два пайплайна:\n"
                "- **Naive**: только vector search\n"
                "- **Advanced**: hybrid (RRF) + реранкинг + parent-child"
            )
            compare_query = gr.Textbox(
                label="Запрос для сравнения",
                placeholder="Например: В чём разница между os.path и pathlib?",
            )
            compare_examples = gr.Examples(
                examples=[[q] for q in EXAMPLE_QUERIES],
                inputs=compare_query,
                label="Примеры",
            )
            compare_btn = gr.Button("Сравнить", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Naive (vector only)")
                    naive_answer = gr.Markdown()
                    with gr.Accordion("Чанки (naive)", open=False):
                        naive_chunks = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### Advanced (hybrid + rerank)")
                    advanced_answer = gr.Markdown()
                    with gr.Accordion("Чанки (advanced)", open=False):
                        advanced_chunks = gr.Markdown()

            summary_table = gr.Markdown(label="Сводная таблица")

            compare_btn.click(
                fn=handle_compare,
                inputs=compare_query,
                outputs=[naive_answer, naive_chunks, advanced_answer, advanced_chunks, summary_table],
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
