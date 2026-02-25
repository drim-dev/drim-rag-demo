"""K8s Knowledge Base — multi-index RAG with evaluation and monitoring."""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from config import EMBEDDING_MODEL, LLM_MODEL
from db import init_db

BASE_DIR = Path(__file__).parent

init_db()

st.set_page_config(
    page_title="K8s Knowledge Base",
    page_icon=":wheel_of_dharma:",
    layout="wide",
)

PAGES = {
    "chat": "💬 Чат",
    "indexing": "📚 Индексация",
    "evaluation": "📊 Оценка",
}


def get_page():
    return st.session_state.get("page", "chat")


def set_page(page):
    st.session_state["page"] = page


# Navigation
cols = st.columns(len(PAGES))
for i, (key, label) in enumerate(PAGES.items()):
    if cols[i].button(label, use_container_width=True, type="primary" if get_page() == key else "secondary"):
        set_page(key)
        st.rerun()

st.divider()


# ---------------------------------------------------------------------------
# Page: Chat
# ---------------------------------------------------------------------------

def page_chat():
    st.title("💬 K8s Knowledge Base")
    st.caption("Мульти-индексный RAG по документации, коду и API Kubernetes")

    with st.sidebar:
        st.subheader("Настройки")
        top_k = st.slider("Top-K результатов", 1, 10, 5, key="top_k")
        use_cache = st.checkbox("Семантический кэш", value=True, key="use_cache")

        st.divider()
        st.header("Примеры запросов")
        examples = [
            "Что такое Pod в Kubernetes?",
            "Чем Deployment отличается от StatefulSet?",
            "Как реализован цикл планировщика?",
            "Какие поля есть в спецификации Pod?",
            "Как работает kubectl apply?",
        ]
        selected_example = None
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                selected_example = ex

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["metadata"]:
                _render_message_metadata(msg["metadata"])

    user_input = st.chat_input("Задайте вопрос о Kubernetes...", key="chat_input_field")

    query = user_input or selected_example
    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            _handle_query(query, top_k, use_cache)


def _handle_query(query: str, top_k: int, use_cache: bool):
    from pipeline import query_pipeline_streaming

    placeholder = st.empty()
    full_response = ""
    metadata = None

    try:
        for token, meta in query_pipeline_streaming(query, top_k=top_k, use_cache=use_cache):
            if meta is not None:
                metadata = meta
                if token:
                    full_response += token
            else:
                full_response += token
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        if metadata:
            _render_message_metadata(metadata)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "metadata": metadata,
            })
            _render_feedback(metadata.get("query_id", ""))

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"Произошла ошибка: {e}",
        })


def _render_message_metadata(metadata: dict):
    route = metadata.get("route", {})
    metrics = metadata.get("metrics", {})
    sources = metadata.get("sources", [])
    cache_hit = metadata.get("cache_hit", False)

    badge_cols = st.columns([1, 1, 1, 2])

    if route.get("label"):
        confidence = route.get("confidence", 0)
        badge_cols[0].markdown(
            f"🎯 **Маршрут:** `{route['label']}` ({confidence:.0%})"
        )

    if cache_hit:
        badge_cols[1].markdown("⚡ **Из кэша**")

    total_ms = metrics.get("total_latency_ms", 0)
    badge_cols[2].markdown(f"⏱️ {total_ms:.0f} мс")

    if sources:
        source_types = set()
        for s in sources:
            stype = s.get("source_type", "")
            if stype == "docs":
                source_types.add("🔵 docs")
            elif stype == "code":
                source_types.add("🟢 code")
            elif stype == "api":
                source_types.add("🟠 API")
        badge_cols[3].markdown(" ".join(source_types))

    with st.expander("📊 Метрики и источники"):
        mcols = st.columns(4)
        mcols[0].metric("Embedding", f"{metrics.get('latency_embedding_ms', 0):.0f} мс")
        mcols[1].metric("Маршрутизация", f"{metrics.get('latency_routing_ms', 0):.0f} мс")
        mcols[2].metric("Поиск", f"{metrics.get('latency_search_ms', 0):.0f} мс")
        mcols[3].metric("Генерация", f"{metrics.get('latency_generation_ms', 0):.0f} мс")

        tcols = st.columns(3)
        tcols[0].metric("Входные токены", metrics.get("input_tokens", 0))
        tcols[1].metric("Выходные токены", metrics.get("output_tokens", 0))
        tcols[2].metric("Стоимость", f"${metrics.get('estimated_cost', 0):.4f}")

        if sources:
            st.subheader("Найденные источники")
            for i, src in enumerate(sources, 1):
                source_type = src.get("source_type", "")
                score = src.get("score", 0) or 0
                icon = {"docs": "🔵", "code": "🟢", "api": "🟠"}.get(source_type, "⚪")
                meta = src.get("metadata", {})
                title = meta.get("page_title", meta.get("symbol_name", meta.get("path", "—")))
                st.markdown(f"**{icon} {i}. {title}** — релевантность: {score:.2f}")
                st.code(src.get("text", "")[:400], language=None)


def _render_feedback(query_id: str):
    if not query_id:
        return

    cols = st.columns([1, 1, 4])
    if cols[0].button("👍", key=f"up_{query_id}"):
        from feedback import save_feedback
        save_feedback(query_id, thumbs_up=True)
        st.toast("Спасибо за отзыв!")

    if cols[1].button("👎", key=f"down_{query_id}"):
        st.session_state[f"show_comment_{query_id}"] = True

    if st.session_state.get(f"show_comment_{query_id}"):
        comment = st.text_input(
            "Что можно улучшить?",
            key=f"comment_{query_id}",
        )
        if st.button("Отправить", key=f"send_{query_id}"):
            from feedback import save_feedback
            save_feedback(query_id, thumbs_up=False, comment=comment)
            st.session_state[f"show_comment_{query_id}"] = False
            st.toast("Спасибо за отзыв!")
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Indexing
# ---------------------------------------------------------------------------

def page_indexing():
    st.title("📚 Индексация")
    st.caption("Управление источниками данных и индексами")

    _render_global_stats()

    cols = st.columns(3)

    with cols[0]:
        _render_source_card(
            title="📄 Документация K8s",
            source_key="docs",
            data_dir=BASE_DIR / "data" / "k8s-docs",
            download_script="scripts/download_docs.py",
            file_pattern="*.md",
        )

    with cols[1]:
        _render_source_card(
            title="💻 Исходный код",
            source_key="code",
            data_dir=BASE_DIR / "data" / "k8s-code",
            download_script="scripts/download_code.py",
            file_pattern="*.go",
        )

    with cols[2]:
        _render_source_card(
            title="🔌 API спецификация",
            source_key="api",
            data_dir=BASE_DIR / "data" / "k8s-api-specs",
            download_script="scripts/download_api_specs.py",
            file_pattern="*.json",
        )

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("🔄 Переиндексировать изменения", use_container_width=True):
        _run_indexing("all", force=False)

    if col2.button("⚠️ Переиндексировать всё", use_container_width=True):
        if st.session_state.get("confirm_reindex"):
            _run_indexing("all", force=True)
            st.session_state["confirm_reindex"] = False
        else:
            st.session_state["confirm_reindex"] = True
            st.warning("Нажмите ещё раз для подтверждения полной переиндексации.")


def _render_global_stats():
    with st.container():
        cols = st.columns(3)
        cols[0].metric("Модель эмбеддингов", EMBEDDING_MODEL)
        cols[1].metric("LLM", LLM_MODEL)

        from db import ContentHash, get_session
        session = get_session()
        try:
            total_files = session.query(ContentHash).count()
            cols[2].metric("Проиндексировано файлов", total_files)
        finally:
            session.close()


def _render_source_card(
    title: str,
    source_key: str,
    data_dir: Path,
    download_script: str,
    file_pattern: str,
):
    st.subheader(title)

    file_count = len(list(data_dir.glob(file_pattern))) if data_dir.exists() else 0
    st.metric("Файлов", file_count)

    from db import ContentHash, get_session
    session = get_session()
    try:
        indexed = (
            session.query(ContentHash)
            .filter(ContentHash.source_type == source_key)
            .count()
        )
        last_entry = (
            session.query(ContentHash)
            .filter(ContentHash.source_type == source_key)
            .order_by(ContentHash.indexed_at.desc())
            .first()
        )
        last_indexed = last_entry.indexed_at.strftime("%Y-%m-%d %H:%M") if last_entry else "—"
    finally:
        session.close()

    st.metric("Проиндексировано", indexed)
    st.caption(f"Последняя индексация: {last_indexed}")

    if st.button(f"Скачать", key=f"dl_{source_key}", use_container_width=True):
        with st.spinner("Скачивание..."):
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / download_script)],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if result.returncode == 0:
                st.success("Скачано!")
            else:
                st.error(f"Ошибка: {result.stderr}")
        st.rerun()

    if st.button(f"Переиндексировать", key=f"idx_{source_key}", use_container_width=True):
        _run_indexing(source_key, force=False)


def _run_indexing(source: str, force: bool):
    cmd = [sys.executable, str(BASE_DIR / "index.py"), "--source", source]
    if force:
        cmd.append("--force")

    with st.spinner(f"Индексация {source}..."):
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(BASE_DIR),
        )
        if result.returncode == 0:
            st.success("Индексация завершена!")
            if result.stdout:
                st.code(result.stdout, language=None)
        else:
            st.error(f"Ошибка индексации")
            st.code(result.stderr, language=None)


# ---------------------------------------------------------------------------
# Page: Evaluation
# ---------------------------------------------------------------------------

def page_evaluation():
    st.title("📊 Оценка качества")

    tab1, tab2, tab3 = st.tabs(["Оценка RAGAS", "История", "Производительность и стоимость"])

    with tab1:
        _render_evaluation_tab()

    with tab2:
        _render_history_tab()

    with tab3:
        _render_performance_tab()


def _render_evaluation_tab():
    test_file = BASE_DIR / "data" / "test_dataset.json"
    if test_file.exists():
        with open(test_file) as f:
            test_data = json.load(f)
        st.metric("Вопросов в датасете", len(test_data))

        with st.expander("Просмотр тестового датасета"):
            for i, item in enumerate(test_data, 1):
                st.markdown(
                    f"**{i}. {item['question']}**\n\n"
                    f"Ожидаемые источники: `{', '.join(item.get('expected_sources', []))}`"
                )
    else:
        st.warning("Тестовый датасет не найден (data/test_dataset.json)")
        return

    if st.button("🚀 Запустить оценку", type="primary"):
        from evaluate import run_evaluation

        progress_bar = st.progress(0, text="Подготовка...")

        def progress_cb(current, total):
            progress_bar.progress(current / total, text=f"Вопрос {current}/{total}")

        with st.spinner("Запуск оценки..."):
            results = run_evaluation(progress_callback=progress_cb)

        progress_bar.empty()
        _render_evaluation_results(results)


def _render_evaluation_results(results: dict):
    st.subheader(f"Результаты (run: {results['run_id']})")

    metrics = results["metrics"]
    cols = st.columns(4)
    metric_names = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_relevancy", "Context Relevancy"),
        ("context_precision", "Context Precision"),
    ]

    for i, (key, label) in enumerate(metric_names):
        value = metrics.get(key, 0)
        color = "🟢" if value >= 0.8 else "🟡" if value >= 0.5 else "🔴"
        cols[i].metric(f"{color} {label}", f"{value:.3f}")

    per_q = results.get("per_question", [])
    if per_q:
        with st.expander("Подробности по вопросам"):
            for i, q in enumerate(per_q, 1):
                st.markdown(f"**{i}. {q['question']}**")
                pcols = st.columns(5)
                pcols[0].caption(f"Маршрут: {q.get('actual_route', '—')}")
                pcols[1].caption(f"Faith: {q.get('faithfulness', 0):.2f}")
                pcols[2].caption(f"Ans Rel: {q.get('answer_relevancy', 0):.2f}")
                pcols[3].caption(f"Ctx Rel: {q.get('context_relevancy', 0):.2f}")
                pcols[4].caption(f"Ctx Prec: {q.get('context_precision', 0):.2f}")
                if q.get("answer_preview"):
                    st.text(q["answer_preview"])
                st.divider()


def _render_history_tab():
    from evaluate import get_evaluation_history

    history = get_evaluation_history()

    if not history:
        st.info("Пока нет запусков оценки.")
        return

    st.subheader(f"Всего запусков: {len(history)}")

    chart_data = {
        "run": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_relevancy": [],
        "context_precision": [],
    }
    for run in reversed(history):
        chart_data["run"].append(run["run_id"][:6])
        chart_data["faithfulness"].append(run["faithfulness"])
        chart_data["answer_relevancy"].append(run["answer_relevancy"])
        chart_data["context_relevancy"].append(run["context_relevancy"])
        chart_data["context_precision"].append(run["context_precision"])

    import pandas as pd
    df = pd.DataFrame(chart_data).set_index("run")
    st.line_chart(df)

    for run in history:
        with st.expander(f"Run {run['run_id']} — {run['timestamp'][:19]}"):
            cols = st.columns(4)
            cols[0].metric("Faithfulness", f"{run['faithfulness']:.3f}")
            cols[1].metric("Answer Rel.", f"{run['answer_relevancy']:.3f}")
            cols[2].metric("Context Rel.", f"{run['context_relevancy']:.3f}")
            cols[3].metric("Context Prec.", f"{run['context_precision']:.3f}")


def _render_performance_tab():
    from feedback import get_feedback_stats, get_metrics_summary

    try:
        metrics_summary = get_metrics_summary()
        feedback_stats = get_feedback_stats()
    except Exception:
        st.info("Пока нет данных. Задайте несколько вопросов в чате.")
        return

    st.subheader("Общая статистика")
    cols = st.columns(4)
    cols[0].metric("Всего запросов", metrics_summary["total_queries"])
    cols[1].metric("Ср. задержка", f"{metrics_summary['avg_latency_ms']:.0f} мс")
    cols[2].metric("Общая стоимость", f"${metrics_summary['total_cost']:.4f}")
    cols[3].metric("Cache hit rate", f"{metrics_summary['cache_hit_rate']:.0%}")

    st.divider()

    fcols = st.columns(3)
    fcols[0].metric("Отзывов", feedback_stats["total"])
    fcols[1].metric("Положительных", feedback_stats["positive"])
    fcols[2].metric("Соотношение 👍", f"{feedback_stats['ratio']:.0%}")

    recent = metrics_summary.get("recent_metrics", [])
    if recent:
        st.subheader("Последние запросы — задержки")
        import pandas as pd

        latency_data = []
        for m in recent[:20]:
            latency_data.append({
                "query": m["query_id"][:6],
                "Embedding": m["latency_embedding_ms"],
                "Routing": m["latency_routing_ms"],
                "Search": m["latency_search_ms"],
                "Generation": m["latency_generation_ms"],
            })

        df = pd.DataFrame(latency_data).set_index("query")
        st.bar_chart(df)

        st.subheader("Тренд стоимости")
        cost_data = [
            {"query": m["query_id"][:6], "cost": m["estimated_cost"]}
            for m in reversed(recent[:20])
        ]
        cost_df = pd.DataFrame(cost_data).set_index("query")
        st.line_chart(cost_df)


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

page = get_page()
if page == "chat":
    page_chat()
elif page == "indexing":
    page_indexing()
elif page == "evaluation":
    page_evaluation()
