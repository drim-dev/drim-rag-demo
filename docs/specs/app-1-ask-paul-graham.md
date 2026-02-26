# Спецификация: App 1 — Ask Paul Graham

## Общая информация

| Параметр | Значение |
|---|---|
| Название | Ask Paul Graham |
| Директория | `app-1-ask-paul-graham/` |
| Фреймворк | LlamaIndex |
| Интерфейс | Streamlit |
| Векторная БД | Qdrant (Docker) |
| Эмбеддинги | OpenAI `text-embedding-3-small` (1536 dim) |
| LLM | GPT-4o-mini (temperature 0.1) |
| Ориентировочная стоимость | ~$0.05 |
| Время настройки | ~3 мин |

## Назначение

Первое знакомство с RAG. Студент получает момент «RAG работает!» — минимальный конвейер от загрузки данных до ответа с цитированием источников. Покрывает разделы урока: проблема, архитектура, реализация.

## Структура файлов

```
app-1-ask-paul-graham/
├── app.py               # Streamlit UI + RAG-конвейер (единый файл)
├── download_data.py      # Скрипт загрузки эссе с paulgraham.com
├── docker-compose.yml    # Qdrant (порты 6333, 6334)
├── .env.example          # OPENAI_API_KEY, QDRANT_URL
└── data/                 # .gitignore — заполняется скриптами
    ├── essays/           # .txt файлы эссе
    └── qdrant/           # Персистентное хранилище Qdrant
```

## Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `OPENAI_API_KEY` | API-ключ OpenAI | — (обязательно) |
| `QDRANT_URL` | URL Qdrant | `http://localhost:6333` |

## Инфраструктура (Docker)

**Qdrant** — единственный Docker-сервис:
- Образ: `qdrant/qdrant:latest`
- Порты: `6333:6333` (HTTP API), `6334:6334` (gRPC)
- Том: `./data/qdrant:/qdrant/storage` (персистентное хранение)
- Коллекция: `paul_graham`

## Компонент: Загрузка данных (`download_data.py`)

### Алгоритм

1. GET запрос на `http://paulgraham.com/articles.html`
2. Парсинг HTML (BeautifulSoup) — извлечение всех ссылок на `.html` страницы
3. Фильтрация: только ссылки на `paulgraham.com`, исключая `articles.html`
4. Для каждой ссылки:
   - GET запрос на страницу эссе
   - Извлечение заголовка (`<title>`) и текста (`<body>`, удаление `<script>`, `<style>`, `<img>`)
   - Пропуск слишком коротких текстов (< 200 символов)
   - Сохранение в `data/essays/{slug}.txt` с заголовком `Title: ...` и `Source: URL` в начале файла
   - Пауза 1 сек между запросами (rate limiting)
5. Вывод итогов: количество скачанных, пропущенных, ошибочных; общий размер

### Формат выходного файла

```
Title: How to Do Great Work
Source: http://paulgraham.com/greatwork.html

<текст эссе>
```

## Компонент: RAG-конвейер (`app.py`)

### Загрузка и чанкинг

- **Загрузчик:** `SimpleDirectoryReader` (LlamaIndex) — читает все `.txt` файлы из `data/essays/`
- **Сплиттер:** `SentenceSplitter` (LlamaIndex)
  - `chunk_size`: настраивается (256 / 512 / 1024), по умолчанию 512
  - `chunk_overlap`: вычисляется как `chunk_size * overlap_pct`, где `overlap_pct` ∈ {0%, 10%, 20%}
- **Метаданные чанка:** `file_name` (извлекается автоматически из имени файла)

### Эмбеддинги и индексация

- **Модель эмбеддингов:** `OpenAIEmbedding(model="text-embedding-3-small")` — 1536 измерений
- **LLM:** `OpenAI(model="gpt-4o-mini", temperature=0.1)` — задаётся через `Settings.llm`
- **Векторное хранилище:** `QdrantVectorStore(collection_name="paul_graham", client=qdrant_client)`
- **Построение индекса:** `VectorStoreIndex(nodes, storage_context, show_progress=True)`
- При перестроении индекса: существующая коллекция удаляется и создаётся заново

### Загрузка существующего индекса

При запуске приложения выполняется попытка загрузки существующего индекса из Qdrant (`VectorStoreIndex.from_vector_store`). Если коллекция существует и содержит точки — индекс загружается без перестроения.

### Запрос и генерация

- **Метод поиска:** косинусное сходство (Qdrant, top-K)
- **Top-K:** настраивается слайдером (1–10, по умолчанию 5)
- **Промпт-шаблон:**
  ```
  You are an AI assistant that answers questions about Paul Graham's essays.
  Use ONLY the context provided below to answer.
  If the context does not contain enough information to answer, say so honestly —
  do not make up information.
  When answering, cite which essay(s) your answer comes from.

  Context:
  -----
  {context_str}
  -----

  Question: {query_str}

  Answer:
  ```
- **Стриминг:** да (`streaming=True` в `query_engine`)
- **Query engine:** `index.as_query_engine(similarity_top_k=top_k, text_qa_template=QA_PROMPT_TEMPLATE, streaming=True)`

### Обработка результатов

Для каждого `source_node` из ответа:
- `title` — извлекается из `metadata["file_name"]`, `.txt` удаляется, дефисы заменяются пробелами, приводится к Title Case
- `score` — оценка косинусного сходства (float)
- `text` — полный текст чанка

## Компонент: Streamlit UI

### Боковая панель

| Элемент | Тип | Описание |
|---|---|---|
| Метрика «Эссе на диске» | `st.sidebar.metric` | Количество `.txt` файлов в `data/essays/` |
| Кнопка «Скачать эссе» | `st.sidebar.button` | Запускает `download_essays()`, показывает спиннер |
| Chunk size | `st.sidebar.selectbox` | Варианты: 256, 512, 1024. По умолчанию: 512 |
| Chunk overlap | `st.sidebar.selectbox` | Варианты: 0%, 10%, 20%. По умолчанию: 10% |
| Кнопка «Построить индекс» | `st.sidebar.button` | Disabled, если эссе не скачаны. Запускает `build_index()` |
| Статистика индекса | `st.sidebar.columns` | Количество чанков, размерность, время построения (сек) |
| Top-K | `st.sidebar.slider` | Диапазон: 1–10, по умолчанию: 5 |

### Основная область

| Элемент | Тип | Описание |
|---|---|---|
| Поле ввода | `st.text_input` | Плейсхолдер: «What is the most important thing for a startup?» |
| Кнопка «Спросить» | `st.button` | `type="primary"`, disabled если нет индекса или пустой ввод |
| Ответ | `st.empty` + markdown | Потоковое отображение (посимвольный вывод из `response_gen`) |
| Найденные фрагменты | `st.expander` | Для каждого чанка: номер, заголовок эссе, релевантность (%), текст (первые 500 символов) |
| Подсказка | `st.info` | «Постройте индекс...», показывается если индекс не создан |

### Хранение состояния (`st.session_state`)

| Ключ | Тип | Описание |
|---|---|---|
| `index` | `VectorStoreIndex` | Текущий индекс |
| `index_stats` | `dict` | `chunk_count`, `embedding_dim`, `elapsed_seconds` |

## Упражнения для студентов

1. Изменить `chunk_size` с 512 на 128 — наблюдать деградацию ответов при маленьких чанках
2. Сравнить top-K = 1 vs top-K = 10 — компромисс точность/полнота
3. Спросить о событиях после 2023 года — наблюдать поведение «Я не знаю» vs галлюцинации
