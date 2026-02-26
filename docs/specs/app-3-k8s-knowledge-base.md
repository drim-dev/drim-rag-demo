# Спецификация: App 3 — K8s Knowledge Base

## Общая информация

| Параметр | Значение |
|---|---|
| Название | K8s Knowledge Base |
| Директория | `app-3-k8s-knowledge-base/` |
| Фреймворк | LlamaIndex |
| Интерфейс | Streamlit (многостраничное) |
| Векторная БД | pgvector (PostgreSQL 16, Docker) |
| Эмбеддинги | Ollama `nomic-embed-text` (768 dim, локально) |
| LLM | Claude Sonnet (`claude-sonnet-4-20250514`) |
| Оценка | RAGAS (через GPT-4o-mini + OpenAI Embeddings) |
| Парсинг кода | tree-sitter (Go) |
| Ориентировочная стоимость | ~$0.50 |
| Время настройки | ~10 мин |

## Назначение

Продакшен-уровневая мульти-индексная RAG-система с тремя источниками данных, LLM-маршрутизацией запросов, оценкой качества (RAGAS), семантическим кешированием, учётом стоимости и обратной связью. Покрывает разделы урока: оценка, продакшен, варианты использования.

## Структура файлов

```
app-3-k8s-knowledge-base/
├── app.py                   # Streamlit UI (3 страницы: Чат, Индексация, Оценка)
├── config.py                # Общая конфигурация (модели, БД, стоимости, пороги)
├── db.py                    # SQLAlchemy ORM: Metric, Feedback, EvaluationRun, SemanticCache, ContentHash
├── router.py                # LLM-маршрутизатор запросов (Claude)
├── pipeline.py              # Полный RAG-конвейер: кэш → маршрутизация → поиск → генерация → метрики
├── index.py                 # Мульти-источниковая индексация (docs, code, API specs)
├── evaluate.py              # Оценка RAGAS на тестовом датасете
├── feedback.py              # Сбор и статистика обратной связи
├── docker-compose.yml       # PostgreSQL + pgvector, Ollama + модель
├── .env.example             # ANTHROPIC_API_KEY, OPENAI_API_KEY, настройки Postgres и Ollama
├── scripts/
│   ├── download_docs.py     # Загрузка markdown-документации K8s
│   ├── download_code.py     # Загрузка Go-исходников K8s
│   └── download_api_specs.py # Загрузка OpenAPI-спецификации K8s
└── data/
    ├── k8s-docs/            # Markdown-файлы документации
    ├── k8s-code/            # Go-файлы исходного кода
    ├── k8s-api-specs/       # swagger.json
    └── test_dataset.json    # 50 троек вопрос-ответ для RAGAS
```

## Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `ANTHROPIC_API_KEY` | API-ключ Anthropic (для Claude) | — (обязательно) |
| `OPENAI_API_KEY` | API-ключ OpenAI (для RAGAS оценки) | — (для оценки) |
| `POSTGRES_HOST` | Хост PostgreSQL | `localhost` |
| `POSTGRES_PORT` | Порт PostgreSQL | `5435` |
| `POSTGRES_DB` | Имя базы данных | `k8s_rag` |
| `POSTGRES_USER` | Пользователь PostgreSQL | `postgres` |
| `POSTGRES_PASSWORD` | Пароль PostgreSQL | `postgres` |
| `OLLAMA_HOST` | URL Ollama | `http://localhost:11434` |

## Инфраструктура (Docker)

### PostgreSQL + pgvector
- Образ: `pgvector/pgvector:pg16`
- Порт: `5435:5432`
- Том: `pgdata` (Docker named volume)
- БД: `k8s_rag`, пользователь: `postgres`, пароль: `postgres`
- Расширение `vector` создаётся автоматически при `init_db()`

### Ollama
- Образ: `ollama/ollama`
- Порт: `11434:11434`
- Том: `ollama_data` (Docker named volume)
- Init-контейнер `ollama-pull`: автоматически скачивает модель `nomic-embed-text`

## Компонент: Конфигурация (`config.py`)

| Константа | Значение | Описание |
|---|---|---|
| `EMBEDDING_MODEL` | `nomic-embed-text` | Модель эмбеддингов Ollama |
| `EMBEDDING_DIM` | `768` | Размерность вектора |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Модель для генерации и маршрутизации |
| `COST_PER_INPUT_TOKEN` | `3.0 / 1_000_000` | Стоимость входного токена Claude Sonnet |
| `COST_PER_OUTPUT_TOKEN` | `15.0 / 1_000_000` | Стоимость выходного токена Claude Sonnet |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Порог косинусного сходства для кэша |
| `MIN_RELEVANCE_SCORE` | `0.6` | Минимальная релевантность чанка |
| `DOCS_INDEX` | `docs_index` | Имя таблицы pgvector для документации |
| `CODE_INDEX` | `code_index` | Имя таблицы pgvector для кода |
| `API_SPECS_INDEX` | `api_specs_index` | Имя таблицы pgvector для API-спецификаций |

## Компонент: База данных (`db.py`)

ORM на SQLAlchemy с расширением pgvector. Все таблицы создаются автоматически через `init_db()`.

### Таблица `metrics`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | Integer PK | Автоинкремент |
| `query_id` | String (indexed) | Идентификатор запроса (UUID[:8]) |
| `timestamp` | DateTime | Время запроса (UTC) |
| `latency_embedding_ms` | Float | Задержка эмбеддинга (мс) |
| `latency_routing_ms` | Float | Задержка маршрутизации (мс) |
| `latency_search_ms` | Float | Задержка поиска (мс) |
| `latency_generation_ms` | Float | Задержка генерации (мс) |
| `total_latency_ms` | Float | Общая задержка (мс) |
| `input_tokens` | Integer | Входные токены LLM |
| `output_tokens` | Integer | Выходные токены LLM |
| `estimated_cost` | Float | Ориентировочная стоимость ($) |
| `cache_hit` | Boolean | Попадание в кэш |

### Таблица `feedback`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | Integer PK | Автоинкремент |
| `query_id` | String (indexed) | Идентификатор запроса |
| `thumbs_up` | Boolean | Положительная оценка |
| `comment` | Text (nullable) | Текстовый комментарий |
| `timestamp` | DateTime | Время отзыва (UTC) |

### Таблица `evaluation_runs`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | Integer PK | Автоинкремент |
| `run_id` | String (unique) | UUID[:8] запуска |
| `timestamp` | DateTime | Время запуска (UTC) |
| `avg_faithfulness` | Float | Средняя Faithfulness |
| `avg_answer_relevancy` | Float | Средняя Answer Relevancy |
| `avg_context_recall` | Float | Средняя Context Recall |
| `avg_context_precision` | Float | Средняя Context Precision |
| `per_question_json` | Text | JSON с результатами по вопросам |

### Таблица `semantic_cache`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | Integer PK | Автоинкремент |
| `query_embedding` | Vector(768) | Вектор запроса |
| `original_query` | Text | Исходный текст запроса |
| `response_text` | Text | Кэшированный ответ |
| `sources_json` | Text | JSON с источниками |
| `route_label` | String | Метка маршрута |
| `created_at` | DateTime | Время создания (UTC) |

### Таблица `content_hashes`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | Integer PK | Автоинкремент |
| `source_type` | String | Тип источника: `docs`, `code`, `api` |
| `file_path` | String (unique) | Путь к файлу |
| `content_hash` | String | SHA-256 содержимого файла |
| `indexed_at` | DateTime | Время индексации (UTC) |

## Компонент: Маршрутизатор запросов (`router.py`)

### Принцип работы

LLM-классификатор на Claude, который по тексту запроса определяет, в каких индексах искать.

### Промпт маршрутизации

Описывает три индекса:
1. **docs** — K8s документация: концепции, задачи, учебники
2. **code** — Go-исходники K8s: функции, типы, интерфейсы
3. **api** — API-спецификация K8s: эндпоинты, схемы запросов/ответов

Примеры маршрутизации в промпте:
- «What is a Pod?» → `docs` (0.95)
- «How is the scheduler loop implemented?» → `code` (0.9)
- «What fields does the Pod spec have?» → `api` (0.9)
- «How does kubectl apply work?» → `all` (0.85)
- «How to set resource limits via API?» → `docs+api` (0.85)

### Формат ответа

```json
{
  "indices": ["docs", "code"],
  "label": "docs+code",
  "confidence": 0.85
}
```

### Fallback

При ошибке парсинга JSON: `{"indices": ["docs"], "label": "docs", "confidence": 0.5}`

## Компонент: Индексация (`index.py`)

### Общая архитектура

Три конвейера индексации (docs, code, api) с общей инфраструктурой:
- **Эмбеддинги:** `OllamaEmbedding(model_name="nomic-embed-text")`
- **Хранилище:** `PGVectorStore` с отдельной таблицей на индекс
- **Чанкинг при вставке:** `SentenceSplitter(chunk_size=1024, chunk_overlap=100)`
- **Инкрементальная индексация:** SHA-256 хеши файлов в таблице `content_hashes`

### Конвейер индексации документации

1. **Данные:** `data/k8s-docs/*.md` (скачиваются `scripts/download_docs.py`)
2. **Обработка каждого файла:**
   - Извлечение заголовка из YAML frontmatter (`title:`) или первого `#`
   - Удаление YAML frontmatter
   - Разбиение по markdown-заголовкам (уровни `#`–`####`)
   - Пропуск секций < 50 символов
3. **Метаданные:**
   - `source_type`: `docs`
   - `file_name`: имя файла
   - `page_title`: заголовок страницы
   - `section`: заголовок секции
   - `category`: `concept` / `task` / `tutorial` / `other` (по подстроке в имени файла)
4. **Индекс:** `docs_index`

### Конвейер индексации кода

1. **Данные:** `data/k8s-code/*.go` (скачиваются `scripts/download_code.py`)
2. **Парсинг Go через tree-sitter:**
   - Извлечение узлов: `function_declaration`, `method_declaration`, `type_declaration`, `type_spec`
   - Для `type_spec` — проверка на `interface_type` для определения типа символа
   - Пропуск символов < 20 символов
3. **Метаданные:**
   - `source_type`: `code`
   - `file_path`: восстановленный путь из имени файла (`__` → `/`)
   - `package`: из `package` директивы Go
   - `symbol_name`: имя функции/типа/интерфейса
   - `symbol_type`: `func` / `type` / `interface`
4. **Индекс:** `code_index`

### Конвейер индексации API-спецификаций

1. **Данные:** `data/k8s-api-specs/swagger.json` (скачивается `scripts/download_api_specs.py`)
2. **Парсинг OpenAPI:** каждый эндпоинт = один документ
   - Формат текста:
     ```
     API Endpoint: {METHOD} {path}
     Operation: {operationId}
     Summary: {summary}
     Description: {description}
     Success Response: {responses.200.description}
     ```
3. **Метаданные:**
   - `source_type`: `api`
   - `api_group`: первый тег из `tags[]` (или `core`)
   - `resource_kind`: извлекается из `operationId` (удаление глаголов CRUD + `Namespaced`)
   - `http_method`: HTTP-метод (верхний регистр)
   - `path`: путь эндпоинта
4. **Индекс:** `api_specs_index`

### CLI

```bash
python index.py --source docs           # Только документация
python index.py --source code           # Только код
python index.py --source api            # Только API-спецификации
python index.py --source all            # Все источники
python index.py --source all --force    # Полная переиндексация (игнорировать хеши)
```

## Компонент: RAG-конвейер (`pipeline.py`)

### Потоковая версия (`query_pipeline_streaming`)

Генератор, yield-ящий кортежи `(token, metadata_or_none)`. Последний yield содержит полные метаданные.

### Алгоритм

1. **Эмбеддинг запроса:** `OllamaEmbedding.get_query_embedding(query)` — замер времени
2. **Проверка кэша** (если `use_cache=True`):
   - SQL-запрос: `1 - (query_embedding <=> :embedding)` ≥ 0.95
   - При попадании: вернуть кэшированный ответ + метрики с `cache_hit=True`
3. **Маршрутизация:** `route_query(query)` → список индексов + метка + уверенность
4. **Параллельный поиск:** `ThreadPoolExecutor` — поиск в каждом из выбранных индексов
   - `VectorStoreIndex.from_vector_store()` → `as_retriever(similarity_top_k=top_k)`
   - Объединение результатов, сортировка по `score` (убывание)
5. **Построение контекста:** `[Source {i} — {label}] ({metadata})\n{text}`, разделённые `---`
6. **Генерация:** Claude с промптом SYNTHESIS_PROMPT (max_tokens=2048)
7. **Учёт токенов:** из `response.raw.usage` (input_tokens, output_tokens)
8. **Сохранение в кэш:** если `use_cache=True` — сохранить эмбеддинг, ответ, источники, маршрут
9. **Запись метрик:** в таблицу `metrics`

### Промпт генерации (SYNTHESIS_PROMPT)

```
You are a Kubernetes expert assistant. Answer the user's question using ONLY
the provided context from multiple sources. Cite your sources using labels
[docs], [code], or [api] after each claim.

If the context is insufficient, say so honestly — do not make up information.

Context:
-----
{context_str}
-----

Question: {query_str}

Answer:
```

### Формат метаданных ответа

```json
{
  "query_id": "abc12345",
  "sources": [
    {
      "text": "первые 300 символов",
      "score": 0.87,
      "source_type": "docs",
      "metadata": {"page_title": "...", "section": "..."}
    }
  ],
  "route": {"label": "docs+code", "confidence": 0.85, "indices": ["docs", "code"]},
  "metrics": {
    "latency_embedding_ms": 120,
    "latency_routing_ms": 800,
    "latency_search_ms": 350,
    "latency_generation_ms": 2100,
    "total_latency_ms": 3370,
    "input_tokens": 3500,
    "output_tokens": 450,
    "estimated_cost": 0.0173
  },
  "cache_hit": false
}
```

## Компонент: Оценка (`evaluate.py`)

### Алгоритм

1. Загрузка тестового датасета из `data/test_dataset.json`
2. Для каждого вопроса: полный прогон через `query_pipeline()` (без кэша)
3. Формирование `Dataset` (HuggingFace) с полями: `question`, `answer`, `contexts`, `ground_truth`
4. Запуск RAGAS `evaluate()` с метриками:
   - `faithfulness` — ответ основан только на контексте
   - `answer_relevancy` — ответ релевантен вопросу
   - `context_recall` — контекст покрывает ground truth
   - `context_precision` — релевантные чанки выше нерелевантных
5. LLM для RAGAS: `GPT-4o-mini` (через LangchainLLMWrapper)
6. Эмбеддинги для RAGAS: `OpenAIEmbeddings` (через LangchainEmbeddingsWrapper)
7. Сохранение результатов в таблицу `evaluation_runs`

### Формат тестового датасета

```json
[
  {
    "question": "What is a Pod in Kubernetes?",
    "expected_answer": "A Pod is the smallest deployable unit...",
    "expected_sources": ["docs"],
    "ground_truth_context": "..."
  }
]
```

### История оценок

`get_evaluation_history()` — возвращает все запуски из таблицы `evaluation_runs`, отсортированные по дате (новейшие первыми).

## Компонент: Обратная связь (`feedback.py`)

- `save_feedback(query_id, thumbs_up, comment)` — сохранение в таблицу `feedback`
- `get_feedback_stats()` — агрегированная статистика: total, positive, negative, ratio
- `get_metrics_summary()` — агрегированные метрики запросов:
  - Общее количество, средняя задержка, общая стоимость, cache hit rate
  - Последние 50 запросов с полной разбивкой по этапам

## Компонент: Streamlit UI (`app.py`)

### Навигация

Три кнопки вверху страницы: «Чат», «Индексация», «Оценка». Текущая страница подсвечена `type="primary"`. Состояние хранится в `st.session_state["page"]`.

### Страница «Чат» (главная)

| Элемент | Описание |
|---|---|
| Боковая панель: Top-K | Slider 1–10, по умолчанию 5 |
| Боковая панель: Семантический кэш | Checkbox, по умолчанию включён |
| Чат-интерфейс | `st.chat_input` + `st.chat_message` |
| Потоковый ответ | Посимвольный вывод с курсором `▌` |
| Бейджи | Маршрут (🎯), Кэш (⚡), Задержка (⏱️), Типы источников (🔵🟢🟠) |
| Метрики (expander) | 4 карточки задержки + 3 карточки токенов/стоимости + список источников |
| Обратная связь | 👍/👎 кнопки, текстовый ввод при 👎 |

### Цветовая кодировка источников

| Тип | Иконка | Цвет |
|---|---|---|
| `docs` | 🔵 | Синий |
| `code` | 🟢 | Зелёный |
| `api` | 🟠 | Оранжевый |

### Страница «Индексация»

| Элемент | Описание |
|---|---|
| Глобальная статистика | Модель эмбеддингов, LLM, количество проиндексированных файлов |
| 3 карточки источников | Каждая: заголовок, количество файлов, проиндексировано, время последней индексации, кнопка скачивания, кнопка переиндексации |
| Переиндексировать изменения | `python index.py --source all` |
| Переиндексировать всё | `python index.py --source all --force` (с подтверждением) |

### Страница «Оценка»

**Вкладка «Оценка RAGAS»:**
- Просмотр тестового датасета (количество вопросов, таблица с вопросами и ожидаемыми источниками)
- Кнопка «Запустить оценку» с прогресс-баром
- 4 карточки метрик с цветовой индикацией: 🟢 ≥ 0.8, 🟡 ≥ 0.5, 🔴 < 0.5
- Подробности по вопросам (expander): маршрут, все 4 метрики, превью ответа

**Вкладка «История»:**
- Линейный график всех метрик по запускам оценки (`st.line_chart`)
- Раскрывающиеся панели по каждому запуску с метриками

**Вкладка «Производительность и стоимость»:**
- Общая статистика: запросов, средняя задержка, общая стоимость, cache hit rate
- Обратная связь: отзывов, положительных, соотношение
- Столбчатая диаграмма задержек по последним 20 запросам
- Линейный график тренда стоимости

## Упражнения для студентов

1. Добавить новый источник данных (напр., посты блога K8s) как четвёртый индекс
2. Заменить эмбеддинги Ollama на OpenAI и сравнить оценки RAGAS
3. Отключить семантическое кеширование и сравнить стоимость за 20 запросов
4. Добавить «порог уверенности» — если ни один чанк не набирает выше 0.6, отвечать «У меня недостаточно информации»
