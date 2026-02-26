# Спецификация: App 2 — Python Docs Assistant

## Общая информация

| Параметр | Значение |
|---|---|
| Название | Python Docs Assistant |
| Директория | `app-2-python-docs-assistant/` |
| Фреймворк | LangChain |
| Интерфейс | Gradio |
| Векторная БД | Qdrant (Docker) |
| Эмбеддинги | Cohere `embed-v4.0` (1536 dim) |
| LLM | GPT-4o (temperature 0, streaming) |
| Реранкер | Cohere `rerank-v3.5` |
| HyDE LLM | GPT-4o-mini (temperature 0.7) |
| Ориентировочная стоимость | ~$0.30 |
| Время настройки | ~5 мин |

## Назначение

Демонстрация ограничений наивного RAG на структурированных документах и продвинутых техник извлечения. Покрывает разделы урока: варианты использования, продвинутые техники.

## Структура файлов

```
app-2-python-docs-assistant/
├── app.py                # Gradio UI (3 вкладки)
├── index.py              # Конвейер индексации (RST-парсинг, чанкинг, Qdrant + BM25)
├── retrieval.py          # 6 методов извлечения + оркестрация
├── download_data.py      # Загрузка RST-файлов из CPython
├── docker-compose.yml    # Qdrant (порты 6333, 6334)
├── .env.example          # OPENAI_API_KEY, COHERE_API_KEY
└── data/
    ├── python-docs/      # RST-файлы (library/ + tutorial/)
    ├── bm25_index.pkl    # Сериализованный BM25 индекс
    └── chunks.pkl        # Сериализованные данные чанков
```

## Переменные окружения

| Переменная | Описание |
|---|---|
| `OPENAI_API_KEY` | API-ключ OpenAI (для GPT-4o генерации и HyDE) |
| `COHERE_API_KEY` | API-ключ Cohere (для эмбеддингов и реранкинга) |

## Инфраструктура (Docker)

**Qdrant:**
- Образ: `qdrant/qdrant:latest`
- Порты: `6333:6333`, `6334:6334`
- Том: `qdrant_data` (Docker named volume)
- Коллекция: `python_docs`
- Параметры вектора: 1536 dim, расстояние COSINE

## Компонент: Загрузка данных (`download_data.py`)

### Алгоритм

1. Загрузка ZIP-архива ветки `main` из `python/cpython` на GitHub
2. Извлечение RST-файлов из директории `Doc/` архива
3. **Фильтрация:** сохраняются только:
   - Все RST-файлы из `tutorial/`
   - Конкретные модули из `library/` (список из 36 модулей: `os`, `sys`, `pathlib`, `asyncio`, `collections`, `json`, `re`, `typing`, `dataclasses`, `logging`, `argparse`, `datetime`, `unittest`, `subprocess` и др.)
4. Сохранение в `data/python-docs/` с сохранением структуры каталогов

### Список модулей

`os`, `sys`, `pathlib`, `shutil`, `asyncio`, `concurrent.futures`, `threading`, `multiprocessing`, `collections`, `itertools`, `functools`, `operator`, `json`, `csv`, `sqlite3`, `pickle`, `re`, `string`, `textwrap`, `typing`, `dataclasses`, `abc`, `enum`, `unittest`, `doctest`, `logging`, `warnings`, `argparse`, `configparser`, `subprocess`, `os.path`, `datetime`, `time`, `calendar`, `http.client`, `http.server`, `urllib.request`, `urllib.parse`, `socket`, `ssl`, `io`, `contextlib`, `pdb`, `traceback`, `importlib`, `pkgutil`, `copy`, `pprint`, `hashlib`, `secrets`, `struct`, `array`

## Компонент: Индексация (`index.py`)

### RST-парсинг

Функция `parse_rst_sections(filepath)` разбирает RST-файл на секции:

- **Обнаружение заголовков:** строка текста + подчёркивание из символов `=#-~^` длиной >= 3
- **Также поддерживается:** overline + title + underline паттерн
- **Иерархия:** отслеживание уровней по символам подчёркивания (порядок появления определяет уровень)
- **Выходные метаданные:**
  - `module` — имя модуля из пути файла (напр., `asyncio`, `tutorial/classes`)
  - `doc_type` — `tutorial` или `api-reference` (по наличию `/tutorial/` в пути)
  - `section_path` — иерархический путь через ` > ` (напр., `asyncio > Streams > StreamReader`)
  - `source_file` — имя RST-файла
- **Минимальная длина секции:** 50 символов

### Три стратегии чанкинга

#### 1. `fixed` — Фиксированный размер (baseline)
- `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
- Разделители: `\n\n`, `\n`, `. `, ` `, ``
- Каждая секция разбивается независимо

#### 2. `header-based` — По RST-заголовкам (рекомендуемая)
- Секция из RST-парсера = один чанк
- Если секция > 1500 символов — дополнительная разбивка с `chunk_overlap=200`
- Части нумеруются: `section_path (part 1)`, `section_path (part 2)`, ...

#### 3. `semantic` — Семантический
- `SemanticChunker` из `langchain_experimental` с `breakpoint_threshold_type="percentile"`
- Эмбеддинги: Cohere `embed-v4.0`
- Пропуск секций < 100 символов
- Fallback на исходный текст при ошибке чанкинга

### Parent-child маппинг

- Каждому чанку присваивается UUID (`chunk_id`)
- Группировка по ключу `source_file::section_path`
- Первый чанк в группе — родитель; все остальные чанки ссылаются на него через `parent_id`
- Одиночный чанк ссылается сам на себя

### Хранение в Qdrant

- Пакетная вставка по 64 чанка
- Вектор: Cohere `embed-v4.0` (embed_documents)
- Payload на каждую точку:
  - `text`, `module`, `doc_type`, `section_path`, `source_file`
  - `chunk_strategy`, `chunk_id`, `parent_id`

### BM25-индекс

- Библиотека: `rank_bm25.BM25Okapi`
- Токенизация: `text.lower().split()`
- Сериализация: pickle в `data/bm25_index.pkl` (BM25-объект + массив чанков)
- Также сохраняется: `data/chunks.pkl` (данные чанков)

### CLI

```bash
python index.py --strategy header-based  # по умолчанию
python index.py --strategy fixed
python index.py --strategy semantic
```

## Компонент: Извлечение (`retrieval.py`)

### Конфигурация (`RetrievalConfig`)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `use_vector` | `bool` | `True` | Включить векторный поиск |
| `use_bm25` | `bool` | `False` | Включить BM25 поиск |
| `use_hybrid` | `bool` | `False` | Гибридный (vector + BM25 + RRF) |
| `use_hyde` | `bool` | `False` | HyDE трансформация запроса |
| `use_reranking` | `bool` | `False` | Cohere реранкинг |
| `use_parent_child` | `bool` | `False` | Расширение до родительских чанков |
| `filter_module` | `str \| None` | `None` | Фильтр по модулю |
| `filter_doc_type` | `str \| None` | `None` | Фильтр по типу документа |
| `top_k` | `int` | `5` | Количество результатов |

### Результат (`RetrievedChunk`)

| Поле | Тип | Описание |
|---|---|---|
| `text` | `str` | Текст чанка |
| `score` | `float` | Оценка релевантности |
| `module` | `str` | Модуль Python |
| `doc_type` | `str` | Тип документа |
| `section_path` | `str` | Путь секции |
| `source_file` | `str` | Файл-источник |
| `chunk_id` | `str` | UUID чанка |
| `parent_id` | `str` | UUID родительского чанка |
| `retrieval_method` | `str` | Метод извлечения (напр., `vector`, `bm25`, `hybrid-rrf`, `hyde`, `+reranked`, `+parent`) |

### 6 методов извлечения

#### 1. Vector search
- `qdrant.query_points()` с вектором запроса (Cohere embed_query)
- Поддержка фильтров: `FieldCondition(key="module"/"doc_type", match=MatchValue(...))`

#### 2. BM25 search
- Загрузка сериализованного BM25 + чанков из pickle
- Токенизация запроса: `query.lower().split()`
- `bm25.get_scores(tokens)` → сортировка по убыванию
- Фильтрация по `filter_module` и `filter_doc_type` вручную
- Пропуск результатов с `score <= 0`

#### 3. Hybrid search (Reciprocal Rank Fusion)
- Параллельный запуск Vector + BM25 с `top_k * 2`
- RRF-формула: `score = Σ 1/(K + rank + 1)`, где `K = 60`
- Объединение по `chunk_id` (или первым 100 символам текста)
- Результат: метод `hybrid-rrf`

#### 4. HyDE (Hypothetical Document Embeddings)
- Генерация гипотетического ответа через GPT-4o-mini:
  ```
  Write a short technical paragraph that would answer this Python documentation
  question. Write as if you are the Python docs:
  {query}
  ```
- Эмбеддинг гипотетического ответа (Cohere)
- Векторный поиск по полученному эмбеддингу

#### 5. Реранкинг (Cohere Rerank)
- Модель: `rerank-v3.5`
- Вход: результаты предыдущего метода
- `CohereRerank.compress_documents(docs, query)` → `top_n=top_k`
- Оценка: `relevance_score` из метаданных Cohere
- Метод: `{предыдущий}+reranked`

#### 6. Parent-child retrieval
- Для каждого чанка, где `parent_id != chunk_id`:
  - Поиск родителя в Qdrant через `scroll` с фильтром `chunk_id == parent_id`
  - Замена дочернего чанка родительским (дедупликация по `parent_id`)
- Метод: `{предыдущий}+parent`

### Оркестрация (`Retriever.retrieve`)

Порядок применения:
1. Выбор базового метода: hybrid > hyde > bm25-only > vector (приоритет)
2. Если `use_reranking` — применить реранкинг
3. Если `use_parent_child` — расширить до родителей
4. Обрезать до `top_k`

### Вспомогательные методы

- `get_available_modules()` — scroll по Qdrant, извлечение уникальных `module`
- `get_available_doc_types()` — scroll по Qdrant, извлечение уникальных `doc_type`

## Компонент: Генерация ответа (`app.py: generate_answer`)

- **LLM:** `ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)`
- **Системный промпт:**
  ```
  Ты — помощник по документации Python. Отвечай на русском языке,
  опираясь ТОЛЬКО на предоставленный контекст из официальной документации Python.
  Если в контексте нет ответа, честно скажи об этом.
  Приводи примеры кода, когда это уместно.
  ```
- **Формат контекста:** `[Источник {i}: {module} — {section_path}]\n{text}`, разделённые `---`
- **Стриминг:** посимвольная аккумуляция через `llm.stream()`

## Компонент: Gradio UI (`app.py`)

### Вкладка 1 — «Индексация»

| Элемент | Описание |
|---|---|
| Кнопка «Скачать документацию Python» | Запускает `handle_download()` → `download_docs()` |
| Статус загрузки | Textbox: «Скачано X файлов, общий размер: Y КБ» |
| Стратегия чанкинга | Radio: `fixed`, `header-based` (по умолчанию), `semantic` |
| Кнопка «Построить индекс» | Запускает `handle_index(strategy)` → `run_indexing(strategy)` |
| Статистика индекса | Textbox: секции, чанки, разбивка по `doc_type` |

### Вкладка 2 — «Поиск» (основная)

| Элемент | Тип | Описание |
|---|---|---|
| Поле ввода запроса | Textbox (2 строки) | Плейсхолдер: «Как читать файл построчно?» |
| Примеры запросов | `gr.Examples` | 6 предзаполненных запросов |
| Vector search | Checkbox | По умолчанию: включён |
| BM25 | Checkbox | По умолчанию: выключен |
| Hybrid (RRF) | Checkbox | По умолчанию: выключен |
| HyDE | Checkbox | По умолчанию: выключен |
| Реранкинг (Cohere) | Checkbox | По умолчанию: выключен |
| Parent-child retrieval | Checkbox | По умолчанию: выключен |
| Фильтр по модулю | Dropdown | Динамически заполняется из индекса |
| Фильтр по типу документа | Dropdown | Динамически заполняется из индекса |
| Top-K | Slider | 1–20, по умолчанию: 5 |
| Кнопка «Спросить» | Button (primary, lg) | Запускает `handle_search()` |
| Ответ | Markdown | Потоковый вывод |
| Источники | Accordion (открыт) | `{i}. {module} — {section_path}` |
| Найденные чанки | Accordion (закрыт) | Метод, релевантность, модуль, тип, секция, текст (до 800 символов) |

### Вкладка 3 — «Сравнение»

| Элемент | Описание |
|---|---|
| Запрос для сравнения | Textbox с примерами |
| Кнопка «Сравнить» | Запускает `handle_compare(query)` |
| Naive (vector only) | Ответ + чанки (left column) |
| Advanced (hybrid + rerank + parent-child) | Ответ + чанки (right column) |
| Сводная таблица | Markdown-таблица: чанков, средняя релевантность, длина ответа, пересечение |

### Конфигурации сравнения

- **Naive:** `RetrievalConfig(use_vector=True, top_k=5)`
- **Advanced:** `RetrievalConfig(use_hybrid=True, use_reranking=True, use_parent_child=True, top_k=5)`

### Примеры запросов

1. «Как читать файл построчно?»
2. «В чём разница между os.path и pathlib?»
3. «asyncio.gather timeout»
4. «GIL»
5. «Как использовать dataclasses?»
6. «logging basicConfig формат»

## Упражнения для студентов

1. Сравнить чанкинг по заголовкам vs фиксированный размер на запросе «Как работает система импорта Python?»
2. Попробовать «asyncio.Lock» с vector-only vs hybrid — BM25 лучше справляется с точными именами API
3. Включить реранкинг и наблюдать изменение порядка результатов для неоднозначных запросов
