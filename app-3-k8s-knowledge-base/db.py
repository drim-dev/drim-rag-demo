"""Database schema and session management for metrics, feedback, evaluation, and cache."""

import json
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import EMBEDDING_DIM, POSTGRES_CONNECTION_STRING

Base = declarative_base()


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    latency_embedding_ms = Column(Float, default=0)
    latency_routing_ms = Column(Float, default=0)
    latency_search_ms = Column(Float, default=0)
    latency_generation_ms = Column(Float, default=0)
    total_latency_ms = Column(Float, default=0)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0)
    cache_hit = Column(Boolean, default=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, nullable=False, index=True)
    thumbs_up = Column(Boolean, nullable=False)
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, unique=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    avg_faithfulness = Column(Float, default=0)
    avg_answer_relevancy = Column(Float, default=0)
    avg_context_relevancy = Column(Float, default=0)
    avg_context_precision = Column(Float, default=0)
    per_question_json = Column(Text, default="{}")

    def get_per_question_results(self) -> list[dict]:
        return json.loads(self.per_question_json) if self.per_question_json else []


class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    original_query = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    sources_json = Column(Text, default="[]")
    route_label = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ContentHash(Base):
    """Tracks file content hashes to enable incremental reindexing."""
    __tablename__ = "content_hashes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False, unique=True)
    content_hash = Column(String, nullable=False)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


engine = create_engine(POSTGRES_CONNECTION_STRING)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
