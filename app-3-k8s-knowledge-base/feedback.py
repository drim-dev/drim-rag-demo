"""Feedback collection and retrieval for query quality tracking."""

from db import Feedback, Metric, get_session, init_db


def save_feedback(query_id: str, thumbs_up: bool, comment: str = ""):
    init_db()
    session = get_session()
    try:
        session.add(
            Feedback(query_id=query_id, thumbs_up=thumbs_up, comment=comment)
        )
        session.commit()
    finally:
        session.close()


def get_feedback_stats() -> dict:
    """Return aggregate feedback statistics."""
    session = get_session()
    try:
        total = session.query(Feedback).count()
        positive = session.query(Feedback).filter(Feedback.thumbs_up.is_(True)).count()
        negative = total - positive
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "ratio": positive / total if total > 0 else 0,
        }
    finally:
        session.close()


def get_metrics_summary() -> dict:
    """Return aggregate metrics from all queries."""
    session = get_session()
    try:
        from sqlalchemy import func

        result = session.query(
            func.count(Metric.id).label("total_queries"),
            func.avg(Metric.total_latency_ms).label("avg_latency"),
            func.sum(Metric.estimated_cost).label("total_cost"),
            func.avg(Metric.estimated_cost).label("avg_cost"),
        ).first()

        total = session.query(Metric).count()
        cache_hits = session.query(Metric).filter(Metric.cache_hit.is_(True)).count()

        metrics = session.query(Metric).order_by(Metric.timestamp.desc()).limit(50).all()

        return {
            "total_queries": total,
            "avg_latency_ms": float(result.avg_latency or 0),
            "total_cost": float(result.total_cost or 0),
            "avg_cost": float(result.avg_cost or 0),
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total if total > 0 else 0,
            "recent_metrics": [
                {
                    "query_id": m.query_id,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else "",
                    "total_latency_ms": m.total_latency_ms,
                    "estimated_cost": m.estimated_cost,
                    "cache_hit": m.cache_hit,
                    "latency_embedding_ms": m.latency_embedding_ms,
                    "latency_routing_ms": m.latency_routing_ms,
                    "latency_search_ms": m.latency_search_ms,
                    "latency_generation_ms": m.latency_generation_ms,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                }
                for m in metrics
            ],
        }
    finally:
        session.close()
