"""Evaluation pipeline using RAGAS metrics on the test dataset."""

import json
import time
import uuid
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from db import EvaluationRun, get_session, init_db
from pipeline import query_pipeline

BASE_DIR = Path(__file__).parent
TEST_DATASET_PATH = BASE_DIR / "data" / "test_dataset.json"


def load_test_dataset() -> list[dict]:
    with open(TEST_DATASET_PATH) as f:
        return json.load(f)


def run_evaluation(progress_callback=None) -> dict:
    """Run the full evaluation pipeline on the test dataset.

    Args:
        progress_callback: Optional callable(current, total) for progress updates.

    Returns dict with: run_id, metrics, per_question_results
    """
    init_db()
    test_data = load_test_dataset()
    run_id = str(uuid.uuid4())[:8]

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    per_question = []

    for i, item in enumerate(test_data):
        if progress_callback:
            progress_callback(i + 1, len(test_data))

        result = query_pipeline(item["question"], use_cache=False)

        context_texts = [s["text"] for s in result["sources"]]

        questions.append(item["question"])
        answers.append(result["response_text"])
        contexts.append(context_texts)
        ground_truths.append(item["expected_answer"])

        per_question.append({
            "question": item["question"],
            "expected_sources": item.get("expected_sources", []),
            "actual_route": result["route"]["label"],
            "answer_preview": result["response_text"][:200],
            "num_sources": len(result["sources"]),
            "metrics": result["metrics"],
        })

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", max_tokens=4096))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    ragas_result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = ragas_result.to_pandas()

    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    avg_metrics = {}
    for m in metric_names:
        values = df[m].dropna()
        avg_metrics[m] = float(values.mean()) if len(values) > 0 else 0.0

    for i, row in df.iterrows():
        if i < len(per_question):
            for m in metric_names:
                per_question[i][m] = float(row.get(m, 0) or 0)

    session = get_session()
    try:
        session.add(
            EvaluationRun(
                run_id=run_id,
                avg_faithfulness=avg_metrics["faithfulness"],
                avg_answer_relevancy=avg_metrics["answer_relevancy"],
                avg_context_recall=avg_metrics["context_recall"],
                avg_context_precision=avg_metrics["context_precision"],
                per_question_json=json.dumps(per_question, default=str),
            )
        )
        session.commit()
    finally:
        session.close()

    return {
        "run_id": run_id,
        "metrics": avg_metrics,
        "per_question": per_question,
    }


def get_evaluation_history() -> list[dict]:
    """Load all past evaluation runs from the database."""
    session = get_session()
    try:
        runs = (
            session.query(EvaluationRun)
            .order_by(EvaluationRun.timestamp.desc())
            .all()
        )
        return [
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "faithfulness": r.avg_faithfulness,
                "answer_relevancy": r.avg_answer_relevancy,
                "context_recall": r.avg_context_recall,
                "context_precision": r.avg_context_precision,
                "per_question": r.get_per_question_results(),
            }
            for r in runs
        ]
    finally:
        session.close()


if __name__ == "__main__":
    print("Running evaluation on test dataset...")
    print(f"Loading {TEST_DATASET_PATH}")

    def print_progress(current, total):
        print(f"  Processing question {current}/{total}...")

    results = run_evaluation(progress_callback=print_progress)

    print(f"\nEvaluation Run: {results['run_id']}")
    print("-" * 40)
    for metric, value in results["metrics"].items():
        print(f"  {metric}: {value:.3f}")
