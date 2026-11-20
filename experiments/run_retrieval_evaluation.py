
"""Evaluate retrieval quality across the five chunking strategies.

Uses existing chunk embeddings and exact source-character evidence mappings.
The original embedding artifacts are not modified.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "20261115T113239215880Z"

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

EMBEDDING_DIR = RESULTS_DIR / "embeddings" / RUN_ID
MAPPING_DIR = RESULTS_DIR / "evidence_mapping" / RUN_ID

OUTPUT_DIR = RESULTS_DIR / "retrieval" / RUN_ID

STRATEGIES = [
    "fixed_size",
    "recursive",
    "semantic",
    "structure",
    "logical_unit",
]

K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)

# Initial relevance thresholds.
CHUNK_RELEVANCE_THRESHOLD = 0.10
EVIDENCE_HIT_THRESHOLD = 0.50


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def merge_intervals(intervals):
    """Merge overlapping or adjacent half-open intervals."""

    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    return merged


def interval_length(intervals):
    return sum(
        end - start
        for start, end in merge_intervals(intervals)
    )


def cosine_similarity_matrix(query_vectors, chunk_vectors):
    """Calculate cosine similarity between normalized vectors."""

    query_norms = np.linalg.norm(
        query_vectors, axis=1, keepdims=True
    )
    chunk_norms = np.linalg.norm(
        chunk_vectors, axis=1, keepdims=True
    )

    query_vectors = query_vectors / np.maximum(query_norms, 1e-12)
    chunk_vectors = chunk_vectors / np.maximum(chunk_norms, 1e-12)

    return query_vectors @ chunk_vectors.T


# ---------------------------------------------------------
# Load and validate evaluation questions
# ---------------------------------------------------------

def load_questions():
    """Use fixed-size mapping as the canonical question/evidence source."""

    path = MAPPING_DIR / "fixed_size_evidence_mapping.json"
    data = load_json(path)

    questions = {}

    for record in data["evidence_records"]:
        question_id = record["question_id"]

        if question_id not in questions:
            questions[question_id] = {
                "question_id": question_id,
                "question": record["question"],
                "expected_answer": record["expected_answer"],
                "evidence_intervals": [],
            }

        question = questions[question_id]

        if question["question"] != record["question"]:
            raise ValueError(
                f"Inconsistent question text: {question_id}"
            )

        question["evidence_intervals"].append(
            (record["start_offset"], record["end_offset"])
        )

    for question in questions.values():
        question["evidence_intervals"] = merge_intervals(
            question["evidence_intervals"]
        )

        question["total_evidence_characters"] = interval_length(
            question["evidence_intervals"]
        )

        if question["total_evidence_characters"] <= 0:
            raise ValueError(
                f"Empty evidence: {question['question_id']}"
            )

    return questions


# ---------------------------------------------------------
# Validate evidence mappings across strategies
# ---------------------------------------------------------

def validate_mapping_consistency(questions):
    """Ensure every strategy uses the same question evidence spans."""

    expected = {
        question_id: intervals
        for question_id, intervals in (
            (qid, q["evidence_intervals"])
            for qid, q in questions.items()
        )
    }

    for strategy in STRATEGIES:
        path = MAPPING_DIR / f"{strategy}_evidence_mapping.json"
        data = load_json(path)

        actual = defaultdict(list)

        for record in data["evidence_records"]:
            actual[record["question_id"]].append(
                (
                    record["start_offset"],
                    record["end_offset"],
                )
            )

        actual = {
            qid: merge_intervals(intervals)
            for qid, intervals in actual.items()
        }

        if actual != expected:
            raise ValueError(
                f"Evidence mapping mismatch for {strategy}"
            )

        print(f"PASS mapping consistency: {strategy}")


# ---------------------------------------------------------
# Build question-specific chunk evidence intervals
# ---------------------------------------------------------

def build_chunk_evidence_intervals(mapping_data):
    """
    Map each question and chunk position to the exact source
    character intervals that the chunk covers.
    """

    result = defaultdict(lambda: defaultdict(list))

    for record in mapping_data["evidence_records"]:
        question_id = record["question_id"]

        evidence_start = record["start_offset"]
        evidence_end = record["end_offset"]

        for relevant in record["relevant_chunks"]:
            position = relevant["chunk_position"]

            chunk_start = relevant["chunk_start_offset"]
            chunk_end = relevant["chunk_end_offset"]

            intersection_start = max(
                evidence_start, chunk_start
            )
            intersection_end = min(
                evidence_end, chunk_end
            )

            if intersection_start < intersection_end:
                result[question_id][position].append(
                    (intersection_start, intersection_end)
                )

    return result


# ---------------------------------------------------------
# Evidence metrics
# ---------------------------------------------------------

def calculate_evidence_coverage(
    question,
    retrieved_positions,
    chunk_evidence,
):
    """Calculate unique evidence coverage for retrieved chunks."""

    intervals = []

    for position in retrieved_positions:
        intervals.extend(
            chunk_evidence.get(position, [])
        )

    covered_characters = interval_length(intervals)

    total_characters = question["total_evidence_characters"]

    return min(covered_characters / total_characters, 1.0)


def calculate_chunk_relevance_scores(
    question,
    number_of_chunks,
    chunk_evidence,
):
    """
    Each chunk receives a graded relevance score:

    unique evidence characters in that chunk /
    total unique evidence characters for the question.
    """

    total_characters = question["total_evidence_characters"]

    scores = {}

    for position in range(1, number_of_chunks + 1):
        covered = interval_length(
            chunk_evidence.get(position, [])
        )

        scores[position] = min(
            covered / total_characters, 1.0
        )

    return scores


# ---------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------

def calculate_mrr_at_k(ranked_positions, relevance_scores, k):
    """Find the first meaningfully relevant chunk within top K."""

    for rank, position in enumerate(ranked_positions[:k], start=1):
        if (
            relevance_scores[position]
            >= CHUNK_RELEVANCE_THRESHOLD
        ):
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    ranked_positions,
    relevance_scores,
    k,
):
    """Calculate nDCG using continuous graded relevance."""

    def dcg(scores):
        return sum(
            (2 ** score - 1) / np.log2(rank + 1)
            for rank, score in enumerate(scores, start=1)
        )

    actual_scores = [
        relevance_scores[position]
        for position in ranked_positions[:k]
    ]

    ideal_scores = sorted(
        relevance_scores.values(),
        reverse=True,
    )[:k]

    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return dcg(actual_scores) / ideal_dcg


# ---------------------------------------------------------
# Load embedding artifacts
# ---------------------------------------------------------

def load_embedding_artifact(strategy):
    path = EMBEDDING_DIR / f"{strategy}_embeddings.json"
    data = load_json(path)

    if data["model_name"] != MODEL_NAME:
        raise ValueError(
            f"Model mismatch in {strategy}: "
            f"{data['model_name']}"
        )

    chunks = data["chunks"]

    if len(chunks) != data["total_chunks"]:
        raise ValueError(
            f"Chunk count mismatch: {strategy}"
        )

    if not chunks:
        raise ValueError(f"No chunks found: {strategy}")

    vectors = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(
            f"Invalid embedding matrix: {strategy}"
        )

    if vectors.shape[1] != data["embedding_dimensions"]:
        raise ValueError(
            f"Dimension mismatch: {strategy}"
        )

    # Position is one-based and refers to list order.
    for position, chunk in enumerate(chunks, start=1):
        if chunk["chunk_index"] != position:
            raise ValueError(
                f"Unexpected chunk index in {strategy}, "
                f"position {position}"
            )

    return data, chunks, vectors


# ---------------------------------------------------------
# Main retrieval evaluation
# ---------------------------------------------------------

def main():
    global MODEL_NAME

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Move or rename the previous experiment before rerunning."
        )

    OUTPUT_DIR.mkdir(parents=True)

    print("Loading evaluation questions...")

    questions = load_questions()

    print("Questions:", len(questions))

    validate_mapping_consistency(questions)

    ordered_questions = list(questions.values())
    question_texts = [
        question["question"]
        for question in ordered_questions
    ]

    # Load the common model name from the first artifact.
    first_artifact = load_json(
        EMBEDDING_DIR / "fixed_size_embeddings.json"
    )

    MODEL_NAME = first_artifact["model_name"]

    print("\nEmbedding model:", MODEL_NAME)
    print("Loading model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model maximum sequence length:", model.max_seq_length)

    print("\nEmbedding evaluation questions...")

    query_vectors = model.encode(
        question_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    retrieval_results = {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "query_count": len(questions),
        "k_values": K_VALUES,
        "strategies": {},
    }

    metrics_output = {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "query_count": len(questions),
        "configuration": {
            "k_values": K_VALUES,
            "similarity": "cosine_similarity",
            "chunk_relevance_threshold": (
                CHUNK_RELEVANCE_THRESHOLD
            ),
            "evidence_hit_threshold": (
                EVIDENCE_HIT_THRESHOLD
            ),
            "embedding_experiment": "original_chunk_embeddings",
        },
        "strategies": {},
    }

    # -----------------------------------------------------
    # Evaluate each strategy
    # -----------------------------------------------------

    for strategy in STRATEGIES:
        print(f"\nEvaluating: {strategy}")

        embedding_data, chunks, chunk_vectors = (
            load_embedding_artifact(strategy)
        )

        mapping_data = load_json(
            MAPPING_DIR / f"{strategy}_evidence_mapping.json"
        )

        chunk_evidence = build_chunk_evidence_intervals(
            mapping_data
        )

        similarity_matrix = cosine_similarity_matrix(
            query_vectors,
            chunk_vectors,
        )

        strategy_results = []
        metric_observations = {
            k: {
                "coverage": [],
                "hit": [],
                "mrr": [],
                "ndcg": [],
            }
            for k in K_VALUES
        }

        for question_index, question in enumerate(
            ordered_questions
        ):
            question_id = question["question_id"]

            similarities = similarity_matrix[question_index]

            ranked_positions = (
                np.argsort(-similarities, kind="stable") + 1
            ).tolist()

            relevance_scores = calculate_chunk_relevance_scores(
                question,
                len(chunks),
                chunk_evidence[question_id],
            )

            top_results = []

            for rank, position in enumerate(
                ranked_positions[:MAX_K],
                start=1,
            ):
                chunk = chunks[position - 1]

                top_results.append({
                    "rank": rank,
                    "chunk_position": position,
                    "chunk_index": chunk["chunk_index"],
                    "cosine_similarity": float(
                        similarities[position - 1]
                    ),
                    "chunk_relevance_score": (
                        relevance_scores[position]
                    ),
                    "text": chunk["text"],
                })

            for k in K_VALUES:
                selected_positions = ranked_positions[:k]

                coverage = calculate_evidence_coverage(
                    question,
                    selected_positions,
                    chunk_evidence[question_id],
                )

                hit = coverage >= EVIDENCE_HIT_THRESHOLD

                mrr = calculate_mrr_at_k(
                    ranked_positions,
                    relevance_scores,
                    k,
                )

                ndcg = calculate_ndcg_at_k(
                    ranked_positions,
                    relevance_scores,
                    k,
                )

                metric_observations[k]["coverage"].append(
                    coverage
                )
                metric_observations[k]["hit"].append(
                    int(hit)
                )
                metric_observations[k]["mrr"].append(mrr)
                metric_observations[k]["ndcg"].append(ndcg)

            strategy_results.append({
                "question_id": question_id,
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "total_evidence_characters": (
                    question["total_evidence_characters"]
                ),
                "top_results": top_results,
            })

        retrieval_results["strategies"][strategy] = (
            strategy_results
        )

        # -------------------------------------------------
        # Aggregate metrics
        # -------------------------------------------------

        strategy_metrics = {
            "question_count": len(questions),
            "total_chunks": len(chunks),
            "metrics_at_k": {},
        }

        for k in K_VALUES:
            observations = metric_observations[k]

            strategy_metrics["metrics_at_k"][str(k)] = {
                "evidence_coverage_at_k": float(
                    np.mean(observations["coverage"])
                ),
                "evidence_hit_rate_at_k": float(
                    np.mean(observations["hit"])
                ),
                "mrr_at_k": float(
                    np.mean(observations["mrr"])
                ),
                "ndcg_at_k": float(
                    np.mean(observations["ndcg"])
                ),
            }

        metrics_output["strategies"][strategy] = (
            strategy_metrics
        )

        print(
            f"PASS {strategy}: "
            f"{len(chunks)} chunks evaluated"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_path = OUTPUT_DIR / "retrieval_results.json"
    metrics_path = OUTPUT_DIR / "retrieval_metrics.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(
            retrieval_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nRetrieval evaluation completed.")

    print("Results:", results_path)
    print("Metrics:", metrics_path)

    print("\nEvidence Coverage@5:")

    for strategy in STRATEGIES:
        value = metrics_output["strategies"][strategy][
            "metrics_at_k"
        ]["5"]["evidence_coverage_at_k"]

        print(f"{strategy}: {value * 100:.2f}%")


if __name__ == "__main__":
    main()
