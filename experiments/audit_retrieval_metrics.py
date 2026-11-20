
"""Calculate overlap-aware, greedy-normalized coverage DCG.

Uses existing retrieval rankings and exact source-character evidence mappings.
Does not modify the original retrieval experiment.
"""

import json
import math
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "20261115T113239215880Z"

RETRIEVAL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "retrieval"
    / RUN_ID
    / "retrieval_results.json"
)

MAPPING_DIRECTORY = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "evidence_mapping"
    / RUN_ID
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "retrieval_metric_audit"
    / RUN_ID
)

STRATEGIES = [
    "fixed_size",
    "recursive",
    "semantic",
    "structure",
    "logical_unit",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def merge_intervals(intervals):
    """Merge overlapping or adjacent half-open intervals."""

    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        last = merged[-1]

        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def interval_length(intervals):
    """Calculate the total length of merged intervals."""

    return sum(end - start for start, end in intervals)


def marginal_gain(chunk_intervals, covered_intervals):
    """Return newly covered characters contributed by a chunk."""

    combined = merge_intervals(
        covered_intervals + chunk_intervals
    )

    previous_coverage = interval_length(covered_intervals)
    new_coverage = interval_length(combined)

    return new_coverage - previous_coverage


def build_question_evidence(mapping_data):
    """Build evidence and chunk intervals for every question."""

    question_data = {}

    for record in mapping_data["evidence_records"]:
        question_id = record["question_id"]

        if question_id not in question_data:
            question_data[question_id] = {
                "evidence_intervals": [],
                "chunk_intervals": defaultdict(list),
            }

        evidence_start = record["start_offset"]
        evidence_end = record["end_offset"]

        question_data[question_id]["evidence_intervals"].append(
            (evidence_start, evidence_end)
        )

        for chunk in record["relevant_chunks"]:
            position = chunk["chunk_position"]

            # Intersect the chunk's source interval with this
            # evidence passage.
            start = max(
                evidence_start,
                chunk["chunk_start_offset"],
            )
            end = min(
                evidence_end,
                chunk["chunk_end_offset"],
            )

            if start < end:
                question_data[question_id]["chunk_intervals"][
                    position
                ].append((start, end))

    # Merge intervals for each question and chunk.
    for question in question_data.values():
        question["evidence_intervals"] = merge_intervals(
            question["evidence_intervals"]
        )

        for position, intervals in question["chunk_intervals"].items():
            question["chunk_intervals"][position] = merge_intervals(
                intervals
            )

        question["total_evidence_characters"] = interval_length(
            question["evidence_intervals"]
        )

    return question_data


def calculate_dcg(ranked_positions, chunk_intervals, total_evidence):
    """Calculate DCG using marginal evidence gain."""

    covered = []
    dcg = 0.0
    rank_gains = []

    for rank, position in enumerate(ranked_positions, start=1):
        intervals = chunk_intervals.get(position, [])

        new_characters = marginal_gain(intervals, covered)

        gain = new_characters / total_evidence

        dcg += gain / math.log2(rank + 1)

        covered = merge_intervals(covered + intervals)

        rank_gains.append({
            "rank": rank,
            "chunk_position": position,
            "new_evidence_characters": new_characters,
            "marginal_gain": gain,
        })

    return dcg, rank_gains


def calculate_greedy_reference(
    chunk_intervals,
    total_evidence,
    max_k,
):
    """Build a greedy reference ranking by marginal evidence gain."""

    selected = set()
    covered = []
    reference_positions = []
    reference_gains = []
    reference_dcg = []

    dcg = 0.0

    for rank in range(1, max_k + 1):
        best_position = None
        best_gain = 0

        for position, intervals in chunk_intervals.items():
            if position in selected:
                continue

            gain = marginal_gain(intervals, covered)

            if gain > best_gain:
                best_gain = gain
                best_position = position

            elif (
                gain == best_gain
                and gain > 0
                and (
                    best_position is None
                    or position < best_position
                )
            ):
                best_position = position

        if best_position is None:
            break

        intervals = chunk_intervals[best_position]

        selected.add(best_position)
        covered = merge_intervals(covered + intervals)

        normalized_gain = best_gain / total_evidence

        dcg += normalized_gain / math.log2(rank + 1)

        reference_positions.append(best_position)
        reference_gains.append(normalized_gain)
        reference_dcg.append(dcg)

    return {
        "positions": reference_positions,
        "gains": reference_gains,
        "dcg_at_k": reference_dcg,
    }


def evaluate_strategy(
    strategy,
    retrieval_questions,
    question_evidence,
    k_values,
):
    """Evaluate one strategy using overlap-aware coverage DCG."""

    per_question = []

    metric_totals = {
        k: 0.0 for k in k_values
    }

    reference_totals = {
        k: 0.0 for k in k_values
    }

    for retrieval_question in retrieval_questions:
        question_id = retrieval_question["question_id"]

        if question_id not in question_evidence:
            raise ValueError(
                f"Missing evidence mapping for {strategy}, {question_id}"
            )

        evidence = question_evidence[question_id]

        total_evidence = evidence["total_evidence_characters"]

        if total_evidence <= 0:
            raise ValueError(
                f"Invalid evidence length for {question_id}"
            )

        # Verify consistency with the original retrieval artifact.
        recorded_total = retrieval_question["total_evidence_characters"]

        if recorded_total != total_evidence:
            raise ValueError(
                f"Evidence length mismatch for {strategy}, "
                f"{question_id}: retrieval={recorded_total}, "
                f"mapping={total_evidence}"
            )

        chunk_intervals = evidence["chunk_intervals"]

        ranked_positions = [
            result["chunk_position"]
            for result in retrieval_question["top_results"]
        ]

        max_k = max(k_values)

        # Calculate the actual retrieval ranking's DCG.
        actual_dcg, rank_gains = calculate_dcg(
            ranked_positions[:max_k],
            chunk_intervals,
            total_evidence,
        )

        # Calculate the greedy reference ranking.
        reference = calculate_greedy_reference(
            chunk_intervals,
            total_evidence,
            max_k,
        )

        question_metrics = {
            "question_id": question_id,
            "total_evidence_characters": total_evidence,
            "actual_rank_gains": rank_gains,
            "greedy_reference_positions": reference["positions"],
            "metrics": {},
        }

        for k in k_values:
            actual_value = sum(
                item["marginal_gain"] / math.log2(item["rank"] + 1)
                for item in rank_gains
                if item["rank"] <= k
            )

            reference_value = (
                reference["dcg_at_k"][k - 1]
                if len(reference["dcg_at_k"]) >= k
                else (
                    reference["dcg_at_k"][-1]
                    if reference["dcg_at_k"]
                    else 0.0
                )
            )

            score = (
                actual_value / reference_value
                if reference_value > 0
                else 0.0
            )

            question_metrics["metrics"][str(k)] = {
                "coverage_dcg": actual_value,
                "greedy_reference_dcg": reference_value,
                "greedy_normalized_coverage_dcg": score,
            }

            metric_totals[k] += score
            reference_totals[k] += reference_value

        per_question.append(question_metrics)

    question_count = len(per_question)

    averages = {}

    for k in k_values:
        averages[str(k)] = {
            "mean_greedy_normalized_coverage_dcg": (
                metric_totals[k] / question_count
            ),
            "mean_greedy_reference_dcg": (
                reference_totals[k] / question_count
            ),
        }

    return {
        "question_count": question_count,
        "averages": averages,
        "per_question": per_question,
    }


def main():
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIRECTORY}\n"
            "Choose a new output directory rather than overwriting results."
        )

    print("Loading retrieval results...")
    retrieval_data = load_json(RETRIEVAL_PATH)

    k_values = retrieval_data["k_values"]

    print("Loading evidence mappings...")

    mapping_by_strategy = {}

    for strategy in STRATEGIES:
        path = (
            MAPPING_DIRECTORY
            / f"{strategy}_evidence_mapping.json"
        )

        mapping_data = load_json(path)

        if mapping_data["strategy"] != strategy:
            raise ValueError(
                f"Strategy mismatch in mapping file: {path}"
            )

        mapping_by_strategy[strategy] = build_question_evidence(
            mapping_data
        )

        print(f"PASS mapping loaded: {strategy}")

    results = {
        "run_id": retrieval_data["run_id"],
        "model_name": retrieval_data["model_name"],
        "metric": "greedy_normalized_coverage_dcg",
        "description": (
            "DCG based on marginal source-evidence characters, "
            "normalized by a greedy reference ranking."
        ),
        "normalization_note": (
            "The greedy reference is a heuristic, not a guaranteed "
            "globally optimal ranking. Scores may exceed 1."
        ),
        "k_values": k_values,
        "strategies": {},
    }

    print("\nEvaluating strategies...")

    for strategy in STRATEGIES:
        print(f"Evaluating {strategy}...")

        retrieval_questions = retrieval_data["strategies"][strategy]

        strategy_results = evaluate_strategy(
            strategy,
            retrieval_questions,
            mapping_by_strategy[strategy],
            k_values,
        )

        results["strategies"][strategy] = strategy_results

        print(f"PASS {strategy}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)

    output_path = OUTPUT_DIRECTORY / "coverage_aware_metrics.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nEvaluation completed.")
    print(f"Results saved to: {output_path}")

    print("\nGREEDY-NORMALIZED COVERAGE DCG")

    for strategy in STRATEGIES:
        print(f"\n{strategy.upper()}")

        averages = results["strategies"][strategy]["averages"]

        for k in k_values:
            score = averages[str(k)][
                "mean_greedy_normalized_coverage_dcg"
            ]

            print(f"K={k:<3} {score:.4f}")


if __name__ == "__main__":
    main()
