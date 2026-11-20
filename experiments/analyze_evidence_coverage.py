
"""Analyze how well each chunking strategy preserves annotated evidence."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "20261115T113239215880Z"

INPUT_DIRECTORY = (
    ROOT / "experiments" / "results" / "evidence_mapping" / RUN_ID
)

OUTPUT_DIRECTORY = (
    ROOT / "experiments" / "results" / "evidence_coverage" / RUN_ID
)

STRATEGIES = [
    "fixed_size",
    "recursive",
    "semantic",
    "structure",
    "logical_unit",
]


def merge_intervals(intervals):
    """Merge overlapping or adjacent half-open intervals."""

    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return merged


def calculate_union_coverage(evidence_start, evidence_end, chunks):
    """Calculate unique evidence characters covered by relevant chunks."""

    clipped_intervals = []

    for chunk in chunks:
        start = max(evidence_start, chunk["chunk_start_offset"])
        end = min(evidence_end, chunk["chunk_end_offset"])

        if start < end:
            clipped_intervals.append((start, end))

    merged = merge_intervals(clipped_intervals)

    covered_characters = sum(
        end - start for start, end in merged
    )

    evidence_length = evidence_end - evidence_start

    return {
        "covered_characters": covered_characters,
        "coverage": covered_characters / evidence_length,
        "merged_intervals": merged,
    }


def calculate_minimum_chunks(evidence_start, evidence_end, chunks):
    """
    Find the minimum number of chunk intervals needed to cover
    the complete evidence interval.

    Returns None if the intervals contain a gap.
    """

    intervals = sorted(
        (
            chunk["chunk_start_offset"],
            chunk["chunk_end_offset"],
        )
        for chunk in chunks
    )

    current_position = evidence_start
    selected_count = 0

    while current_position < evidence_end:
        furthest_end = current_position

        for start, end in intervals:
            if start <= current_position < end:
                furthest_end = max(furthest_end, end)

        if furthest_end == current_position:
            return None

        current_position = min(furthest_end, evidence_end)
        selected_count += 1

    return selected_count


def analyze_evidence(record):
    """Calculate coverage and fragmentation metrics for one passage."""

    evidence_start = record["start_offset"]
    evidence_end = record["end_offset"]
    evidence_length = record["evidence_length"]

    chunks = record["relevant_chunks"]

    union_result = calculate_union_coverage(
        evidence_start,
        evidence_end,
        chunks,
    )

    individual_coverages = [
        chunk["overlap_characters"] / evidence_length
        for chunk in chunks
    ]

    single_chunk_containment = any(
        chunk["chunk_start_offset"] <= evidence_start
        and chunk["chunk_end_offset"] >= evidence_end
        for chunk in chunks
    )

    minimum_chunks = calculate_minimum_chunks(
        evidence_start,
        evidence_end,
        chunks,
    )

    return {
        "question_id": record["question_id"],
        "evidence_position": record["evidence_position"],
        "pdf_page": record["pdf_page"],
        "relevance": record["relevance"],
        "evidence_length": evidence_length,
        "number_of_relevant_chunks": len(chunks),
        "union_covered_characters": union_result["covered_characters"],
        "union_evidence_coverage": union_result["coverage"],
        "max_single_chunk_evidence_coverage": max(
            individual_coverages,
            default=0.0,
        ),
        "single_chunk_contains_full_evidence": single_chunk_containment,
        "minimum_chunks_needed": minimum_chunks,
        "evidence_fully_covered": (
            union_result["covered_characters"] == evidence_length
        ),
        "merged_coverage_intervals": union_result["merged_intervals"],
    }


def analyze_strategy(strategy):
    """Analyze all evidence passages for one chunking strategy."""

    input_path = (
        INPUT_DIRECTORY / f"{strategy}_evidence_mapping.json"
    )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    evidence_records = data["evidence_records"]

    detailed_records = [
        analyze_evidence(record)
        for record in evidence_records
    ]

    total = len(detailed_records)

    if total == 0:
        raise ValueError(f"No evidence records found for {strategy}")

    union_coverages = [
        record["union_evidence_coverage"]
        for record in detailed_records
    ]

    max_coverages = [
        record["max_single_chunk_evidence_coverage"]
        for record in detailed_records
    ]

    chunk_counts = [
        record["number_of_relevant_chunks"]
        for record in detailed_records
    ]

    contained_count = sum(
        record["single_chunk_contains_full_evidence"]
        for record in detailed_records
    )

    fully_covered_count = sum(
        record["evidence_fully_covered"]
        for record in detailed_records
    )

    minimum_chunk_counts = [
        record["minimum_chunks_needed"]
        for record in detailed_records
        if record["minimum_chunks_needed"] is not None
    ]

    summary = {
        "strategy": strategy,
        "evidence_passages": total,
        "average_union_evidence_coverage": mean(union_coverages),
        "minimum_union_evidence_coverage": min(union_coverages),
        "maximum_union_evidence_coverage": max(union_coverages),
        "average_max_single_chunk_coverage": mean(max_coverages),
        "average_relevant_chunks_per_evidence": mean(chunk_counts),
        "single_chunk_containment_count": contained_count,
        "single_chunk_containment_rate": contained_count / total,
        "fully_covered_passage_count": fully_covered_count,
        "fully_covered_passage_rate": fully_covered_count / total,
        "average_minimum_chunks_needed": (
            mean(minimum_chunk_counts)
            if minimum_chunk_counts
            else None
        ),
        "passages_with_coverage_gaps": total - fully_covered_count,
    }

    return summary, detailed_records


def main():
    if not INPUT_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Evidence mapping directory not found: {INPUT_DIRECTORY}"
        )

    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIRECTORY}\n"
            "Remove it or choose a new output location before rerunning."
        )

    OUTPUT_DIRECTORY.mkdir(parents=True)

    summaries = []
    detailed_results = {}

    for strategy in STRATEGIES:
        summary, details = analyze_strategy(strategy)

        summaries.append(summary)
        detailed_results[strategy] = details

        print(
            f"PASS {strategy}: "
            f"{summary['evidence_passages']} passages, "
            f"average union coverage = "
            f"{summary['average_union_evidence_coverage']:.2%}, "
            f"single-chunk containment = "
            f"{summary['single_chunk_containment_rate']:.2%}"
        )

    summary_output = {
        "run_id": RUN_ID,
        "analysis_method": "merged_character_interval_coverage",
        "strategies": summaries,
    }

    detailed_output = {
        "run_id": RUN_ID,
        "analysis_method": "merged_character_interval_coverage",
        "strategies": detailed_results,
    }

    summary_path = OUTPUT_DIRECTORY / "evidence_coverage_summary.json"
    details_path = OUTPUT_DIRECTORY / "evidence_coverage_by_strategy.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_output, file, indent=2)

    with details_path.open("w", encoding="utf-8") as file:
        json.dump(detailed_output, file, indent=2)

    print("\nAnalysis completed.")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


if __name__ == "__main__":
    main()
