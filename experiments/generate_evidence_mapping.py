
"""Map annotated question evidence to exact source-preserving chunks."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.extraction_artifact import load_extraction_artifact


RUN_ID = "20261115T113239215880Z"

ARTIFACT_PATH = (
    ROOT / "data" / "processed" / "extracted" / "extracted_pages.json"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "questions_with_line_numbers.json"
)

CHUNK_DIRECTORY = (
    ROOT / "experiments" / "results" / "chunk_statistics" / RUN_ID
)

OUTPUT_DIRECTORY = (
    ROOT / "experiments" / "results" / "evidence_mapping" / RUN_ID
)

FIXED_SIZE_OVERLAP = 150

STRATEGIES = {
    "fixed_size": "fixed_size_chunks.json",
    "recursive": "recursive_chunks.json",
    "semantic": "semantic_chunks.json",
    "structure": "structure_chunks.json",
    "logical_unit": "logical_unit_chunks.json",
}


def normalize(text: str) -> str:
    """Normalize whitespace in the same way as the evidence audit."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize whitespace while retaining original character positions."""
    normalized_chars = []
    original_positions = []

    for position, char in enumerate(text):
        if char.isspace():
            if normalized_chars and normalized_chars[-1] != " ":
                normalized_chars.append(" ")
                original_positions.append(position)
        else:
            normalized_chars.append(char)
            original_positions.append(position)

    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        original_positions.pop()

    return "".join(normalized_chars), original_positions


def locate_evidence(document, evidence):
    """Convert one page-and-line annotation into exact document offsets."""
    page_number = evidence.get("pdf_page")
    line_start = evidence.get("line_start")
    line_end = evidence.get("line_end")
    evidence_text = evidence.get("evidence_text")

    if not isinstance(page_number, int) or isinstance(page_number, bool):
        return {"status": "invalid_page_number"}

    if (
        not isinstance(line_start, int)
        or isinstance(line_start, bool)
        or not isinstance(line_end, int)
        or isinstance(line_end, bool)
    ):
        return {"status": "missing_or_invalid_line_numbers"}

    if not isinstance(evidence_text, str) or not evidence_text.strip():
        return {"status": "missing_evidence_text"}

    if page_number < 1 or page_number > len(document.pages):
        return {"status": "page_not_found"}

    page = document.pages[page_number - 1]

    lines = page.text.splitlines()
    lines_with_endings = page.text.splitlines(keepends=True)

    if (
        line_start < 1
        or line_end < line_start
        or line_end > len(lines)
    ):
        return {"status": "invalid_line_range"}

    # Calculate the exact character range occupied by the annotated lines.
    local_start = sum(
        len(line) for line in lines_with_endings[: line_start - 1]
    )

    local_end = sum(
        len(line) for line in lines_with_endings[:line_end]
    )

    selected_text = page.text[local_start:local_end]

    expected_text = normalize(evidence_text)
    normalized_selected, original_positions = normalize_with_offsets(
        selected_text
    )

    # Preserve the validation rule used by audit_question_evidence.py.
    if expected_text not in normalized_selected:
        return {"status": "evidence_text_mismatch"}

    # Find the exact occurrence inside the annotated line range.
    match_positions = []
    search_from = 0

    while True:
        match_position = normalized_selected.find(
            expected_text,
            search_from,
        )

        if match_position == -1:
            break

        match_positions.append(match_position)
        search_from = match_position + 1

    if len(match_positions) != 1:
        return {
            "status": "ambiguous_evidence_text",
            "number_of_matches": len(match_positions),
        }

    match_position = match_positions[0]
    match_end = match_position + len(expected_text)

    # Convert normalized positions back to original page-text positions.
    exact_local_start = (
        local_start + original_positions[match_position]
    )

    exact_local_end = (
        local_start + original_positions[match_end - 1] + 1
    )

    page_span = document.page_spans[page_number - 1]

    global_start = page_span.start_offset + exact_local_start
    global_end = page_span.start_offset + exact_local_end

    actual_text = document.text[global_start:global_end]

    if normalize(actual_text) != expected_text:
        return {"status": "source_offset_validation_failed"}

    return {
        "status": "validated",
        "pdf_page": page_number,
        "line_start": line_start,
        "line_end": line_end,
        "start_offset": global_start,
        "end_offset": global_end,
        "evidence_length": global_end - global_start,
        "evidence_text": evidence_text,
        "relevance": evidence.get("relevance"),
    }



def load_chunks(strategy, filename, document):
    """Load chunk records and calculate their exact source spans."""
    path = CHUNK_DIRECTORY / filename

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {filename}")

    if data.get("status") != "completed":
        raise ValueError(
            f"Chunking run for {strategy} was not completed: "
            f"{data.get('status')}"
        )

    chunks = data.get("chunks")

    if not isinstance(chunks, list):
        raise ValueError(f"Expected a 'chunks' list in {filename}")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Invalid chunk record at position {index} in {filename}"
            )

        if not isinstance(chunk.get("text"), str):
            raise ValueError(
                f"Missing or invalid chunk text at position {index} "
                f"in {filename}"
            )

    texts = [chunk["text"] for chunk in chunks]

    if strategy == "fixed_size":
        overlap_lengths = (
            [0] + [FIXED_SIZE_OVERLAP] * (len(texts) - 1)
            if texts else []
        )

    elif strategy == "recursive":
        overlap_lengths = []

        for index, chunk in enumerate(chunks):
            overlap = chunk.get("overlap_length")

            if not isinstance(overlap, int) or overlap < 0:
                raise ValueError(
                    f"Invalid overlap_length in recursive chunk {index}"
                )

            overlap_lengths.append(overlap)

    else:
        overlap_lengths = None

    spans = document.chunk_spans_for_chunks(
        texts,
        overlap_lengths,
    )

    return chunks, spans


def map_evidence_to_chunks(evidence_record, chunks, spans):
    """Calculate evidence overlap with every chunk."""
    if evidence_record["status"] != "validated":
        return []

    evidence_start = evidence_record["start_offset"]
    evidence_end = evidence_record["end_offset"]
    evidence_length = evidence_end - evidence_start

    relevant_chunks = []

    for position, (chunk, span) in enumerate(zip(chunks, spans), start=1):
        overlap_start = max(evidence_start, span.start_offset)
        overlap_end = min(evidence_end, span.end_offset)

        overlap_characters = max(0, overlap_end - overlap_start)

        if overlap_characters == 0:
            continue

        chunk_length = span.end_offset - span.start_offset

        relevant_chunks.append(
            {
                "chunk_position": position,
                "chunk_index": chunk.get("chunk_index"),
                "logical_unit_id": chunk.get("logical_unit_id"),
                "overlap_characters": overlap_characters,
                "evidence_coverage": (
                    overlap_characters / evidence_length
                ),
                "chunk_coverage": (
                    overlap_characters / chunk_length
                    if chunk_length
                    else 0
                ),
                "chunk_start_offset": span.start_offset,
                "chunk_end_offset": span.end_offset,
                "chunk_source_pages": list(
                    span.source_page_numbers
                ),
            }
        )

    return relevant_chunks


def main():
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIRECTORY}"
        )

    document = load_extraction_artifact(ARTIFACT_PATH)

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # Convert every annotated evidence passage into source character offsets.
    evidence_records = []

    for question in dataset["questions"]:
        for evidence_position, evidence in enumerate(
            question["evidence"],
            start=1,
        ):
            located = locate_evidence(document, evidence)

            evidence_records.append(
                {
                    "question_id": question["question_id"],
                    "evidence_position": evidence_position,
                    "question": question["question"],
                    "question_type": question.get("question_type"),
                    "expected_answer": question.get("expected_answer"),
                    **located,
                }
            )

    OUTPUT_DIRECTORY.mkdir(parents=True)

    summary = {
        "run_id": RUN_ID,
        "source_document": document.source_document,
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "annotation_status": dataset.get("annotation_status"),
        "total_questions": len(dataset["questions"]),
        "total_evidence_passages": len(evidence_records),
        "validated_evidence_passages": sum(
            record["status"] == "validated"
            for record in evidence_records
        ),
        "failed_evidence_passages": sum(
            record["status"] != "validated"
            for record in evidence_records
        ),
        "strategies": {},
    }

    for strategy, filename in STRATEGIES.items():
        chunks, spans = load_chunks(strategy, filename, document)

        strategy_records = []

        for evidence_record in evidence_records:
            relevant_chunks = map_evidence_to_chunks(
                evidence_record,
                chunks,
                spans,
            )

            strategy_records.append(
                {
                    **evidence_record,
                    "relevant_chunks": relevant_chunks,
                    "number_of_relevant_chunks": len(relevant_chunks),
                }
            )

        output_path = (
            OUTPUT_DIRECTORY / f"{strategy}_evidence_mapping.json"
        )

        output_path.write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "run_id": RUN_ID,
                    "mapping_method": "exact_source_character_overlap",
                    "range_convention": "half-open [start_offset, end_offset)",
                    "relevance_definition": (
                        "A chunk is relevant when it has positive "
                        "character overlap with an annotated evidence span."
                    ),
                    "evidence_records": strategy_records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        mapped_count = sum(
            record["status"] == "validated"
            and record["number_of_relevant_chunks"] > 0
            for record in strategy_records
        )

        summary["strategies"][strategy] = {
            "number_of_chunks": len(chunks),
            "evidence_passages_with_matching_chunks": mapped_count,
            "evidence_passages_without_matching_chunks": (
                len(strategy_records) - mapped_count
            ),
            "output_file": output_path.name,
        }

        print(
            f"PASS {strategy}: "
            f"{len(chunks)} chunks, "
            f"{mapped_count}/{len(strategy_records)} evidence passages mapped"
        )

    summary_path = OUTPUT_DIRECTORY / "summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n========== EVIDENCE MAPPING SUMMARY ==========")
    print(f"Questions: {summary['total_questions']}")
    print(f"Evidence passages: {summary['total_evidence_passages']}")
    print(f"Validated: {summary['validated_evidence_passages']}")
    print(f"Failed: {summary['failed_evidence_passages']}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
