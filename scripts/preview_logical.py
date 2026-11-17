"""Run logical-unit chunking on the Computer Networks laboratory manual."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.logical import (  # noqa: E402
    DEFAULT_MAX_CHUNK_SIZE,
    LogicalChunk,
    _validate_chunks as _validate_logical_chunks,
    logical_unit_chunk,
)
from src.ingestion.extraction_artifact import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    load_extraction_artifact,
)

EXTRACTION_ARTIFACT = DEFAULT_ARTIFACT_PATH
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "logical_preview"
MAX_CHUNK_SIZE = DEFAULT_MAX_CHUNK_SIZE
PAGE_SEPARATOR = "\n"


def _validate_chunks(
    chunks: list[LogicalChunk],
    combined_text: str,
    max_chunk_size: int,
    source_document: str | None = None,
    page_count: int | None = None,
) -> None:
    """Validate exact reconstruction, ordering, metadata, and the size limit."""
    expected_source_document = source_document
    if expected_source_document is None and chunks:
        expected_source_document = chunks[0].source_document
    _validate_logical_chunks(
        chunks,
        combined_text,
        max_chunk_size,
        expected_source_document,
    )
    if page_count is not None:
        if (
            not isinstance(page_count, int)
            or isinstance(page_count, bool)
            or page_count < 0
        ):
            raise RuntimeError("The PDF page count must be a non-negative integer.")
        if any(
            page_number > page_count
            for chunk in chunks
            for page_number in chunk.source_page_numbers
        ):
            raise RuntimeError("A chunk references a page outside the source PDF.")


def _page_break_offsets(cleaned_pages: list[str]) -> tuple[int, ...]:
    offsets: list[int] = []
    position = 0
    for page_index, page_text in enumerate(cleaned_pages):
        position += len(page_text)
        if page_index < len(cleaned_pages) - 1:
            offsets.append(position)
            position += len(PAGE_SEPARATOR)
    return tuple(offsets)


def _write_outputs(
    output_directory: Path,
    output_contents: tuple[tuple[Path, str], ...],
) -> None:
    if output_directory.exists():
        raise FileExistsError(
            "Logical preview output directory already exists and will not be "
            f"overwritten: {output_directory}"
        )

    output_directory.mkdir(parents=True)
    created_paths: list[Path] = []
    try:
        for path, content in output_contents:
            with path.open("x", encoding="utf-8", newline="\n") as output_file:
                created_paths.append(path)
                output_file.write(content)
    except Exception as write_error:
        cleanup_errors: list[str] = []
        for path in reversed(created_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{path}: {cleanup_error}")
        if not cleanup_errors:
            try:
                output_directory.rmdir()
            except OSError as cleanup_error:
                cleanup_errors.append(f"{output_directory}: {cleanup_error}")
        if cleanup_errors:
            raise RuntimeError(
                "Preview writing failed and partial output cleanup was incomplete: "
                + "; ".join(cleanup_errors)
            ) from write_error
        raise


def main(output_directory: Path | None = None) -> None:
    """Extract, clean, chunk, validate, and save a logical-unit preview."""
    selected_output_directory = (
        OUTPUT_DIRECTORY if output_directory is None else Path(output_directory)
    )
    if (
        not isinstance(MAX_CHUNK_SIZE, int)
        or isinstance(MAX_CHUNK_SIZE, bool)
        or MAX_CHUNK_SIZE <= 0
    ):
        raise ValueError("MAX_CHUNK_SIZE must be a positive integer.")
    if selected_output_directory.exists():
        raise FileExistsError(
            "Logical preview output directory already exists and will not be "
            f"overwritten: {selected_output_directory}"
        )

    document = load_extraction_artifact(EXTRACTION_ARTIFACT)
    combined_text = document.text
    if not combined_text.strip():
        raise RuntimeError(
            "No usable text was extracted after cleaning; the preview cannot proceed."
        )

    chunks = logical_unit_chunk(
        combined_text,
        max_chunk_size=MAX_CHUNK_SIZE,
        page_break_offsets=document.page_break_offsets,
        source_document=document.source_document,
    )
    _validate_chunks(
        chunks,
        combined_text,
        MAX_CHUNK_SIZE,
        source_document=document.source_document,
        page_count=len(document.pages),
    )

    unit_chunks: dict[str, list[LogicalChunk]] = {}
    for chunk in chunks:
        unit_chunks.setdefault(chunk.logical_unit_id, []).append(chunk)

    unit_types: dict[str, int] = {}
    experiment_counts: dict[str, int] = {}
    representative_units: list[dict[str, object]] = []
    for unit_id, unit_parts in unit_chunks.items():
        first = unit_parts[0]
        unit_types[first.logical_unit_type] = (
            unit_types.get(first.logical_unit_type, 0) + 1
        )
        if (
            first.logical_unit_type == "experiment"
            and first.experiment_number is not None
        ):
            experiment_counts[first.experiment_number] = (
                experiment_counts.get(first.experiment_number, 0) + 1
            )
        if len(representative_units) < 12:
            representative_units.append(
                {
                    "logical_unit_id": unit_id,
                    "logical_unit_title": first.logical_unit_title,
                    "experiment_number": first.experiment_number,
                    "parent_heading": first.parent_heading,
                    "logical_unit_type": first.logical_unit_type,
                    "chunk_count": len(unit_parts),
                    "source_page_numbers": sorted(
                        {
                            page_number
                            for part in unit_parts
                            for page_number in part.source_page_numbers
                        }
                    ),
                }
            )

    lengths = [chunk.character_count for chunk in chunks]
    split_unit_count = sum(parts[0].is_split for parts in unit_chunks.values())
    summary = {
        "chunking_strategy": "logical_unit",
        "source_pdf_filename": document.source_document,
        "number_of_pages_extracted": len(document.pages),
        "number_of_empty_pages": len(document.empty_pages),
        "empty_page_numbers": list(document.empty_pages),
        "document_characters_after_cleaning": len(combined_text),
        "page_separator": PAGE_SEPARATOR,
        "maximum_chunk_size_characters": MAX_CHUNK_SIZE,
        "number_of_chunks": len(chunks),
        "number_of_logical_units": len(unit_chunks),
        "number_of_experiment_units": unit_types.get("experiment", 0),
        "number_of_experiment_context_units": unit_types.get(
            "experiment_context",
            0,
        ),
        "logical_unit_count_by_type": unit_types,
        "experiment_unit_count_by_number": experiment_counts,
        "number_of_split_logical_units": split_unit_count,
        "number_of_unsplit_logical_units": len(unit_chunks) - split_unit_count,
        "representative_logical_units": representative_units,
        "chunk_size_statistics_characters": {
            "average": sum(lengths) / len(lengths) if lengths else 0,
            "minimum": min(lengths, default=None),
            "maximum": max(lengths, default=None),
        },
        "average_chunk_length_characters": (
            sum(lengths) / len(lengths) if lengths else 0
        ),
        "minimum_chunk_length_characters": min(lengths, default=None),
        "maximum_chunk_length_characters": max(lengths, default=None),
        "text_preserved_exactly": "".join(chunk.text for chunk in chunks)
        == combined_text,
        "maximum_chunk_size_validated": all(
            chunk.character_count <= MAX_CHUNK_SIZE for chunk in chunks
        ),
        "page_level_provenance_available": True,
    }

    chunk_metadata = [
        {
            "chunk_index": chunk.chunk_index,
            "character_count": chunk.character_count,
            "logical_unit_id": chunk.logical_unit_id,
            "logical_unit_title": chunk.logical_unit_title,
            "experiment_number": chunk.experiment_number,
            "parent_heading": chunk.parent_heading,
            "logical_unit_type": chunk.logical_unit_type,
            "section_heading": chunk.section_heading,
            "is_split": chunk.is_split,
            "boundary_reason": chunk.boundary_reason,
            "source_document": chunk.source_document,
            "source_page_numbers": list(chunk.source_page_numbers),
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    chunks_text = "\n\n".join(
        (
            f"CHUNK {chunk.chunk_index} | {chunk.logical_unit_id} | "
            f"{chunk.logical_unit_title} | {chunk.character_count} characters\n"
            f"{chunk.text}"
        )
        for chunk in chunks
    )

    _write_outputs(
        selected_output_directory,
        (
            (selected_output_directory / "chunks.txt", chunks_text),
            (
                selected_output_directory / "summary.json",
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            ),
            (
                selected_output_directory / "chunk_metadata.json",
                json.dumps(chunk_metadata, indent=2, ensure_ascii=False) + "\n",
            ),
        )
    )

    print(f"Pages loaded: {len(document.pages)}")
    if document.empty_pages:
        print(f"Empty pages: {list(document.empty_pages)}")
    else:
        print("Empty pages: none")
    print(f"Logical units created: {len(unit_chunks)}")
    print(f"Logical chunks created: {len(chunks)}")
    print(f"Logical units split: {split_unit_count}")
    print(f"Average chunk length: {summary['average_chunk_length_characters']:.2f}")
    print(f"Maximum chunk length: {summary['maximum_chunk_length_characters']}")
    print(f"Output directory: {selected_output_directory}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a logical-unit preview of the laboratory manual."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Use a new output directory instead of the default "
            "experiments/results/logical_preview path."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().output_directory)
