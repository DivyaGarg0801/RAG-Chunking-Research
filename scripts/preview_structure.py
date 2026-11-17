"""Run the structure-aware preview on the local laboratory manual."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.structure import StructureChunk, _detect_heading, structure_chunk
from src.ingestion.extraction_artifact import (
    DEFAULT_ARTIFACT_PATH,
    load_extraction_artifact,
)

EXTRACTION_ARTIFACT = DEFAULT_ARTIFACT_PATH
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "structure_based_preview_v2_corrected"
)
MAX_CHUNK_SIZE = 1500
PAGE_SEPARATOR = "\n"


def _validate_chunks(
    chunks: list[StructureChunk],
    combined_text: str,
    max_chunk_size: int,
) -> None:
    """Validate chunk metadata, reconstruction, and configured size limits."""
    if not chunks:
        if combined_text:
            raise RuntimeError("The structure-based preview produced no chunks.")
        return

    if any(not chunk.text for chunk in chunks):
        raise RuntimeError("One or more generated chunks are empty.")

    if any(len(chunk.text) > max_chunk_size for chunk in chunks):
        raise RuntimeError("A generated chunk exceeds the configured size limit.")

    reconstructed_text = "".join(chunk.text for chunk in chunks)
    if reconstructed_text != combined_text:
        raise RuntimeError(
            "The structure-based preview did not reconstruct the document text exactly."
        )

    if any(
        not isinstance(chunk.chunk_index, int)
        or isinstance(chunk.chunk_index, bool)
        for chunk in chunks
    ):
        raise RuntimeError("Chunk indexes must be integers.")

    chunk_indices = [chunk.chunk_index for chunk in chunks]
    if chunk_indices != list(range(1, len(chunks) + 1)):
        raise RuntimeError("Chunk indexes are not sequential.")

    for chunk in chunks:
        if (
            not isinstance(chunk.heading_level, int)
            or isinstance(chunk.heading_level, bool)
            or chunk.heading_level not in (0, 1, 2, 3)
        ):
            raise RuntimeError("A generated chunk has an invalid heading level.")
        if chunk.heading is None and chunk.heading_level != 0:
            raise RuntimeError(
                "An unheaded chunk must have heading level 0."
            )
        if chunk.heading is None and (
            chunk.parent_heading is not None
            or chunk.section_path
            or chunk.heading_occurrence is not None
        ):
            raise RuntimeError("An unheaded chunk cannot have heading metadata.")
        if chunk.heading is not None and chunk.heading_level == 0:
            raise RuntimeError("A headed chunk must have a positive heading level.")
        if chunk.heading is not None and not chunk.section_path:
            raise RuntimeError(
                "A headed chunk must have a non-empty section path."
            )
        if chunk.heading is not None and any(
            not isinstance(path_heading, str) or not path_heading
            for path_heading in chunk.section_path
        ):
            raise RuntimeError("A section path must contain non-empty headings.")
        if chunk.heading is not None and chunk.section_path[-1] != chunk.heading:
            raise RuntimeError("A headed chunk has an inconsistent section path.")
        if chunk.heading is not None:
            if chunk.parent_heading is None and len(chunk.section_path) != 1:
                raise RuntimeError(
                    "A root heading must have a one-item section path."
                )
            if chunk.parent_heading is None and chunk.heading_level == 3:
                raise RuntimeError("A level-three heading must have a parent.")
            if chunk.parent_heading is not None:
                if (
                    len(chunk.section_path) != chunk.heading_level
                    or len(chunk.section_path) < 2
                    or chunk.section_path[-2] != chunk.parent_heading
                ):
                    raise RuntimeError(
                        "A headed chunk has an inconsistent parent and section path."
                    )
        if chunk.heading is not None and (
            not isinstance(chunk.heading_occurrence, int)
            or isinstance(chunk.heading_occurrence, bool)
            or chunk.heading_occurrence <= 0
        ):
            raise RuntimeError("A headed chunk must have a positive occurrence ID.")

    occurrence_metadata: dict[
        int, tuple[str | None, int, str | None, tuple[str, ...]]
    ] = {}
    occurrence_text: dict[int, list[str]] = {}
    closed_occurrences: set[int] = set()
    previous_occurrence: int | None = None
    for chunk in chunks:
        if chunk.heading_occurrence is None:
            if previous_occurrence is not None:
                closed_occurrences.add(previous_occurrence)
                previous_occurrence = None
            continue
        if chunk.heading_occurrence != previous_occurrence:
            if previous_occurrence is not None:
                closed_occurrences.add(previous_occurrence)
            if chunk.heading_occurrence in closed_occurrences:
                raise RuntimeError(
                    "A heading occurrence ID was reused for a separate appearance."
                )
            previous_occurrence = chunk.heading_occurrence
        occurrence_text.setdefault(chunk.heading_occurrence, []).append(chunk.text)
        metadata = (
            chunk.heading,
            chunk.heading_level,
            chunk.parent_heading,
            chunk.section_path,
        )
        previous_metadata = occurrence_metadata.setdefault(
            chunk.heading_occurrence,
            metadata,
        )
        if previous_metadata != metadata:
            raise RuntimeError(
                "Chunks from one heading occurrence have inconsistent metadata."
            )

    for occurrence, text_parts in occurrence_text.items():
        section_lines = "".join(text_parts).splitlines()
        detected_headings = [
            _detect_heading(line)
            for line in section_lines
            if _detect_heading(line) is not None
        ]
        if (
            len(detected_headings) != 1
            or detected_headings[0]["heading"]
            != occurrence_metadata[occurrence][0]
        ):
            raise RuntimeError(
                "A heading occurrence ID must identify exactly one heading appearance."
            )


def main(output_directory: Path | None = None) -> None:
    """Create a structure-based preview and save chunk metadata and summary."""
    selected_output_directory = (
        OUTPUT_DIRECTORY if output_directory is None else Path(output_directory)
    )
    if (
        not isinstance(MAX_CHUNK_SIZE, int)
        or isinstance(MAX_CHUNK_SIZE, bool)
        or MAX_CHUNK_SIZE <= 0
    ):
        raise ValueError("MAX_CHUNK_SIZE must be a positive integer.")

    output_paths = (
        selected_output_directory / "chunks.txt",
        selected_output_directory / "summary.json",
        selected_output_directory / "chunk_metadata.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Preview output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    document = load_extraction_artifact(EXTRACTION_ARTIFACT)
    combined_text = document.text

    if not combined_text.strip():
        raise RuntimeError(
            "No usable text was extracted after cleaning; the preview cannot proceed."
        )

    chunks = structure_chunk(combined_text, max_chunk_size=MAX_CHUNK_SIZE)
    _validate_chunks(chunks, combined_text, MAX_CHUNK_SIZE)
    page_numbers = document.page_numbers_for_chunks(
        [chunk.text for chunk in chunks]
    )

    chunk_counts_by_heading: dict[str, int] = {}
    heading_by_occurrence: dict[int, str] = {}
    unheaded_chunk_count = 0
    for chunk in chunks:
        if chunk.heading is None:
            unheaded_chunk_count += 1
            continue
        chunk_counts_by_heading[chunk.heading] = (
            chunk_counts_by_heading.get(chunk.heading, 0) + 1
        )
        if chunk.heading_occurrence is not None:
            heading_by_occurrence[chunk.heading_occurrence] = chunk.heading

    occurrence_counts_by_heading: dict[str, int] = {}
    for heading in heading_by_occurrence.values():
        occurrence_counts_by_heading[heading] = (
            occurrence_counts_by_heading.get(heading, 0) + 1
        )

    total_characters = sum(len(chunk.text) for chunk in chunks)
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    unique_heading_keys = {
        chunk.heading.casefold() for chunk in chunks if chunk.heading is not None
    }
    representative_chunks: list[dict[str, object]] = []
    seen_representative_occurrences: set[int] = set()
    for chunk in chunks:
        if (
            chunk.heading is None
            or chunk.heading_occurrence in seen_representative_occurrences
        ):
            continue
        if chunk.heading_occurrence is None:
            continue
        seen_representative_occurrences.add(chunk.heading_occurrence)
        representative_chunks.append(
            {
                "chunk_index": chunk.chunk_index,
                "heading": chunk.heading,
                "heading_level": chunk.heading_level,
                "parent_heading": chunk.parent_heading,
                "section_path": list(chunk.section_path),
                "text_preview": chunk.text[:240],
            }
        )
        if len(representative_chunks) == 8:
            break

    summary = {
        "source_pdf_filename": document.source_document,
        "number_of_pages_extracted": len(document.pages),
        "number_of_empty_pages": len(document.empty_pages),
        "empty_page_numbers": list(document.empty_pages),
        "document_characters_after_cleaning": len(combined_text),
        "structure_heading_detection": "conservative_line_based",
        "maximum_chunk_size_characters": MAX_CHUNK_SIZE,
        "number_of_chunks": len(chunks),
        "number_of_unique_headings": len(unique_heading_keys),
        "number_of_heading_occurrences": len(heading_by_occurrence),
        "heading_occurrence_count_by_heading": occurrence_counts_by_heading,
        "number_of_unheaded_chunks": unheaded_chunk_count,
        "chunk_count_by_heading_level": {
            str(level): sum(chunk.heading_level == level for chunk in chunks)
            for level in range(4)
        },
        "representative_chunks": representative_chunks,
        "chunk_size_statistics_characters": {
            "average": total_characters / len(chunks) if chunks else 0,
            "minimum": min(chunk_lengths, default=None),
            "maximum": max(chunk_lengths, default=None),
        },
        "average_chunk_length_characters": (
            total_characters / len(chunks) if chunks else 0
        ),
        "minimum_chunk_length_characters": min(chunk_lengths, default=None),
        "maximum_chunk_length_characters": max(chunk_lengths, default=None),
        "chunk_count_by_heading": chunk_counts_by_heading,
        "text_preserved_exactly": True,
        "page_separator": PAGE_SEPARATOR,
        "page_level_chunk_provenance_available": True,
    }

    chunk_metadata = [
        {
            "chunk_index": chunk.chunk_index,
            "heading": chunk.heading,
            "heading_level": chunk.heading_level,
            "parent_heading": chunk.parent_heading,
            "section_path": list(chunk.section_path),
            "heading_occurrence": chunk.heading_occurrence,
            "character_count": len(chunk.text),
            "source_page_numbers": list(page_numbers[chunk.chunk_index - 1]),
            "text": chunk.text,
        }
        for chunk in chunks
    ]

    chunks_text = "\n\n".join(
        f"CHUNK {chunk.chunk_index} ({chunk.heading or 'unheaded'})\n{chunk.text}"
        for chunk in chunks
    )

    selected_output_directory.mkdir(parents=True, exist_ok=True)
    output_contents = (
        (output_paths[0], chunks_text),
        (output_paths[1], json.dumps(summary, indent=2, ensure_ascii=False) + "\n"),
        (output_paths[2], json.dumps(chunk_metadata, indent=2, ensure_ascii=False) + "\n"),
    )
    created_output_paths: list[Path] = []
    try:
        for path, content in output_contents:
            with path.open("x", encoding="utf-8", newline="\n") as output_file:
                created_output_paths.append(path)
                output_file.write(content)
    except Exception as write_error:
        cleanup_errors: list[str] = []
        for path in reversed(created_output_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{path}: {cleanup_error}")
        if cleanup_errors:
            raise RuntimeError(
                "Preview writing failed and some partial output files could not "
                f"be removed: {'; '.join(cleanup_errors)}"
            ) from write_error
        raise

    print(f"Pages loaded: {len(document.pages)}")
    if document.empty_pages:
        print(f"Empty pages: {list(document.empty_pages)}")
    else:
        print("Empty pages: none")
    print(f"Structure chunks created: {len(chunks)}")
    print(f"Unique detected headings: {summary['number_of_unique_headings']}")
    print(f"Heading occurrences: {summary['number_of_heading_occurrences']}")
    print(f"Unheaded chunks: {unheaded_chunk_count}")
    print(f"Average chunk length: {summary['average_chunk_length_characters']:.2f}")
    print(f"Output directory: {selected_output_directory}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a structure-based preview from the shared text artifact."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Use a fresh output directory instead of the default "
            "experiments/results/structure_based_preview_v2_corrected path."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().output_directory)
