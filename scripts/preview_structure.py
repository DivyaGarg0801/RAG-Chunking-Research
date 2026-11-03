"""Run the structure-aware preview on the local laboratory manual."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.structure import StructureChunk, structure_chunk
from src.ingestion.pdf_loader import extract_pdf_pages
from src.preprocessing.text_cleaner import clean_text

SOURCE_PDF = PROJECT_ROOT / "data" / "raw" / "computer_networks_lab_manual.pdf"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "structure_preview"
MAX_CHUNK_SIZE = 1500
PAGE_SEPARATOR = "\n"


def _validate_chunks(
    chunks: list[StructureChunk],
    combined_text: str,
    max_chunk_size: int,
) -> None:
    """Validate chunk metadata, reconstruction, and configured size limits."""
    if not chunks:
        raise RuntimeError("The structure-based preview produced no chunks.")

    chunk_indices = [chunk.chunk_index for chunk in chunks]
    if chunk_indices != list(range(1, len(chunks) + 1)):
        raise RuntimeError("Chunk indexes are not sequential.")

    if any(not chunk.text for chunk in chunks):
        raise RuntimeError("One or more generated chunks are empty.")

    for chunk in chunks:
        if chunk.heading_level not in (0, 1, 2, 3):
            raise RuntimeError("A generated chunk has an invalid heading level.")
        if chunk.heading is None and chunk.heading_level != 0:
            raise RuntimeError(
                "An unheaded chunk must have heading level 0."
            )
        if chunk.heading is not None and chunk.heading_level == 0:
            raise RuntimeError("A headed chunk must have a positive heading level.")
        if chunk.heading is not None and not chunk.section_path:
            raise RuntimeError(
                "A headed chunk must have a non-empty section path."
            )
        if chunk.heading is not None and chunk.section_path[-1] != chunk.heading:
            raise RuntimeError("A headed chunk has an inconsistent section path.")
        if chunk.heading is not None:
            if chunk.parent_heading is None and len(chunk.section_path) != 1:
                raise RuntimeError(
                    "A root heading must have a one-item section path."
                )
            if chunk.parent_heading is not None and (
                len(chunk.section_path) != chunk.heading_level
                or len(chunk.section_path) < 2
                or chunk.section_path[-2] != chunk.parent_heading
            ):
                raise RuntimeError(
                    "A headed chunk has an inconsistent parent and section path."
                )

    occurrence_metadata: dict[
        int, tuple[str | None, int, str | None, tuple[str, ...]]
    ] = {}
    for chunk in chunks:
        if chunk.heading_occurrence is None:
            continue
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

    reconstructed_text = "".join(chunk.text for chunk in chunks)
    if reconstructed_text != combined_text:
        raise RuntimeError(
            "The structure-based preview did not reconstruct the document text exactly."
        )

    if any(len(chunk.text) > max_chunk_size for chunk in chunks):
        raise RuntimeError("A generated chunk exceeds the configured size limit.")


def main() -> None:
    """Create a structure-based preview and save chunk metadata and summary."""
    if (
        not isinstance(MAX_CHUNK_SIZE, int)
        or isinstance(MAX_CHUNK_SIZE, bool)
        or MAX_CHUNK_SIZE <= 0
    ):
        raise ValueError("MAX_CHUNK_SIZE must be a positive integer.")

    if not SOURCE_PDF.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {SOURCE_PDF}")

    output_paths = (
        OUTPUT_DIRECTORY / "chunks.txt",
        OUTPUT_DIRECTORY / "summary.json",
        OUTPUT_DIRECTORY / "chunk_metadata.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Preview output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    extraction = extract_pdf_pages(SOURCE_PDF)
    cleaned_pages = [clean_text(page.text) for page in extraction.pages]
    combined_text = PAGE_SEPARATOR.join(cleaned_pages)

    if not combined_text.strip():
        raise RuntimeError(
            "No usable text was extracted after cleaning; the preview cannot proceed."
        )

    chunks = structure_chunk(combined_text, max_chunk_size=MAX_CHUNK_SIZE)
    _validate_chunks(chunks, combined_text, MAX_CHUNK_SIZE)

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
    summary = {
        "source_pdf_filename": SOURCE_PDF.name,
        "number_of_pages_extracted": len(extraction.pages),
        "number_of_empty_pages": len(extraction.empty_pages),
        "empty_page_numbers": list(extraction.empty_pages),
        "document_characters_after_cleaning": len(combined_text),
        "structure_heading_detection": "conservative_line_based",
        "maximum_chunk_size_characters": MAX_CHUNK_SIZE,
        "number_of_chunks": len(chunks),
        "number_of_unique_headings": len(unique_heading_keys),
        "number_of_heading_occurrences": len(heading_by_occurrence),
        "heading_occurrence_count_by_heading": occurrence_counts_by_heading,
        "number_of_unheaded_chunks": unheaded_chunk_count,
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
            "text": chunk.text,
        }
        for chunk in chunks
    ]

    chunks_text = "\n\n".join(
        f"CHUNK {chunk.chunk_index} ({chunk.heading or 'unheaded'})\n{chunk.text}"
        for chunk in chunks
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
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

    print(f"Pages extracted: {len(extraction.pages)}")
    if extraction.empty_pages:
        print(f"Empty pages: {list(extraction.empty_pages)}")
    else:
        print("Empty pages: none")
    print(f"Structure chunks created: {len(chunks)}")
    print(f"Unique detected headings: {summary['number_of_unique_headings']}")
    print(f"Heading occurrences: {summary['number_of_heading_occurrences']}")
    print(f"Unheaded chunks: {unheaded_chunk_count}")
    print(f"Average chunk length: {summary['average_chunk_length_characters']:.2f}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
