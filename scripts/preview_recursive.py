"""Run recursive chunking on the shared page-preserving text artifact."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.recursive import RecursiveChunk, recursive_chunk  # noqa: E402
from src.ingestion.extraction_artifact import (  # noqa: E402
    PagePreservingText,
    load_extraction_artifact,
)


EXTRACTION_ARTIFACT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extracted"
    / "extracted_pages.json"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "recursive_preview"
CHUNK_SIZE = 1000
OVERLAP = 0


def _validate_chunks(
    document: PagePreservingText,
    chunks: list[RecursiveChunk],
) -> tuple[tuple[int, ...], ...]:
    """Validate exact reconstruction and size before assigning page metadata."""
    if any(len(chunk.text) > CHUNK_SIZE for chunk in chunks):
        raise RuntimeError("A generated chunk exceeds the configured size limit.")

    overlaps = [chunk.overlap_length for chunk in chunks]
    try:
        page_numbers = document.page_numbers_for_chunks(
            [chunk.text for chunk in chunks],
            overlaps,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "The recursive chunks do not reconstruct the source text exactly."
        ) from error
    return page_numbers


def main(output_directory: Path | None = None) -> None:
    """Load the shared artifact, chunk it, and save a recursive preview."""
    selected_output = (
        OUTPUT_DIRECTORY if output_directory is None else Path(output_directory)
    )
    if selected_output.exists():
        raise FileExistsError(
            f"Recursive preview output already exists and will not be overwritten: "
            f"{selected_output}"
        )

    document = load_extraction_artifact(EXTRACTION_ARTIFACT)
    chunks = recursive_chunk(document.text, CHUNK_SIZE, OVERLAP)
    page_numbers = _validate_chunks(document, chunks)
    lengths = [len(chunk.text) for chunk in chunks]
    summary = {
        "chunking_strategy": "recursive",
        "source_document": document.source_document,
        "number_of_pages_extracted": len(document.pages),
        "number_of_empty_pages": len(document.empty_pages),
        "empty_page_numbers": list(document.empty_pages),
        "cleaned_document_characters": len(document.text),
        "page_separator": "\n",
        "chunk_size_characters": CHUNK_SIZE,
        "overlap_characters": OVERLAP,
        "number_of_chunks": len(chunks),
        "average_chunk_length_characters": (
            sum(lengths) / len(lengths) if lengths else 0
        ),
        "minimum_chunk_length_characters": min(lengths, default=None),
        "maximum_chunk_length_characters": max(lengths, default=None),
        "text_preserved_exactly": True,
        "maximum_chunk_size_validated": all(
            length <= CHUNK_SIZE for length in lengths
        ),
        "page_level_provenance_available": True,
    }
    chunk_metadata = [
        {
            "chunk_index": index,
            "character_count": len(chunk.text),
            "overlap_length": chunk.overlap_length,
            "source_page_numbers": list(page_numbers[index - 1]),
            "text": chunk.text,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    chunks_text = "\n\n".join(
        (
            f"CHUNK {index} (characters: {len(chunk.text)}; "
            f"overlap: {chunk.overlap_length})\n{chunk.text}"
        )
        for index, chunk in enumerate(chunks, start=1)
    )

    selected_output.mkdir(parents=True)
    output_contents = (
        (selected_output / "chunks.txt", chunks_text),
        (
            selected_output / "summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        ),
        (
            selected_output / "chunk_metadata.json",
            json.dumps(chunk_metadata, indent=2, ensure_ascii=False) + "\n",
        ),
    )
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
                path.unlink()
            except OSError as cleanup_error:
                cleanup_errors.append(f"{path}: {cleanup_error}")
        if not cleanup_errors:
            try:
                selected_output.rmdir()
            except OSError as cleanup_error:
                cleanup_errors.append(f"{selected_output}: {cleanup_error}")
        if cleanup_errors:
            raise RuntimeError(
                "Recursive preview writing failed and partial output cleanup "
                f"was incomplete: {'; '.join(cleanup_errors)}"
            ) from write_error
        raise

    print(f"Source document: {document.source_document}")
    print(f"Pages loaded: {len(document.pages)}")
    print(f"Recursive chunks created: {len(chunks)}")
    print(f"Output directory: {selected_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a recursive preview from the shared text artifact."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Use a fresh output directory instead of the default "
            "experiments/results/recursive_preview path."
        ),
    )
    main(parser.parse_args().output_directory)
