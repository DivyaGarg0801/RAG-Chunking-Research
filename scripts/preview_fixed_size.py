"""Run fixed-size chunking on the shared cleaned-text artifact."""

import argparse
import json
from pathlib import Path

from src.chunking.fixed_size import fixed_size_chunk
from src.ingestion.extraction_artifact import (
    DEFAULT_ARTIFACT_PATH,
    PAGE_SEPARATOR,
    load_extraction_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_ARTIFACT = DEFAULT_ARTIFACT_PATH
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "fixed_size_preview"

CHUNK_SIZE = 1000
OVERLAP = 150


def main(output_directory: Path | None = None) -> None:
    """Load the shared source artifact and save a fixed-size preview."""
    selected_output_directory = (
        OUTPUT_DIRECTORY if output_directory is None else Path(output_directory)
    )
    output_paths = (
        selected_output_directory / "chunks.txt",
        selected_output_directory / "summary.json",
        selected_output_directory / "chunk_metadata.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Preview output already exists and will not be overwritten: {paths}"
        )

    document = load_extraction_artifact(EXTRACTION_ARTIFACT)
    combined_text = document.text
    chunks = fixed_size_chunk(combined_text, CHUNK_SIZE, OVERLAP)
    overlap_lengths = (
        [0] + [OVERLAP] * (len(chunks) - 1) if chunks else []
    )
    page_numbers = document.page_numbers_for_chunks(chunks, overlap_lengths)

    total_chunk_characters = sum(len(chunk) for chunk in chunks)
    reconstructed_text = (
        chunks[0] + "".join(chunk[OVERLAP:] for chunk in chunks[1:])
        if chunks
        else ""
    )
    if reconstructed_text != combined_text:
        raise RuntimeError("The generated chunks do not reconstruct the input text.")

    chunk_lengths = [len(chunk) for chunk in chunks]
    repeated_characters = total_chunk_characters - len(combined_text)
    summary = {
        "source_pdf_filename": document.source_document,
        "number_of_pages_extracted": len(document.pages),
        "number_of_empty_pages": len(document.empty_pages),
        "empty_page_numbers": list(document.empty_pages),
        "total_cleaned_document_characters": len(combined_text),
        "page_separator": PAGE_SEPARATOR,
        "chunk_size_characters": CHUNK_SIZE,
        "overlap_characters": OVERLAP,
        "number_of_chunks": len(chunks),
        "average_chunk_length_characters": (
            total_chunk_characters / len(chunks) if chunks else 0
        ),
        "minimum_chunk_length_characters": min(chunk_lengths, default=None),
        "maximum_chunk_length_characters": max(chunk_lengths, default=None),
        "total_characters_across_all_chunks": total_chunk_characters,
        "derived_overlap_statistics": {
            "estimated_repeated_characters": repeated_characters,
            "repeated_characters_as_percent_of_chunk_text": (
                repeated_characters / total_chunk_characters * 100
                if total_chunk_characters
                else 0
            ),
            "method": (
                "Derived as total characters across chunks minus the combined "
                "cleaned document character count; depends on complete coverage "
                "and order-preserving chunks."
            ),
        },
        "chunks_may_cross_page_boundaries": True,
        "page_level_chunk_provenance_available": True,
    }
    chunk_metadata = [
        {
            "chunk_index": index,
            "character_count": len(chunk),
            "source_page_numbers": list(page_numbers[index - 1]),
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    chunks_text = "\n\n".join(
        f"CHUNK {index} (characters: {len(chunk)})\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    )

    selected_output_directory.mkdir(parents=True, exist_ok=True)
    output_contents = (
        (output_paths[0], chunks_text),
        (
            output_paths[1],
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        ),
        (
            output_paths[2],
            json.dumps(chunk_metadata, indent=2, ensure_ascii=False) + "\n",
        ),
    )
    for path, content in output_contents:
        with path.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)

    print(f"Pages loaded: {len(document.pages)}")
    if document.empty_pages:
        print(f"Pages with no cleaned text: {list(document.empty_pages)}")
    else:
        print("Pages with no extractable text: none")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk length: {summary['average_chunk_length_characters']:.2f}")
    print(f"Minimum chunk length: {summary['minimum_chunk_length_characters']}")
    print(f"Maximum chunk length: {summary['maximum_chunk_length_characters']}")
    print(f"Output directory: {selected_output_directory}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-size preview from the shared text artifact."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Use a fresh output directory instead of the default "
            "experiments/results/fixed_size_preview path."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().output_directory)
