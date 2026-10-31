"""Run the existing PDF, preprocessing, and fixed-size chunking pipeline."""

import json
from pathlib import Path

from src.chunking.fixed_size import fixed_size_chunk
from src.ingestion.pdf_loader import extract_pdf_pages
from src.preprocessing.text_cleaner import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = PROJECT_ROOT / "data" / "raw" / "computer_networks_lab_manual.pdf"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "fixed_size_preview"

CHUNK_SIZE = 1000
OVERLAP = 150
PAGE_SEPARATOR = "\n"


def main() -> None:
    """Create a fixed-size preview and save its chunks and statistics."""
    if not SOURCE_PDF.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {SOURCE_PDF}")

    output_paths = (
        OUTPUT_DIRECTORY / "chunks.txt",
        OUTPUT_DIRECTORY / "summary.json",
        OUTPUT_DIRECTORY / "chunk_metadata.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Preview output already exists and will not be overwritten: {paths}"
        )

    extraction = extract_pdf_pages(SOURCE_PDF)
    cleaned_page_texts = [clean_text(page.text) for page in extraction.pages]
    combined_text = PAGE_SEPARATOR.join(cleaned_page_texts)
    chunks = fixed_size_chunk(combined_text, CHUNK_SIZE, OVERLAP)

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
        "source_pdf_filename": SOURCE_PDF.name,
        "number_of_pages_extracted": len(extraction.pages),
        "number_of_empty_pages": len(extraction.empty_pages),
        "empty_page_numbers": list(extraction.empty_pages),
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
        "page_level_chunk_provenance_available": False,
    }
    chunk_metadata = [
        {
            "chunk_index": index,
            "character_count": len(chunk),
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    chunks_text = "\n\n".join(
        f"CHUNK {index} (characters: {len(chunk)})\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
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

    print(f"Pages extracted: {len(extraction.pages)}")
    if extraction.empty_pages:
        print(f"Pages with no extractable text: {list(extraction.empty_pages)}")
    else:
        print("Pages with no extractable text: none")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk length: {summary['average_chunk_length_characters']:.2f}")
    print(f"Minimum chunk length: {summary['minimum_chunk_length_characters']}")
    print(f"Maximum chunk length: {summary['maximum_chunk_length_characters']}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
