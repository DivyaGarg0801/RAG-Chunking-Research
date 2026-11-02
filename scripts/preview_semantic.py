"""Run the local semantic chunking pipeline and save a reproducible preview."""

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.chunking.semantic import (
    DEFAULT_MODEL_NAME,
    SemanticChunkingResult,
    load_sentence_model,
    semantic_chunk,
)
from src.ingestion.pdf_loader import extract_pdf_pages
from src.preprocessing.text_cleaner import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = PROJECT_ROOT / "data" / "raw" / "computer_networks_lab_manual.pdf"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "semantic_preview"

MODEL_NAME = DEFAULT_MODEL_NAME
MODEL_REVISION: str | None = None
MAX_CHUNK_SIZE = 1000
SIMILARITY_THRESHOLD = 0.5
BATCH_SIZE = 32
DEVICE = "cpu"
PAGE_SEPARATOR = "\n"


def _runtime_versions() -> dict[str, str]:
    """Return library versions needed to identify the local runtime."""
    versions = {"python": platform.python_version()}
    for package in (
        "PyMuPDF",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "requests",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required package version could not be recorded: {package}"
            ) from error
    return versions


def _build_summary(
    result: SemanticChunkingResult,
    combined_text: str,
    page_count: int,
    empty_pages: tuple[int, ...],
    model: object | None,
) -> dict[str, object]:
    """Build the run configuration and derived chunk statistics."""
    lengths = [chunk.character_count for chunk in result.chunks]
    resolved_revision = _resolved_model_revision(model)
    return {
        "source_pdf_filename": SOURCE_PDF.name,
        "number_of_pages_extracted": page_count,
        "number_of_empty_pages": len(empty_pages),
        "empty_page_numbers": list(empty_pages),
        "total_cleaned_document_characters": len(combined_text),
        "page_separator": PAGE_SEPARATOR,
        "chunking": {
            "strategy": result.metadata.strategy_name,
            "model_name": result.metadata.model_name,
            "requested_model_revision": MODEL_REVISION,
            "resolved_model_revision": resolved_revision,
            "model_revision_note": (
                "Resolved revision is read from the loaded Transformers model "
                "configuration when available; otherwise it is unknown."
            ),
            "similarity_method": result.metadata.similarity_method,
            "similarity_threshold": result.metadata.similarity_threshold,
            "maximum_chunk_size_characters": result.metadata.maximum_chunk_size,
            "batch_size": result.metadata.batch_size,
            "device": result.metadata.device,
            "model_loaded_during_call": result.metadata.model_loaded_during_call,
            "sentence_segmentation_approach": (
                result.metadata.sentence_segmentation_approach
            ),
            "runtime_versions": _runtime_versions(),
        },
        "number_of_original_segments": result.metadata.original_segment_count,
        "number_of_embedding_inputs": result.metadata.embedding_input_count,
        "number_of_chunks": len(result.chunks),
        "average_chunk_length_characters": (
            sum(lengths) / len(lengths) if lengths else 0
        ),
        "minimum_chunk_length_characters": min(lengths, default=None),
        "maximum_chunk_length_characters": max(lengths, default=None),
        "total_characters_across_all_chunks": sum(lengths),
        "overlap_characters": 0,
        "chunk_boundary_counts": {
            "document_start": sum(
                chunk.boundary_before == "document_start"
                for chunk in result.chunks
            ),
            "semantic": sum(
                chunk.boundary_before == "semantic" for chunk in result.chunks
            ),
            "size": sum(
                chunk.boundary_before == "size" for chunk in result.chunks
            ),
        },
        "chunking_time_seconds": result.metadata.execution_time_seconds,
        "chunks_may_cross_page_boundaries": True,
        "page_level_chunk_provenance_available": False,
    }


def _resolved_model_revision(model: object | None) -> str | None:
    """Read the resolved Transformers commit hash when the model exposes it."""
    if model is None:
        return None
    first_module = getattr(model, "_first_module", None)
    if not callable(first_module):
        return None
    transformer_module = first_module()
    auto_model = getattr(transformer_module, "auto_model", None)
    config = getattr(auto_model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else None


def main() -> None:
    """Extract, clean, semantically chunk, and save the local PDF preview."""
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
    model = (
        load_sentence_model(MODEL_NAME, DEVICE, revision=MODEL_REVISION)
        if combined_text.strip()
        else None
    )
    result = semantic_chunk(
        combined_text,
        chunk_size=MAX_CHUNK_SIZE,
        similarity_threshold=SIMILARITY_THRESHOLD,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        device=DEVICE,
        model=model,
    )

    if "".join(chunk.text for chunk in result.chunks) != combined_text:
        raise RuntimeError("The generated chunks do not reconstruct the input text.")
    if any(chunk.character_count > MAX_CHUNK_SIZE for chunk in result.chunks):
        raise RuntimeError("A generated chunk exceeds the configured size limit.")

    summary = _build_summary(
        result,
        combined_text,
        len(extraction.pages),
        tuple(extraction.empty_pages),
        model,
    )
    chunk_metadata = [
        {
            "chunk_index": chunk.chunk_index,
            "character_count": chunk.character_count,
            "boundary_before": chunk.boundary_before,
            "text": chunk.text,
        }
        for chunk in result.chunks
    ]
    chunks_text = "\n\n".join(
        f"CHUNK {chunk.chunk_index} "
        f"(characters: {chunk.character_count}; "
        f"boundary: {chunk.boundary_before})\n{chunk.text}"
        for chunk in result.chunks
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_contents = (
        (output_paths[0], chunks_text),
        (output_paths[1], json.dumps(summary, indent=2, ensure_ascii=False) + "\n"),
        (
            output_paths[2],
            json.dumps(chunk_metadata, indent=2, ensure_ascii=False) + "\n",
        ),
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
        print(f"Pages with no extractable text: {list(extraction.empty_pages)}")
    else:
        print("Pages with no extractable text: none")
    print(f"Chunks created: {len(result.chunks)}")
    print(f"Average chunk length: {summary['average_chunk_length_characters']:.2f}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
