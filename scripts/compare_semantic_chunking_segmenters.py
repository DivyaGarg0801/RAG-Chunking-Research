"""Compare semantic chunking with both document segmentation variants."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.chunking.semantic import (
    SemanticChunkingResult,
    SentenceEmbeddingModel,
    load_sentence_model,
    segment_text,
    semantic_chunk,
)
from src.chunking.semantic_segmentation import segment_text_paragraph_aware
from src.ingestion.pdf_loader import extract_pdf_pages
from src.preprocessing.text_cleaner import clean_text
from scripts.preview_semantic import (
    _resolved_model_revision,
    _runtime_versions,
    BATCH_SIZE,
    DEVICE,
    MAX_CHUNK_SIZE,
    MODEL_REVISION,
    MODEL_NAME,
    PAGE_SEPARATOR,
    SIMILARITY_THRESHOLD,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = PROJECT_ROOT / "data" / "raw" / "computer_networks_lab_manual.pdf"
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "semantic_segmentation_comparison"
)
BASELINE_DIRECTORY = PROJECT_ROOT / "experiments" / "results" / "semantic_preview"


class TimedEmbeddingModel:
    """Delegate model operations while measuring only encode wall time."""

    def __init__(self, model: SentenceEmbeddingModel) -> None:
        self.model = model
        self.encode_seconds = 0.0
        self.max_seq_length: int = model.max_seq_length
        self.tokenizer: Any = model.tokenizer

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        device: str,
    ) -> Any:
        started_at = time.perf_counter()
        try:
            return self.model.encode(
                sentences,
                batch_size=batch_size,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                device=device,
            )
        finally:
            self.encode_seconds += time.perf_counter() - started_at

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.model.get_sentence_embedding_dimension()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_state() -> dict[str, Any]:
    paths = (
        BASELINE_DIRECTORY / "chunks.txt",
        BASELINE_DIRECTORY / "summary.json",
        BASELINE_DIRECTORY / "chunk_metadata.json",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The original semantic preview is required for baseline verification: "
            + ", ".join(str(path) for path in missing)
        )

    summary = json.loads(paths[1].read_text(encoding="utf-8"))
    chunk_metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    if (
        summary.get("total_cleaned_document_characters") != 78_531
        or summary.get("number_of_original_segments") != 1_833
        or summary.get("number_of_chunks") != 1_599
        or summary.get("chunking", {}).get("similarity_threshold")
        != SIMILARITY_THRESHOLD
    ):
        raise RuntimeError(
            "The original preview summary no longer matches the known baseline; "
            "inspect it before comparing runs."
        )
    return {
        "paths": paths,
        "hashes_before": {str(path): _sha256(path) for path in paths},
        "summary": summary,
        "chunks": chunk_metadata,
    }


def _verify_baseline_unchanged(baseline: dict[str, Any]) -> None:
    changed = [
        path
        for path, digest in baseline["hashes_before"].items()
        if _sha256(Path(path)) != digest
    ]
    if changed:
        raise RuntimeError(
            "The original semantic preview changed during the experiment: "
            + ", ".join(changed)
        )


def _chunk_statistics(
    result: SemanticChunkingResult,
    source_text: str,
) -> dict[str, Any]:
    chunks = result.chunks
    lengths = [chunk.character_count for chunk in chunks]
    return {
        "input_segment_count": result.metadata.original_segment_count,
        "embedding_input_count": result.metadata.embedding_input_count,
        "final_chunk_count": len(chunks),
        "average_chunk_length_characters": (
            sum(lengths) / len(lengths) if lengths else 0
        ),
        "minimum_chunk_length_characters": min(lengths, default=None),
        "maximum_chunk_length_characters": max(lengths, default=None),
        "chunks_at_most_10_characters": sum(length <= 10 for length in lengths),
        "chunks_at_most_20_characters": sum(length <= 20 for length in lengths),
        "chunks_at_most_50_characters": sum(length <= 50 for length in lengths),
        "semantic_boundary_count": sum(
            chunk.boundary_before == "semantic" for chunk in chunks
        ),
        "size_boundary_count": sum(
            chunk.boundary_before == "size" for chunk in chunks
        ),
        "exact_source_reconstruction": (
            "".join(chunk.text for chunk in chunks) == source_text
        ),
        "all_chunks_within_maximum_size": all(
            chunk.character_count <= MAX_CHUNK_SIZE for chunk in chunks
        ),
        "total_chunk_characters": sum(lengths),
    }


def _run_variant(
    variant_name: str,
    segments: list[str],
    source_text: str,
    model: SentenceEmbeddingModel,
    segmentation_seconds: float,
) -> tuple[dict[str, Any], SemanticChunkingResult]:
    timed_model = TimedEmbeddingModel(model)
    started_at = time.perf_counter()
    result = semantic_chunk(
        source_text,
        chunk_size=MAX_CHUNK_SIZE,
        similarity_threshold=SIMILARITY_THRESHOLD,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        device=DEVICE,
        model=timed_model,
        input_segments=segments,
        segmentation_approach=variant_name,
    )
    semantic_call_seconds = time.perf_counter() - started_at
    if not "".join(chunk.text for chunk in result.chunks) == source_text:
        raise RuntimeError(f"{variant_name} chunks did not reconstruct the source.")
    if any(chunk.character_count > MAX_CHUNK_SIZE for chunk in result.chunks):
        raise RuntimeError(f"{variant_name} exceeded the maximum chunk size.")

    return (
        {
            **_chunk_statistics(result, source_text),
            "segmentation_seconds": segmentation_seconds,
            "embedding_inference_seconds": timed_model.encode_seconds,
            "non_embedding_chunking_seconds": max(
                0.0,
                semantic_call_seconds - timed_model.encode_seconds,
            ),
            "semantic_chunk_call_seconds": semantic_call_seconds,
            "total_processing_seconds": (
                segmentation_seconds + semantic_call_seconds
            ),
            "model_loaded_during_semantic_chunk_call": (
                result.metadata.model_loaded_during_call
            ),
        },
        result,
    )


def _baseline_matches(
    result: SemanticChunkingResult,
    baseline: dict[str, Any],
) -> bool:
    expected = [
        (record["text"], record["boundary_before"])
        for record in baseline["chunks"]
    ]
    actual = [
        (chunk.text, chunk.boundary_before)
        for chunk in result.chunks
    ]
    return actual == expected


def _example_chunks(
    chunks: Sequence[Any],
) -> dict[str, list[dict[str, int | str]]]:
    phrases = {
        "experiment_heading_and_aim": ("Experiment-1", "Aim: To gain familiarity"),
        "procedure_and_ping_explanation": (
            "Procedure:",
            "The ping command is usually used as a",
            "simple way to verify",
        ),
        "udp_socket_programming": ("UDP Socket Programming:",),
        "tcp_flow_control": (
            "TCP Flow Control:",
            "Transmission Control Protocol (TCP) Flow Control",
        ),
        "numbered_steps": ("1) Troubleshooting", "a) ping <address>"),
        "commands": ("sudo apt-get install ifplugd", "gcc -o server tcp_server.c"),
    }
    examples: dict[str, list[dict[str, int | str]]] = {}
    for category, targets in phrases.items():
        matches: list[dict[str, int | str]] = []
        seen_indices: set[int] = set()
        for target in targets:
            for chunk in chunks:
                if target.casefold() in chunk.text.casefold():
                    if chunk.chunk_index not in seen_indices:
                        matches.append(
                            {
                                "chunk_index": chunk.chunk_index,
                                "matched_text": target,
                                "text": chunk.text,
                            }
                        )
                        seen_indices.add(chunk.chunk_index)
                    break
        examples[category] = matches
    return examples


def _write_outputs(
    results: dict[str, SemanticChunkingResult],
    comparison: dict[str, Any],
    output_directory: Path,
) -> None:
    output_files: list[tuple[Path, str]] = []
    for variant, result in results.items():
        variant_directory = output_directory / variant
        variant_summary = {
            "generated_at_utc": comparison["generated_at_utc"],
            "source": comparison["source"],
            "segmentation_approach": variant,
            "configuration": comparison["configuration"],
            "timing_definitions": comparison["timing_definitions"],
            "shared_setup": comparison["shared_setup"],
            "statistics": comparison["variants"][variant],
            "exact_source_reconstruction": comparison["variants"][variant][
                "exact_source_reconstruction"
            ],
            "all_chunks_within_maximum_size": comparison["variants"][variant][
                "all_chunks_within_maximum_size"
            ],
        }
        output_files.extend(
            [
                (
                    variant_directory / "summary.json",
                    json.dumps(
                        variant_summary,
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                ),
                (
                    variant_directory / "chunks.txt",
                    "\n\n".join(
                        f"CHUNK {chunk.chunk_index} "
                        f"(characters: {chunk.character_count}; "
                        f"boundary: {chunk.boundary_before})\n{chunk.text}"
                        for chunk in result.chunks
                    ),
                ),
                (
                    variant_directory / "chunk_metadata.json",
                    json.dumps(
                        [
                            {
                                "chunk_index": chunk.chunk_index,
                                "character_count": chunk.character_count,
                                "boundary_before": chunk.boundary_before,
                                "text": chunk.text,
                            }
                            for chunk in result.chunks
                        ],
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                ),
            ]
        )
    output_files.append(
        (
            output_directory / "comparison.json",
            json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        )
    )
    existing = [path for path, _ in output_files if path.exists()]
    if existing:
        raise FileExistsError(
            "Comparison output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )

    created: list[Path] = []
    try:
        for path, content in output_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as output_file:
                created.append(path)
                output_file.write(content)
    except Exception as write_error:
        cleanup_errors: list[str] = []
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"{path}: {cleanup_error}")
        if cleanup_errors:
            raise RuntimeError(
                "Writing the comparison failed and partial output cleanup also "
                f"failed: {'; '.join(cleanup_errors)}"
            ) from write_error
        raise


def main(output_directory: Path = OUTPUT_DIRECTORY) -> None:
    if not SOURCE_PDF.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {SOURCE_PDF}")
    resolved_output = output_directory.resolve()
    resolved_baseline = BASELINE_DIRECTORY.resolve()
    if resolved_output == resolved_baseline or resolved_baseline in resolved_output.parents:
        raise ValueError(
            "Comparison outputs cannot be placed inside the original preview directory."
        )
    baseline = _baseline_state()
    preparation_started = time.perf_counter()
    extraction = extract_pdf_pages(SOURCE_PDF)
    cleaned_pages = [clean_text(page.text) for page in extraction.pages]
    source_text = PAGE_SEPARATOR.join(cleaned_pages)
    source_preparation_seconds = time.perf_counter() - preparation_started
    if len(source_text) != 78_531 or len(cleaned_pages) != 87:
        raise RuntimeError(
            "The extracted/cleaned source differs from the expected 87-page, "
            "78,531-character input."
        )

    original_segment_started = time.perf_counter()
    original_segments = segment_text(source_text)
    original_segmentation_seconds = (
        time.perf_counter() - original_segment_started
    )
    page_offsets: list[int] = []
    offset = 0
    for page_text in cleaned_pages[:-1]:
        offset += len(page_text)
        page_offsets.append(offset)
        offset += len(PAGE_SEPARATOR)
    paragraph_segmentation_started = time.perf_counter()
    paragraph_segments = segment_text_paragraph_aware(
        source_text,
        page_break_offsets=tuple(page_offsets),
    )
    paragraph_segmentation_seconds = (
        time.perf_counter() - paragraph_segmentation_started
    )
    if "".join(original_segments) != source_text:
        raise RuntimeError("The original segmenter did not reconstruct the source.")
    if "".join(paragraph_segments) != source_text:
        raise RuntimeError(
            "The paragraph-aware segmenter did not reconstruct the source."
        )

    model_load_started = time.perf_counter()
    model = load_sentence_model(
        MODEL_NAME,
        DEVICE,
        revision=MODEL_REVISION,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    warmup_started = time.perf_counter()
    model.encode(
        ["semantic chunking timing warm-up"],
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=DEVICE,
    )
    model_warmup_seconds = time.perf_counter() - warmup_started

    definitions = {
        "rule_based_line_and_sentence_v1": (
            original_segments,
            original_segmentation_seconds,
        ),
        "paragraph_aware_v2": (
            paragraph_segments,
            paragraph_segmentation_seconds,
        ),
    }
    statistics_by_variant: dict[str, dict[str, Any]] = {}
    results: dict[str, SemanticChunkingResult] = {}
    for variant_name, (segments, segmentation_seconds) in definitions.items():
        stats, result = _run_variant(
            variant_name,
            segments,
            source_text,
            model,
            segmentation_seconds,
        )
        statistics_by_variant[variant_name] = stats
        results[variant_name] = result

    if not _baseline_matches(
        results["rule_based_line_and_sentence_v1"],
        baseline,
    ):
        raise RuntimeError(
            "The original-segmenter run differs from the saved 0.5 preview. "
            "Check the cached model revision and runtime before interpreting results."
        )
    _verify_baseline_unchanged(baseline)

    comparison: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "pdf_filename": SOURCE_PDF.name,
            "pdf_sha256": _sha256(SOURCE_PDF),
            "page_count": len(cleaned_pages),
            "cleaned_character_count": len(source_text),
            "empty_page_numbers": list(extraction.empty_pages),
            "page_separator": PAGE_SEPARATOR,
        },
        "configuration": {
            "strategy": "semantic",
            "segmentation_variants": list(definitions),
            "model_name": MODEL_NAME,
            "requested_model_revision": MODEL_REVISION,
            "resolved_model_revision": _resolved_model_revision(model),
            "model_revision_note": (
                "The resolved revision is read from the loaded Transformers "
                "configuration when available; null means unavailable. The "
                "requested revision is not pinned."
            ),
            "similarity_method": "cosine_similarity",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "maximum_chunk_size_characters": MAX_CHUNK_SIZE,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "model_loaded_once_and_reused": True,
            "embedding_runs": 2,
            "warmup_text": "semantic chunking timing warm-up",
            "offline_model_loading": True,
            "runtime_versions": _runtime_versions(),
        },
        "timing_definitions": {
            "source_preparation_seconds": (
                "Shared PDF extraction and cleaning duration."
            ),
            "model_load_seconds": (
                "One-time local model load, reported separately and excluded "
                "from per-variant totals."
            ),
            "model_warmup_seconds": (
                "One discarded fixed-text encode used to initialize the model; "
                "excluded from per-variant measurements."
            ),
            "embedding_inference_seconds": (
                "Wall time inside the model.encode call for that variant."
            ),
            "non_embedding_chunking_seconds": (
                "Semantic call duration minus model.encode wall time; includes "
                "token-safe preparation, normalization, similarity, and chunk "
                "assembly."
            ),
            "total_processing_seconds": (
                "Segmentation duration plus the complete semantic_chunk call; "
                "includes embedding time but excludes shared source preparation, "
                "model load, and warm-up."
            ),
        },
        "shared_setup": {
            "source_preparation_seconds": source_preparation_seconds,
            "model_load_seconds": model_load_seconds,
            "model_warmup_seconds": model_warmup_seconds,
            "baseline_preview_files_sha256_before": baseline["hashes_before"],
        },
        "variants": statistics_by_variant,
        "qualitative_examples": {
            variant: _example_chunks(result.chunks)
            for variant, result in results.items()
        },
        "original_preview": {
            "verified_output_matches_baseline": True,
            "verified_files_unchanged": True,
            "files_sha256_after": {
                path: _sha256(Path(path))
                for path in baseline["hashes_before"]
            },
        },
    }
    _write_outputs(results, comparison, output_directory)
    print("| Variant | Input segments | Embedding inputs | Chunks | Avg chars | Min–max | <=10 | <=20 | <=50 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant, stats in statistics_by_variant.items():
        print(
            f"| {variant} | {stats['input_segment_count']} | "
            f"{stats['embedding_input_count']} | {stats['final_chunk_count']} | "
            f"{stats['average_chunk_length_characters']:.2f} | "
            f"{stats['minimum_chunk_length_characters']}–"
            f"{stats['maximum_chunk_length_characters']} | "
            f"{stats['chunks_at_most_10_characters']} | "
            f"{stats['chunks_at_most_20_characters']} | "
            f"{stats['chunks_at_most_50_characters']} |"
        )
        print(
            f"  semantic boundaries={stats['semantic_boundary_count']}, "
            f"size boundaries={stats['size_boundary_count']}, "
            f"segmentation={stats['segmentation_seconds']:.4f}s, "
            f"embedding={stats['embedding_inference_seconds']:.4f}s, "
            f"non-embedding chunking="
            f"{stats['non_embedding_chunking_seconds']:.4f}s, "
            f"total processing={stats['total_processing_seconds']:.4f}s"
        )
        print(
            f"  reconstructs={stats['exact_source_reconstruction']}, "
            f"within_size_limit={stats['all_chunks_within_maximum_size']}"
        )
    print(
        f"Model revision: requested={MODEL_REVISION!r}, "
        f"resolved={_resolved_model_revision(model)!r}"
    )
    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help=(
            "New output directory; existing output files are never overwritten."
        ),
    )
    arguments = parser.parse_args()
    main(arguments.output_directory)
