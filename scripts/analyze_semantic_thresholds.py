"""Analyze semantic threshold sensitivity with one local embedding pass."""

import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

from src.chunking.semantic import (
    DEFAULT_MODEL_NAME,
    SemanticChunkingResult,
    _prepare_model_segments,
    load_sentence_model,
    semantic_chunk,
    segment_text,
)
from src.ingestion.pdf_loader import extract_pdf_pages
from src.preprocessing.text_cleaner import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = PROJECT_ROOT / "data" / "raw" / "computer_networks_lab_manual.pdf"
BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "semantic_preview"
    / "summary.json"
)
BASELINE_CHUNKS = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "semantic_preview"
    / "chunk_metadata.json"
)
REPORT_PATH = PROJECT_ROOT / "docs" / "semantic_chunking_diagnostics.md"

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
CHUNK_SIZE = 1000
BATCH_SIZE = 32
DEVICE = "cpu"
PAGE_SEPARATOR = "\n"


class CachedEmbeddingModel:
    """Return one already-generated embedding matrix to each threshold run."""

    def __init__(
        self,
        model: Any,
        segments: list[str],
        embeddings: np.ndarray,
    ) -> None:
        self.max_seq_length = model.max_seq_length
        self.tokenizer = model.tokenizer
        self._segments = segments
        self._embeddings = embeddings
        self._dimension = model.get_sentence_embedding_dimension()
        self.cache_hits = 0

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        device: str,
    ) -> np.ndarray:
        if sentences != self._segments:
            raise RuntimeError(
                "A threshold run requested different embedding inputs."
            )
        self.cache_hits += 1
        return self._embeddings

    def get_sentence_embedding_dimension(self) -> int | None:
        return self._dimension


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in (
        "PyMuPDF",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Could not determine version for {package}.") from error
    return versions


def _normalized_embeddings(model: Any, segments: list[str]) -> tuple[np.ndarray, float]:
    started_at = time.perf_counter()
    encoded = model.encode(
        segments,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=DEVICE,
    )
    embedding_seconds = time.perf_counter() - started_at

    embeddings = np.asarray(encoded, dtype=np.float64)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(segments)
        or embeddings.shape[1] == 0
    ):
        raise ValueError("The model did not return one vector per embedding input.")
    if not np.isfinite(embeddings).all():
        raise ValueError("The model returned a non-finite embedding.")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("The model returned a zero-length embedding.")
    return embeddings, embedding_seconds


def _result_row(
    result: SemanticChunkingResult,
    threshold: float,
    adjacent_similarities: np.ndarray,
) -> dict[str, int | float]:
    lengths = [chunk.character_count for chunk in result.chunks]
    pair_count = len(adjacent_similarities)
    below_count = int(np.count_nonzero(adjacent_similarities < threshold))
    return {
        "threshold": threshold,
        "chunks": len(result.chunks),
        "average_length": sum(lengths) / len(lengths) if lengths else 0,
        "minimum_length": min(lengths, default=0),
        "maximum_length": max(lengths, default=0),
        "semantic_boundaries": sum(
            chunk.boundary_before == "semantic" for chunk in result.chunks
        ),
        "size_boundaries": sum(
            chunk.boundary_before == "size" for chunk in result.chunks
        ),
        "adjacent_pairs_below": below_count,
        "adjacent_pairs_percent_below": (
            below_count / pair_count * 100 if pair_count else 0
        ),
        "chunking_seconds": result.metadata.execution_time_seconds,
    }


def _similarity_statistics(similarities: np.ndarray) -> dict[str, float]:
    percentiles = np.percentile(similarities, [10, 25, 50, 75, 90])
    return {
        "minimum": float(np.min(similarities)),
        "maximum": float(np.max(similarities)),
        "mean": float(np.mean(similarities)),
        "median": float(np.median(similarities)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
    }


def _verify_baseline(
    baseline_summary: dict[str, Any],
    baseline_chunks: list[dict[str, Any]],
    combined_text: str,
    segments: list[str],
    prepared_segments: list[str],
    result: SemanticChunkingResult,
) -> None:
    if len(combined_text) != baseline_summary["total_cleaned_document_characters"]:
        raise RuntimeError("The current cleaned text differs from the saved preview.")
    if len(segments) != baseline_summary["number_of_original_segments"]:
        raise RuntimeError("Sentence segmentation differs from the saved preview.")
    if len(prepared_segments) != baseline_summary["number_of_embedding_inputs"]:
        raise RuntimeError("Token-safe segmentation differs from the saved preview.")

    saved_records = [
        (item["text"], item["boundary_before"]) for item in baseline_chunks
    ]
    current_records = [
        (chunk.text, chunk.boundary_before) for chunk in result.chunks
    ]
    if saved_records != current_records:
        raise RuntimeError(
            "The 0.5 threshold result does not match the saved preview; "
            "investigate model revision or runtime changes before interpreting."
        )


def _format_similarity_table(
    statistics: dict[str, float],
    thresholds: list[dict[str, int | float]],
) -> str:
    lines = [
        "| Statistic | Cosine similarity |",
        "|---|---:|",
        f"| Minimum | {statistics['minimum']:.4f} |",
        f"| Maximum | {statistics['maximum']:.4f} |",
        f"| Mean | {statistics['mean']:.4f} |",
        f"| Median | {statistics['median']:.4f} |",
        f"| 10th percentile | {statistics['p10']:.4f} |",
        f"| 25th percentile | {statistics['p25']:.4f} |",
        f"| 50th percentile | {statistics['p50']:.4f} |",
        f"| 75th percentile | {statistics['p75']:.4f} |",
        f"| 90th percentile | {statistics['p90']:.4f} |",
        "",
        "| Threshold | Adjacent pairs below threshold | Percent below |",
        "|---:|---:|---:|",
    ]
    for row in thresholds:
        lines.append(
            f"| {row['threshold']:.1f} | {row['adjacent_pairs_below']} | "
            f"{row['adjacent_pairs_percent_below']:.2f}% |"
        )
    return "\n".join(lines)


def _format_threshold_table(thresholds: list[dict[str, int | float]]) -> str:
    lines = [
        "| Threshold | Chunks | Average chars | Min | Max | Semantic boundaries | "
        "Size boundaries | Cached chunk time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in thresholds:
        lines.append(
            f"| {row['threshold']:.1f} | {row['chunks']} | "
            f"{row['average_length']:.2f} | {row['minimum_length']} | "
            f"{row['maximum_length']} | {row['semantic_boundaries']} | "
            f"{row['size_boundaries']} | {row['chunking_seconds']:.4f} |"
        )
    return "\n".join(lines)


def _build_report(
    baseline_summary: dict[str, Any],
    segments: list[str],
    prepared_segments: list[str],
    similarities: np.ndarray,
    statistics: dict[str, float],
    threshold_rows: list[dict[str, int | float]],
    embedding_seconds: float,
    cache_hits: int,
    resolved_revision: str | None,
    versions: dict[str, str],
) -> str:
    first_row = threshold_rows[0]
    baseline_row = next(row for row in threshold_rows if row["threshold"] == 0.5)
    last_row = threshold_rows[-1]
    pair_count = len(similarities)
    return f"""# Semantic Chunking: Real-Document Diagnostic Analysis

## Scope and reproducibility

This diagnostic uses `{baseline_summary['source_pdf_filename']}` with the existing
page extraction and cleaner, joined in page order with a newline. The cleaned input
contains {baseline_summary['total_cleaned_document_characters']:,} characters across
{baseline_summary['number_of_pages_extracted']} extracted pages. The original
semantic preview files were read for verification and were not modified.

The script loaded `{DEFAULT_MODEL_NAME}` from the local cache in offline mode;
it did not request model weights from the network. The model revision resolved
from the loaded configuration was `{resolved_revision or 'unavailable'}`. Runtime
versions: `{json.dumps(versions, sort_keys=True)}`.

The current segmenter produced {len(segments):,} original sentence-like segments.
After token-limit preparation there were {len(prepared_segments):,} embedding
inputs. One embedding pass took {embedding_seconds:.4f} seconds. The same
embedding matrix was reused for all five threshold runs; the diagnostic confirms
the saved preview's 0.5 result has identical chunk text and boundary labels.
There are {pair_count:,} adjacent embedding pairs for the similarity distribution.

## Adjacent cosine similarity distribution

{_format_similarity_table(statistics, threshold_rows)}

The similarity boundary condition is strict (`similarity < threshold`). Thus the
pair counts describe all adjacent embedding pairs; in the chunker, only pairs
between different original sentence-like segments can create a semantic boundary.
Here, the original segment and embedding input counts are
{len(segments):,} and {len(prepared_segments):,}, respectively.

## Threshold sensitivity

Maximum character size is held at {CHUNK_SIZE:,}, with batch size {BATCH_SIZE}
and device `{DEVICE}`. The model and cached embeddings are identical across rows.
“Cached chunk time” measures the same chunking/validation path using the saved
embedding matrix; it excludes the one-time {embedding_seconds:.4f}-second
embedding inference and is not end-to-end run time.

{_format_threshold_table(threshold_rows)}

At 0.3, {first_row['semantic_boundaries']} boundaries are semantic; at the current
0.5 threshold there are {baseline_row['semantic_boundaries']}; at 0.7 there are
{last_row['semantic_boundaries']}. A higher threshold is expected to classify more
adjacent pairs as below-threshold; this sensitivity table is a structural
diagnostic only, not a basis for selecting a preferred threshold or a retrieval
quality result.

## Qualitative fragmentation and causes

Examples below are taken directly from the saved preview output:

- **Experiment heading and aim:** Chunk 123 is `Experiment-1`, chunk 124 contains
  the Aim ending in “available in the Linux”, and chunk 125 is only `OS.`. The
  physical line break after “Linux” becomes separate segment input; the semantic
  boundary decision then makes that continuation a five-character chunk.
- **Procedure explanation split across physical lines:** chunks 134–136 divide
  “The ping command is usually used as a” / “simple way to verify...” / “computer
  or network device.” The extracted text contains line breaks at these positions;
  line-oriented segmentation supplies separate inputs and the semantic scores
  decide to split them.
- **Separated headings and content:** chunks 129 and 130 contain `Procedure:` and
  `Run the following commands...`; the headings are separate physical lines and
  the threshold places boundaries between those inputs.
- **Very short chunks and page numbers:** chunk 1 is `1`, chunk 11 is `2`, and
  chunk 141 is `7`. These are page-number text extracted from the document layout;
  semantic boundaries isolate them from the surrounding text.
- **Sentence endings and explanatory continuation:** chunks 462–464 split the
  DropTail/RED discussion at “In contrast to the more complex” / “algorithms like
  RED and WRED...”. The segmenter treated the source line break as an input
  boundary, and the semantic decision produced an incomplete phrase chunk.

These are interacting causes: PDF extraction exposes layout line breaks, the
rule-based segmenter treats physical lines conservatively as separate inputs,
and similarity decisions determine whether those neighboring inputs remain
together. The text is not lost or changed: the preview reconstructs all
{baseline_summary['total_cleaned_document_characters']:,} cleaned source characters.

### Extraction artifacts observed before chunking

The extracted page text itself contains examples such as
`Thisistocertifythatthisisabonafiderecord...` (merged words) and replacement
characters in `�networks�` and redacted/unrecognized glyph sequences. A URL is
also extracted in the form `available athttp://...`. These are already present
before segmentation and are not introduced by semantic boundary decisions.
The targeted scan did not confirm a specific separated-letter figure label in
this manual's extracted text; figure labels such as `Figure.1.` do appear.

## What the distribution indicates

At the current 0.5 threshold, {baseline_row['adjacent_pairs_below']} of
{pair_count} adjacent pairs ({baseline_row['adjacent_pairs_percent_below']:.2f}%)
are below threshold, matching the {baseline_row['semantic_boundaries']} semantic
boundaries in the saved preview. This explains the high boundary count: most
adjacent embeddings are classified as dissimilar under the current threshold.
The average chunk is only {baseline_row['average_length']:.2f} characters, and the
minimum is {baseline_row['minimum_length']} characters.

The threshold changes chunk grouping, not the underlying embedding space or
retrieval performance. No threshold is selected here. The five thresholds should
be compared using judged retrieval queries and the same retrieval setup when
that evaluation milestone is ready.

## Possible improvements and trade-offs (not implemented)

1. **More reliable sentence segmentation:** a local sentence-boundary model or
   library may reduce splits caused by PDF line wrapping, but adds dependency,
   runtime and reproducibility considerations and can still mishandle technical
   abbreviations.
2. **Keep headings with following content:** classify short headings and attach
   them to the next content segment. This is structure-aware behavior and may
   improve usability, but changes the method from purely adjacent embedding
   decisions and must be applied consistently and documented.
3. **Avoid tiny standalone chunks:** report or flag short chunks for inspection
   before adding a minimum size. A minimum-size constraint or merge policy can
   suppress useful boundaries, blur genuinely distinct topics, and bias results;
   it needs a documented rationale and validation rather than an arbitrary
   character cutoff.
4. **Merge incomplete fragments:** a rule that avoids boundaries inside an
   incomplete phrase could use punctuation/grammar signals, but simple
   punctuation heuristics are unreliable for headings, lists, and technical
   notation. Compare this as a separately specified variant.

## Recommended next experiment

First, inspect a stratified sample of segments and boundary decisions across the
five thresholds, including headings, numbered procedures, and very short
fragments. Then run the unchanged threshold variants against manually reviewed
queries and relevance judgements once retrieval evaluation is implemented.
Do not choose a threshold solely because it creates fewer chunks.
"""


def main() -> None:
    if not SOURCE_PDF.is_file():
        raise FileNotFoundError(f"Source PDF not found: {SOURCE_PDF}")
    if not BASELINE_SUMMARY.is_file() or not BASELINE_CHUNKS.is_file():
        raise FileNotFoundError(
            "The saved semantic preview summary and chunk metadata are required."
        )
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Diagnostic report already exists and will not be overwritten: {REPORT_PATH}"
        )

    baseline_summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline_chunks = json.loads(BASELINE_CHUNKS.read_text(encoding="utf-8"))
    extraction = extract_pdf_pages(SOURCE_PDF)
    cleaned_pages = [clean_text(page.text) for page in extraction.pages]
    combined_text = PAGE_SEPARATOR.join(cleaned_pages)
    original_segments = segment_text(combined_text)

    model = load_sentence_model(DEFAULT_MODEL_NAME, DEVICE)
    prepared_segments, original_segment_ids = _prepare_model_segments(
        model,
        original_segments,
    )
    if len(original_segment_ids) != len(prepared_segments):
        raise RuntimeError("Prepared segment lineage metadata is inconsistent.")
    raw_embeddings, embedding_seconds = _normalized_embeddings(
        model,
        prepared_segments,
    )
    normalized_for_similarity = raw_embeddings / np.linalg.norm(
        raw_embeddings,
        axis=1,
        keepdims=True,
    )
    similarities = np.sum(
        normalized_for_similarity[:-1] * normalized_for_similarity[1:],
        axis=1,
    )
    if not np.isfinite(similarities).all():
        raise ValueError("Cosine similarity calculation produced a non-finite value.")

    cached_model = CachedEmbeddingModel(
        model,
        prepared_segments,
        raw_embeddings,
    )
    threshold_results: list[SemanticChunkingResult] = []
    threshold_rows: list[dict[str, int | float]] = []
    for threshold in THRESHOLDS:
        result = semantic_chunk(
            combined_text,
            chunk_size=CHUNK_SIZE,
            similarity_threshold=threshold,
            batch_size=BATCH_SIZE,
            model_name=DEFAULT_MODEL_NAME,
            device=DEVICE,
            model=cached_model,
        )
        if "".join(chunk.text for chunk in result.chunks) != combined_text:
            raise RuntimeError(
                f"Threshold {threshold} did not reconstruct the source text."
            )
        if any(chunk.character_count > CHUNK_SIZE for chunk in result.chunks):
            raise RuntimeError(
                f"Threshold {threshold} produced an oversized final chunk."
            )
        threshold_results.append(result)
        threshold_rows.append(
            _result_row(result, threshold, similarities)
        )

    if cached_model.cache_hits != len(THRESHOLDS):
        raise RuntimeError("A threshold run did not reuse the embedding cache.")
    _verify_baseline(
        baseline_summary,
        baseline_chunks,
        combined_text,
        original_segments,
        prepared_segments,
        threshold_results[THRESHOLDS.index(0.5)],
    )

    model_module = model._first_module()
    auto_model = getattr(model_module, "auto_model", None)
    model_config = getattr(auto_model, "config", None)
    resolved_revision = getattr(model_config, "_commit_hash", None)
    if not isinstance(resolved_revision, str) or not resolved_revision:
        resolved_revision = None

    report = _build_report(
        baseline_summary,
        original_segments,
        prepared_segments,
        similarities,
        _similarity_statistics(similarities),
        threshold_rows,
        embedding_seconds,
        cached_model.cache_hits,
        resolved_revision,
        _package_versions(),
    )
    with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as report_file:
        report_file.write(report)

    print(f"Original segments: {len(original_segments)}")
    print(f"Embedding inputs: {len(prepared_segments)}")
    print(f"Adjacent pairs: {len(similarities)}")
    print(f"One-pass embedding time: {embedding_seconds:.4f} seconds")
    print(f"Cached embedding reuse count: {cached_model.cache_hits}")
    print(f"Similarity statistics: {_similarity_statistics(similarities)}")
    for row in threshold_rows:
        print(row)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
