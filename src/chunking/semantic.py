<<<<<<< HEAD
"""Sentence-embedding-based semantic text chunking."""

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_SEGMENTATION_METHOD = "rule_based_line_and_sentence_v1"
_CLOSING_PUNCTUATION = "\"'”’»)]}"
_ABBREVIATIONS = (
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "fig.",
    "eq.",
)


class SentenceEmbeddingModel(Protocol):
    """The small part of a sentence-transformer model used by this module."""

    max_seq_length: int
    tokenizer: Any

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
        """Return one vector for each input sentence."""

    def get_sentence_embedding_dimension(self) -> int | None:
        """Return the model's expected vector dimension when available."""


@dataclass(frozen=True)
class SemanticChunk:
    """One semantic chunk with its ordering and boundary information."""

    chunk_index: int
    character_count: int
    text: str
    boundary_before: str


@dataclass(frozen=True)
class SemanticChunkingMetadata:
    """Configuration and basic execution details for one chunking operation."""

    strategy_name: str
    model_name: str
    similarity_method: str
    similarity_threshold: float
    maximum_chunk_size: int
    batch_size: int
    device: str
    sentence_segmentation_approach: str
    original_segment_count: int
    embedding_input_count: int
    output_chunk_count: int
    execution_time_seconds: float
    model_loaded_during_call: bool


@dataclass(frozen=True)
class SemanticChunkingResult:
    """Chunks and reproducibility metadata returned by semantic chunking."""

    chunks: tuple[SemanticChunk, ...]
    metadata: SemanticChunkingMetadata


def load_sentence_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    revision: str | None = None,
) -> SentenceEmbeddingModel:
    """Load a local-capable Sentence Transformers model once.

    The returned model can be passed to later calls to reuse it across
    documents. The model weights are downloaded on first use if not cached.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")
    if revision is not None and (
        not isinstance(revision, str) or not revision.strip()
    ):
        raise ValueError("revision must be None or a non-empty string.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "Semantic chunking requires Sentence Transformers. Install the "
            "project dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    try:
        load_options = {"device": device}
        if revision is not None:
            load_options["revision"] = revision
        return SentenceTransformer(model_name, **load_options)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load embedding model '{model_name}' on '{device}': {error}"
        ) from error


def _is_abbreviation(line: str, period_index: int) -> bool:
    text_before_period = line[: period_index + 1].lower()
    return any(text_before_period.endswith(item) for item in _ABBREVIATIONS)


def _split_line_into_sentences(line: str) -> list[str]:
    """Split one line conservatively while retaining punctuation and spaces."""
    if not line or line.isspace():
        return [line] if line else []

    if re.match(r"\s*\d+[.)]\s+", line):
        return [line]

    pieces: list[str] = []
    start = 0
    index = 0

    while index < len(line):
        character = line[index]
        if character not in ".!?":
            index += 1
            continue

        if character == ".":
            previous_is_digit = index > 0 and line[index - 1].isdigit()
            next_is_digit = index + 1 < len(line) and line[index + 1].isdigit()
            if (
                (previous_is_digit and next_is_digit)
                or _is_abbreviation(line, index)
            ):
                index += 1
                continue

        boundary_end = index + 1
        while (
            boundary_end < len(line)
            and line[boundary_end] in _CLOSING_PUNCTUATION
        ):
            boundary_end += 1

        if boundary_end < len(line) and line[boundary_end].isspace():
            while boundary_end < len(line) and line[boundary_end].isspace():
                boundary_end += 1
            pieces.append(line[start:boundary_end])
            start = boundary_end
            index = boundary_end
        else:
            index += 1

    if start < len(line):
        pieces.append(line[start:])

    return pieces


def _is_code_like_indented_line(line: str) -> bool:
    """Keep clearly code-like indented lines intact during sentence splitting."""
    indentation = len(line) - len(line.lstrip(" \t"))
    if not indentation:
        return False

    content = line[indentation:].lstrip()
    return bool(
        re.match(
            r"(?:"
            r"#|"
            r"(?:from|import|def|class|return|print|raise|pass|break|continue)\b|"
            r"(?:if|elif|else|for|while|try|except|with)\b.*:|"
            r"[\w.]+\s*(?:\+=|-=|\*=|/=|:=|=(?!=))|"
            r"[{}\[\]();]"
            r")",
            content,
        )
    )


def segment_text(text: str) -> list[str]:
    """Return ordered sentence-like segments whose concatenation equals text.

    Indentation is attached to the first sentence on a line. Numbered procedure
    lines and clearly code-like indented lines remain intact.
    """
    if not text:
        return []

    segments: list[str] = []
    pending_whitespace = ""
    line_start = 0

    while line_start < len(text):
        newline_index = text.find("\n", line_start)
        line_end = len(text) if newline_index == -1 else newline_index + 1
        line = text[line_start:line_end]
        line_ending = "\n" if line.endswith("\n") else ""
        line_content = line[:-1] if line_ending else line
        indentation_length = len(line_content) - len(
            line_content.lstrip(" \t")
        )
        indentation = line_content[:indentation_length]
        content = line_content[indentation_length:]

        if not content:
            whitespace = indentation + line_ending
            if segments:
                segments[-1] += whitespace
            else:
                pending_whitespace += whitespace
            line_start = line_end
            continue

        if re.match(r"\d+[.)]\s+", content) or _is_code_like_indented_line(line):
            pieces = [content + line_ending]
        else:
            pieces = _split_line_into_sentences(content + line_ending)

        if pending_whitespace:
            indentation = pending_whitespace + indentation
            pending_whitespace = ""
        if indentation:
            pieces[0] = indentation + pieces[0]
        segments.extend(pieces)

        line_start = line_end

    if pending_whitespace:
        if segments:
            segments[-1] += pending_whitespace
        else:
            segments.append(pending_whitespace)

    if "".join(segments) != text:
        raise RuntimeError("Sentence segmentation did not preserve the source text.")

    return segments


def _token_count(tokenizer: Any, text: str) -> int:
    """Count the exact tokens the model would receive, including special tokens."""
    try:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "The semantic model tokenizer must support encode(..., "
            "add_special_tokens=True, truncation=False)."
        ) from error
    if not hasattr(token_ids, "__len__"):
        raise ValueError("The semantic model tokenizer returned invalid token IDs.")
    return len(token_ids)


def _fitting_prefix_length(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
    total_tokens: int,
) -> int:
    """Return a directly verified prefix without assuming token-count monotonicity.

    The initial estimate uses the token count of the complete text. If that
    candidate does not fit, it is repeatedly shortened and retokenized. The
    returned prefix has itself been checked with truncation disabled.
    """
    if total_tokens <= maximum_tokens:
        return len(text)

    candidate_length = max(
        1,
        len(text) * maximum_tokens // total_tokens,
    )
    while candidate_length:
        if (
            _token_count(tokenizer, text[:candidate_length])
            <= maximum_tokens
        ):
            return candidate_length
        if candidate_length == 1:
            break
        candidate_length = max(1, candidate_length // 2)

    return 0


def _split_for_model_limit(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
) -> list[str]:
    """Split a source span at whitespace when possible, within a token limit."""
    pieces: list[str] = []
    start = 0

    while start < len(text):
        remaining = text[start:]
        remaining_tokens = _token_count(tokenizer, remaining)
        if remaining_tokens <= maximum_tokens:
            pieces.append(remaining)
            break

        fitting_end = _fitting_prefix_length(
            remaining,
            tokenizer,
            maximum_tokens,
            remaining_tokens,
        )
        if fitting_end == 0:
            raise ValueError(
                "The model maximum sequence length is too small to encode "
                "even one source character with its special tokens."
            )

        natural_ends = [
            index + 1
            for index in range(fitting_end)
            if remaining[index].isspace()
        ]
        end = fitting_end
        if natural_ends:
            natural_end = natural_ends[-1]
            if (
                _token_count(tokenizer, remaining[:natural_end])
                <= maximum_tokens
            ):
                end = natural_end
        piece = remaining[:end]
        if _token_count(tokenizer, piece) > maximum_tokens:
            raise RuntimeError("A semantic embedding input exceeds the model limit.")
        pieces.append(piece)
        start += end

    return pieces


def _prepare_model_segments(
    model: SentenceEmbeddingModel,
    segments: list[str],
) -> tuple[list[str], list[int]]:
    """Split source spans and retain each embedding input's original segment ID."""
    maximum_tokens = getattr(model, "max_seq_length", None)
    tokenizer = getattr(model, "tokenizer", None)
    if (
        not isinstance(maximum_tokens, int)
        or isinstance(maximum_tokens, bool)
        or maximum_tokens <= 0
    ):
        raise ValueError("The semantic model must expose a positive max_seq_length.")
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise ValueError("The semantic model must expose its tokenizer.")

    prepared_segments: list[str] = []
    original_segment_ids: list[int] = []
    for segment_id, segment in enumerate(segments):
        pieces = _split_for_model_limit(segment, tokenizer, maximum_tokens)
        prepared_segments.extend(pieces)
        original_segment_ids.extend([segment_id] * len(pieces))
    if "".join(prepared_segments) != "".join(segments):
        raise RuntimeError("Token-limit splitting did not preserve source text.")
    return prepared_segments, original_segment_ids


def _get_normalized_embeddings(
    model: SentenceEmbeddingModel,
    segments: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode all segments in internal batches and validate the vectors."""
    encoded = model.encode(
        segments,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=device,
    )
    try:
        embeddings = np.asarray(encoded, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("The model returned embeddings that are not a numeric matrix.") from error

    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(segments)
        or embeddings.shape[1] == 0
    ):
        raise ValueError(
            "The model must return one non-empty embedding vector per input segment."
        )

    get_dimension = getattr(model, "get_sentence_embedding_dimension", None)
    expected_dimension = get_dimension() if callable(get_dimension) else None
    if (
        expected_dimension is not None
        and embeddings.shape[1] != expected_dimension
    ):
        raise ValueError(
            "The embedding dimension does not match the model's declared dimension "
            f"({embeddings.shape[1]} != {expected_dimension})."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("The model returned an embedding with a non-finite value.")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("The model returned a zero-length embedding vector.")

    return embeddings / norms[:, np.newaxis]


def _split_oversized_text(text: str, maximum_size: int) -> list[str]:
    """Split long text at a nearby whitespace boundary, or by characters."""
    pieces: list[str] = []
    start = 0

    while start < len(text):
        limit = min(start + maximum_size, len(text))
        if limit == len(text):
            end = limit
        else:
            whitespace_positions = [
                index + 1
                for index in range(start, limit)
                if text[index].isspace()
            ]
            end = max(whitespace_positions, default=limit)

        pieces.append(text[start:end])
        start = end

    return pieces


def _validate_settings(
    chunk_size: int,
    similarity_threshold: float,
    batch_size: int,
    model_name: str,
    device: str,
) -> None:
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")
    if (
        not isinstance(similarity_threshold, (int, float))
        or isinstance(similarity_threshold, bool)
        or not math.isfinite(similarity_threshold)
        or not 0 <= similarity_threshold <= 1
    ):
        raise ValueError("similarity_threshold must be a finite number from 0 to 1.")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")
=======
"""Semantic chunking based on sentence similarity and topic change detection."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled by the local fallback path.
    SentenceTransformer = None  # type: ignore[assignment]


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving non-empty content."""
    if not text or not text.strip():
        return []

    pieces = [piece.strip() for piece in re.split(_SENTENCE_BOUNDARY_RE, text) if piece.strip()]
    return pieces


def _tokenize(text: str) -> list[str]:
    """Return lowercase word tokens for simple semantic similarity."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _token_counts(text: str) -> Counter[str]:
    """Count token frequency for a lightweight fallback embedding."""
    return Counter(_tokenize(text))


def _cosine_similarity(left: Sequence[float] | dict[str, float], right: Sequence[float] | dict[str, float]) -> float:
    """Compute cosine similarity for numeric vectors or token-frequency mappings."""
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        if not keys:
            return 0.0
        left_values = [float(left.get(key, 0.0)) for key in keys]
        right_values = [float(right.get(key, 0.0)) for key in keys]
        left_norm = math.sqrt(sum(value * value for value in left_values))
        right_norm = math.sqrt(sum(value * value for value in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left_values, right_values))
        return dot_product / (left_norm * right_norm)

    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        length = min(len(left_values), len(right_values))
        left_values = left_values[:length]
        right_values = right_values[:length]
    if not left_values or not right_values:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    return dot_product / (left_norm * right_norm)


def _similarity_for_pair(left: str, right: str, use_model: bool) -> float:
    """Compare two sentences using a real model when available, otherwise token overlap."""
    if use_model and SentenceTransformer is not None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode([left, right], convert_to_numpy=True, show_progress_bar=False)
        return _cosine_similarity(embeddings[0], embeddings[1])
    return _cosine_similarity(_token_counts(left), _token_counts(right))
>>>>>>> origin/master


def semantic_chunk(
    text: str,
<<<<<<< HEAD
    chunk_size: int = 1000,
    similarity_threshold: float = 0.5,
    batch_size: int = 32,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    model: SentenceEmbeddingModel | None = None,
    input_segments: Sequence[str] | None = None,
    segmentation_approach: str | None = None,
) -> SemanticChunkingResult:
    """Group adjacent sentence embeddings by cosine similarity and size.

    A boundary is semantic when adjacent normalized sentence embeddings have
    cosine similarity below the configured threshold. Size-driven splits use
    whitespace where possible and character boundaries otherwise. No overlap
    is added, so concatenating chunk text reconstructs the input exactly.

    ``input_segments`` allows a separately documented segmenter to provide
    source-preserving segments. When supplied, their concatenation must equal
    ``text`` exactly.
    """
    _validate_settings(
        chunk_size,
        similarity_threshold,
        batch_size,
        model_name,
        device,
    )
    started_at = time.perf_counter()
    if input_segments is None:
        if segmentation_approach is not None:
            raise ValueError(
                "segmentation_approach can only be set with input_segments."
            )
        segments = segment_text(text)
        segmentation_method = SENTENCE_SEGMENTATION_METHOD
    else:
        if isinstance(input_segments, (str, bytes)):
            raise TypeError("input_segments must be a sequence of text segments.")
        segments = list(input_segments)
        if any(not isinstance(segment, str) for segment in segments):
            raise TypeError("Every input segment must be a string.")
        if any(not segment for segment in segments):
            raise ValueError("input_segments cannot contain empty segments.")
        if "".join(segments) != text:
            raise ValueError(
                "Concatenated input_segments must exactly match text."
            )
        if segmentation_approach is None:
            segmentation_method = "caller_provided_segments"
        elif not isinstance(segmentation_approach, str) or not segmentation_approach.strip():
            raise ValueError("segmentation_approach must be a non-empty string.")
        else:
            segmentation_method = segmentation_approach

    if not text:
        return SemanticChunkingResult(
            chunks=(),
            metadata=SemanticChunkingMetadata(
                strategy_name="semantic",
                model_name=model_name,
                similarity_method="cosine_similarity",
                similarity_threshold=float(similarity_threshold),
                maximum_chunk_size=chunk_size,
                batch_size=batch_size,
                device=device,
                sentence_segmentation_approach=segmentation_method,
                original_segment_count=0,
                embedding_input_count=0,
                output_chunk_count=0,
                execution_time_seconds=time.perf_counter() - started_at,
                model_loaded_during_call=False,
            ),
        )

    if not any(not segment.isspace() for segment in segments):
        pieces = _split_oversized_text(text, chunk_size)
        chunks = tuple(
            SemanticChunk(
                chunk_index=index,
                character_count=len(piece),
                text=piece,
                boundary_before="document_start" if index == 1 else "size",
            )
            for index, piece in enumerate(pieces, start=1)
        )
        return SemanticChunkingResult(
            chunks=chunks,
            metadata=SemanticChunkingMetadata(
                strategy_name="semantic",
                model_name=model_name,
                similarity_method="cosine_similarity",
                similarity_threshold=float(similarity_threshold),
                maximum_chunk_size=chunk_size,
                batch_size=batch_size,
                device=device,
                sentence_segmentation_approach=segmentation_method,
                original_segment_count=len(segments),
                embedding_input_count=0,
                output_chunk_count=len(chunks),
                execution_time_seconds=time.perf_counter() - started_at,
                model_loaded_during_call=False,
            ),
        )

    model_was_loaded = model is None
    if model is None:
        model = load_sentence_model(model_name, device)
    original_segment_count = len(segments)
    segments, original_segment_ids = _prepare_model_segments(model, segments)

    normalized_embeddings = _get_normalized_embeddings(
        model,
        segments,
        batch_size,
        device,
    )
    adjacent_similarities = np.sum(
        normalized_embeddings[:-1] * normalized_embeddings[1:],
        axis=1,
    )

    raw_chunks: list[tuple[str, str]] = []
    current_text = ""
    next_boundary = "document_start"

    def append_current() -> None:
        nonlocal current_text
        if current_text:
            raw_chunks.append((current_text, next_boundary))
            current_text = ""

    for index, segment in enumerate(segments):
        if (
            index > 0
            and original_segment_ids[index - 1] != original_segment_ids[index]
            and adjacent_similarities[index - 1] < similarity_threshold
        ):
            append_current()
            next_boundary = "semantic"

        if len(segment) > chunk_size:
            append_current()
            parts = _split_oversized_text(segment, chunk_size)
            for part_index, part in enumerate(parts):
                raw_chunks.append(
                    (
                        part,
                        next_boundary if part_index == 0 else "size",
                    )
                )
            next_boundary = "size"
        elif current_text and len(current_text) + len(segment) > chunk_size:
            append_current()
            next_boundary = "size"
            current_text = segment
        else:
            current_text += segment

    append_current()

    chunks = tuple(
        SemanticChunk(
            chunk_index=index,
            character_count=len(chunk_text),
            text=chunk_text,
            boundary_before=boundary_before,
        )
        for index, (chunk_text, boundary_before) in enumerate(raw_chunks, start=1)
    )
    if any(not chunk.text for chunk in chunks):
        raise RuntimeError("Semantic chunking produced an empty chunk.")
    if any(chunk.character_count > chunk_size for chunk in chunks):
        raise RuntimeError("Semantic chunking exceeded the configured maximum size.")
    if "".join(chunk.text for chunk in chunks) != text:
        raise RuntimeError("Semantic chunking did not preserve the source text.")

    metadata = SemanticChunkingMetadata(
        strategy_name="semantic",
        model_name=model_name,
        similarity_method="cosine_similarity",
        similarity_threshold=float(similarity_threshold),
        maximum_chunk_size=chunk_size,
        batch_size=batch_size,
        device=device,
        sentence_segmentation_approach=segmentation_method,
        original_segment_count=original_segment_count,
        embedding_input_count=len(segments),
        output_chunk_count=len(chunks),
        execution_time_seconds=time.perf_counter() - started_at,
        model_loaded_during_call=model_was_loaded,
    )
    return SemanticChunkingResult(chunks=chunks, metadata=metadata)
=======
    chunk_size: int = 250,
    similarity_threshold: float = 0.25,
    use_model: bool = False,
) -> list[str]:
    """Group text by sentence similarity while respecting a maximum chunk size.

    This implementation is intentionally conservative: it will split when the topic
    changes enough for the similarity between adjacent sentences to drop below the
    configured threshold, or when the next sentence would exceed the chunk size.
    The default behavior uses a lightweight local token-overlap similarity so the
    project remains runnable even before a sentence-transformers model is installed.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
        raise ValueError("similarity_threshold must be a number between 0 and 1.")
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1.")
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_length = len(sentences[0])

    for sentence in sentences[1:]:
        next_length = current_length + 1 + len(sentence)
        if current and current_length > 0 and next_length > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            current_length = len(sentence)
            continue

        similarity = _similarity_for_pair(current[-1], sentence, use_model=use_model)
        if similarity < similarity_threshold and len(current) > 1:
            chunks.append(" ".join(current))
            current = [sentence]
            current_length = len(sentence)
        else:
            current.append(sentence)
            current_length = next_length

    if current:
        chunks.append(" ".join(current))

    return [chunk.strip() for chunk in chunks if chunk.strip()]
>>>>>>> origin/master
