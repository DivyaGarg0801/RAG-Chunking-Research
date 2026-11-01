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


def semantic_chunk(
    text: str,
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
