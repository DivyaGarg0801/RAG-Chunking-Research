"""Recursive text chunking that prefers increasingly smaller boundaries."""

import re
from dataclasses import dataclass


_SEPARATOR_PATTERNS = (
    re.compile(r"\n[ \t]*\n+"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"\s+"),
)
_OVERLAP_BOUNDARY_PATTERNS = _SEPARATOR_PATTERNS[::2] + (
    _SEPARATOR_PATTERNS[3],
)


@dataclass(frozen=True)
class RecursiveChunk:
    """Chunk text and the number of characters repeated from the prior chunk."""

    text: str
    overlap_length: int


def _split_after_matches(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split after separator matches so every character stays in a piece."""
    pieces: list[str] = []
    start = 0

    for match in pattern.finditer(text):
        pieces.append(text[start : match.end()])
        start = match.end()

    if start < len(text):
        pieces.append(text[start:])

    return pieces


def _merge_pieces(pieces: list[str], maximum_size: int) -> list[str]:
    """Greedily merge adjacent pieces while their combined size fits."""
    merged: list[str] = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= maximum_size:
            current += piece
        else:
            if current:
                merged.append(current)
            current = piece

    if current:
        merged.append(current)

    return merged


def _split_recursively(
    text: str,
    maximum_size: int,
    patterns: tuple[re.Pattern[str], ...] = _SEPARATOR_PATTERNS,
) -> list[str]:
    """Split oversized text using the first available separator."""
    if len(text) <= maximum_size:
        return [text] if text else []

    if not patterns:
        return [
            text[start : start + maximum_size]
            for start in range(0, len(text), maximum_size)
        ]

    pieces = _split_after_matches(text, patterns[0])
    if len(pieces) == 1:
        return _split_recursively(text, maximum_size, patterns[1:])

    smaller_pieces: list[str] = []
    for piece in pieces:
        if len(piece) > maximum_size:
            smaller_pieces.extend(
                _split_recursively(piece, maximum_size, patterns[1:])
            )
        else:
            smaller_pieces.append(piece)

    return _merge_pieces(smaller_pieces, maximum_size)


def _find_overlap_start(text: str, end: int, requested_overlap: int) -> int:
    """Choose a nearby natural boundary for overlap, or use exact characters."""
    if requested_overlap == 0:
        return end

    earliest_start = max(0, end - requested_overlap)
    boundary_positions = {
        match.end()
        for pattern in _OVERLAP_BOUNDARY_PATTERNS
        for match in pattern.finditer(text, 0, end)
        if earliest_start <= match.end() < end
    }

    if boundary_positions:
        return min(boundary_positions)
    return earliest_start


def recursive_chunk(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[RecursiveChunk]:
    """Split text by paragraph, line, sentence, word, then character boundaries.

    Overlap is a maximum target. When possible, it begins at the earliest valid
    paragraph, sentence, or word boundary within the requested range. This
    gives the largest natural-boundary overlap that does not exceed the target.
    If no such boundary exists, the requested character count is used exactly.
    """
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be greater than zero.")
    if (
        not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError("overlap must be at least zero and less than chunk_size.")
    if not text:
        return []

    chunks: list[RecursiveChunk] = []
    source_position = 0

    while source_position < len(text):
        chunk_start = (
            _find_overlap_start(text, source_position, overlap)
            if chunks
            else source_position
        )
        actual_overlap = source_position - chunk_start
        available_size = chunk_size - actual_overlap
        next_pieces = _split_recursively(text[source_position:], available_size)
        new_text = next_pieces[0]

        if not new_text:
            raise RuntimeError("Recursive splitting did not advance through the text.")

        chunks.append(
            RecursiveChunk(
                text=text[chunk_start:source_position] + new_text,
                overlap_length=actual_overlap,
            )
        )
        source_position += len(new_text)

    return chunks
