"""Experiment-aware logical-unit chunking for lab-manual text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.chunking.structure import is_experiment_heading, structure_chunk


@dataclass(frozen=True)
class LogicalUnitChunk:
    """One logical unit preserving the whole experiment context."""

    chunk_index: int
    text: str
    experiment_id: str | None
    experiment_title: str | None
    section: str | None
    parent_experiment_id: str | None
    experiment_occurrence_id: int | None = None
    parent_experiment_occurrence_id: int | None = None
    source_document: str | None = None
    source_pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class _FallbackBlock:
    """Simple internal container for fallback text that is not tied to an experiment."""

    text: str
    section: str | None = None
    experiment_id: str | None = None
    experiment_title: str | None = None
    parent_experiment_id: str | None = None
    experiment_occurrence_id: int | None = None
    parent_experiment_occurrence_id: int | None = None


def _parse_experiment_heading(heading: str) -> tuple[str | None, str | None]:
    """Extract the experiment identifier and optional title from a heading."""
    if not isinstance(heading, str):
        raise TypeError("heading must be a string.")

    normalized = heading.strip()
    if not normalized:
        return None, None

    match = re.match(
        r"^(?P<label>Experi(?:\s*ment|\s*met)|Exper[l1]\s*ment|Experment|Expt)"
        r"(?P<between>\s*[:.\-–—]\s*|\s*)"
        r"(?P<number>\d+)"
        r"(?:\s*(?P<after>[:\-–—])\s*(?P<title>.*?))?$",
        normalized,
        re.IGNORECASE,
    )
    if match is None:
        return None, None

    experiment_id = f"Experiment {match.group('number')}"
    title = (match.group("title") or "").strip()
    return experiment_id, title or None


def _flatten_chunks(parts: list[str]) -> list[str]:
    """Return a flat list of chunk texts that preserves the original order."""
    flattened: list[str] = []
    for part in parts:
        if part:
            flattened.append(part)
    return flattened


def _character_split(text: str, max_chunk_size: int) -> list[str]:
    """Safe character-based fallback that preserves every character exactly."""
    if len(text) <= max_chunk_size:
        return [text]
    return [text[index : index + max_chunk_size] for index in range(0, len(text), max_chunk_size)]


def _split_text_with_limit(text: str, max_chunk_size: int) -> list[str]:
    """Split text to comply with the configured maximum size without losing characters."""
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    try:
        structure_parts = structure_chunk(text, max_chunk_size=max_chunk_size)
        parts = [part.text for part in structure_parts]
        if parts and all(len(part) <= max_chunk_size for part in parts):
            return _flatten_chunks(parts)
    except RuntimeError:
        pass

    fallback = _character_split(text, max_chunk_size)
    if any(len(part) > max_chunk_size for part in fallback):
        raise RuntimeError(
            "The configured maximum chunk size could not be enforced without dropping characters."
        )
    return fallback


def _group_experiment_blocks(text: str) -> list[_FallbackBlock | dict[str, object]]:
    """Group text into experiment blocks and contiguous non-experiment fallback blocks."""
    if not text:
        return []

    sections = structure_chunk(text)
    if not sections:
        return [_FallbackBlock(text=text)]

    blocks: list[_FallbackBlock | dict[str, object]] = []
    current_experiment: dict[str, object] | None = None
    current_fallback_text = ""
    current_fallback_section: str | None = None
    experiment_occurrence_counter = 0
    last_occurrence_by_experiment_id: dict[str, int] = {}

    def flush_fallback() -> None:
        nonlocal current_fallback_text, current_fallback_section
        if current_fallback_text:
            blocks.append(
                _FallbackBlock(
                    text=current_fallback_text,
                    section=current_fallback_section,
                )
            )
            current_fallback_text = ""
            current_fallback_section = None

    def flush_experiment() -> None:
        nonlocal current_experiment
        if current_experiment is not None:
            blocks.append(current_experiment)
            current_experiment = None

    for chunk in sections:
        if is_experiment_heading(chunk.heading):
            flush_fallback()
            flush_experiment()

            experiment_id, experiment_title = _parse_experiment_heading(chunk.heading or "")
            if experiment_id is None:
                current_fallback_text += chunk.text
                continue

            experiment_occurrence_counter += 1
            parent_experiment_id = None
            parent_occurrence_id = None
            if chunk.parent_heading is not None:
                candidate_parent_id, _ = _parse_experiment_heading(chunk.parent_heading)
                if candidate_parent_id is not None:
                    parent_experiment_id = candidate_parent_id
                    parent_occurrence_id = last_occurrence_by_experiment_id.get(candidate_parent_id)

            current_experiment = {
                "text": chunk.text,
                "experiment_id": experiment_id,
                "experiment_title": experiment_title,
                "parent_experiment_id": parent_experiment_id,
                "experiment_occurrence_id": experiment_occurrence_counter,
                "parent_experiment_occurrence_id": parent_occurrence_id,
                "section": chunk.heading,
            }
            last_occurrence_by_experiment_id[experiment_id] = experiment_occurrence_counter
            continue

        if current_experiment is not None:
            current_experiment["text"] = str(current_experiment["text"]) + chunk.text
            continue

        if chunk.heading is not None:
            current_fallback_section = chunk.heading
        current_fallback_text += chunk.text

    flush_experiment()
    flush_fallback()

    if not blocks:
        return [_FallbackBlock(text=text)]
    return blocks


def logical_unit_chunk(
    text: str,
    max_chunk_size: int | None = None,
    *,
    source_document: str | None = None,
    source_pages: tuple[int, ...] = (),
) -> list[LogicalUnitChunk]:
    """Chunk text by entire experiment while preserving both source and experiment metadata."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if max_chunk_size is not None:
        if (
            not isinstance(max_chunk_size, int)
            or isinstance(max_chunk_size, bool)
            or max_chunk_size <= 0
        ):
            raise ValueError("max_chunk_size must be a positive integer or None.")
    if not text:
        return []

    blocks = _group_experiment_blocks(text)
    output: list[LogicalUnitChunk] = []

    for block in blocks:
        if isinstance(block, _FallbackBlock):
            block_text = block.text
            if max_chunk_size is None or len(block_text) <= max_chunk_size:
                parts = [block_text]
            else:
                parts = _split_text_with_limit(block_text, max_chunk_size)

            for part in parts:
                if not part:
                    continue
                output.append(
                    LogicalUnitChunk(
                        chunk_index=len(output) + 1,
                        text=part,
                        experiment_id=None,
                        experiment_title=None,
                        section=block.section,
                        parent_experiment_id=None,
                        experiment_occurrence_id=None,
                        parent_experiment_occurrence_id=None,
                        source_document=source_document,
                        source_pages=tuple(source_pages),
                    )
                )
            continue

        block_text = str(block["text"])
        experiment_id = block["experiment_id"]
        experiment_title = block["experiment_title"]
        parent_experiment_id = block["parent_experiment_id"]
        experiment_occurrence_id = block["experiment_occurrence_id"]
        parent_experiment_occurrence_id = block["parent_experiment_occurrence_id"]
        section = block["section"]

        if max_chunk_size is None or len(block_text) <= max_chunk_size:
            output.append(
                LogicalUnitChunk(
                    chunk_index=len(output) + 1,
                    text=block_text,
                    experiment_id=experiment_id,
                    experiment_title=experiment_title,
                    section=section,
                    parent_experiment_id=parent_experiment_id,
                    experiment_occurrence_id=experiment_occurrence_id,
                    parent_experiment_occurrence_id=parent_experiment_occurrence_id,
                    source_document=source_document,
                    source_pages=tuple(source_pages),
                )
            )
            continue

        candidate_parts: list[str]
        try:
            split_parts = structure_chunk(block_text, max_chunk_size=max_chunk_size)
            candidate_parts = [part.text for part in split_parts if part.text]
        except RuntimeError:
            candidate_parts = []

        if not candidate_parts or any(len(part) > max_chunk_size for part in candidate_parts):
            candidate_parts = _split_text_with_limit(block_text, max_chunk_size)

        for part in candidate_parts:
            if not part:
                continue
            part_section = section
            if not part_section:
                for candidate in structure_chunk(part):
                    if candidate.heading is not None and not is_experiment_heading(candidate.heading):
                        part_section = candidate.heading
                        break
            output.append(
                LogicalUnitChunk(
                    chunk_index=len(output) + 1,
                    text=part,
                    experiment_id=experiment_id,
                    experiment_title=experiment_title,
                    section=part_section,
                    parent_experiment_id=parent_experiment_id,
                    experiment_occurrence_id=experiment_occurrence_id,
                    parent_experiment_occurrence_id=parent_experiment_occurrence_id,
                    source_document=source_document,
                    source_pages=tuple(source_pages),
                )
            )

    if not output:
        return []

    if max_chunk_size is not None:
        for chunk in output:
            if len(chunk.text) > max_chunk_size:
                raise RuntimeError(
                    "A logical-unit chunk exceeds the configured maximum chunk size."
                )

    reconstructed = "".join(chunk.text for chunk in output)
    if reconstructed != text:
        raise RuntimeError("Logical-unit chunking did not preserve the source text exactly.")

    for index, chunk in enumerate(output, start=1):
        if chunk.chunk_index != index:
            raise RuntimeError("Logical-unit chunk indexes are inconsistent.")

    return output
