"""Logical-unit chunking for experiment-based laboratory manuals."""

import re
from bisect import bisect_right
from dataclasses import dataclass


DEFAULT_MAX_CHUNK_SIZE = 1500

_EXPERIMENT_HEADING_PATTERN = re.compile(
    r"^(?P<label>Experi(?:\s*ment|\s*met)|Exper[l1]\s*ment|Experment|Expt)"
    r"(?P<between>\s*[:.\-–—]\s*|\s*)"
    r"(?P<number>\d+)"
    r"(?:\s*(?P<after>[:.\-–—])\s*(?P<title>.*?))?$",
    re.IGNORECASE,
)
_SECTION_LABELS = (
    "Aim",
    "Objective",
    "Theory",
    "Requirement",
    "Requirements",
    "Algorithm",
    "Procedure",
    "Program",
    "Output",
    "Observation",
    "Analysis",
    "Discussion",
    "Conclusion",
    "Result",
    "Summary",
    "Method",
    "Implementation",
    "Hypothesis",
    "Task",
)
_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?"
    r"(?P<label>" + "|".join(re.escape(label) for label in _SECTION_LABELS) + r")"
    r"(?:\s*[:\-–]\s*.*)?\s*$",
    re.IGNORECASE,
)
_SECTION_LABEL_PREFIX = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?"
    r"(?:" + "|".join(re.escape(label) for label in _SECTION_LABELS) + r")"
    r"\s*[:\-–]\s*$",
    re.IGNORECASE,
)
_GENERIC_COLON_HEADING = re.compile(
    r"^\s*(?P<title>[A-Za-z][A-Za-z0-9 &'()/,+\-–—]{0,79}):\s*$"
)
_GENERIC_COMMAND_START = re.compile(
    r"^(?:to|find|give|analyse|analyze|modify|filter|identify|run|enter|"
    r"click|execute|use|open|select|type|observe|record|connect|configure|"
    r"set|add|remove|create|start|stop|show|view|check|follow|study|sudo|"
    r"apt(?:-get)?|ping|nslookup|traceroute|tcpdump|wireshark|gcc|python|java)\b",
    re.IGNORECASE,
)
_GENERIC_SENTENCE_START = re.compile(
    r"^(?:in|for|at|numbers?|three|the following)\b",
    re.IGNORECASE,
)
_GENERIC_CLAUSE_VERB = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|has|have|had|can|could|"
    r"will|would|shall|should|must|do|does|did|travel|travels)\b",
    re.IGNORECASE,
)
_GENERIC_CODE_MARKER = re.compile(
    r"[`$=<>\\]|(?:\.\.?/)|(?:->|→)|(?:\s-[A-Za-z]\b)"
)
_GENERIC_CODE_LABELS = {
    "main",
    "else",
    "elif",
    "try",
    "except",
    "finally",
    "default",
    "case",
    "switch",
    "end",
}
_PARAGRAPH_BREAK = re.compile(r"(?:\r\n|\r|\n)[ \t]*(?:(?:\r\n|\r|\n))+")
_LINE_ENDING = re.compile(r"\r\n|\r|\n")
_CLOSING_PUNCTUATION = "\"'”’»)]}"
_LOGICAL_UNIT_TYPES = {
    "experiment",
    "experiment_context",
    "preamble",
    "unclassified",
}
_BOUNDARY_REASONS = {
    "unit_start",
    "subsection",
    "paragraph",
    "sentence",
    "line",
    "word",
    "character",
}
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
_MAX_ABBREVIATION_LENGTH = max(map(len, _ABBREVIATIONS))
_SENTENCE_LIKE_EXPERIMENT_TITLE = re.compile(
    r"^(?:the|a|an|this|these|those|we|they|he|she|it|in|on|at|during|"
    r"after|when|while|to|for|then)\b|\b(?:am|is|are|was|were|has|have|"
    r"had|can|could|will|would|should|must|sent|received|transmitted|"
    r"increased|decreased|occurred|shows|showed|depends|travels|travel)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogicalChunk:
    """A physical chunk and the logical experiment unit it belongs to."""

    chunk_index: int
    character_count: int
    text: str
    logical_unit_id: str
    logical_unit_title: str
    experiment_number: str | None
    parent_heading: str | None
    logical_unit_type: str
    section_heading: str | None
    is_split: bool
    boundary_reason: str
    source_document: str | None
    source_page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class _ExperimentHeading:
    number: str
    title: str | None
    heading: str
    is_outer: bool


@dataclass(frozen=True)
class _UnitMetadata:
    title: str
    experiment_number: str | None
    parent_heading: str | None
    unit_type: str


@dataclass(frozen=True)
class _LogicalUnit:
    start: int
    end: int
    metadata: _UnitMetadata


def _read_lines(text: str) -> list[tuple[int, int, str]]:
    """Return exact line spans as (start, end, content) tuples."""
    lines: list[tuple[int, int, str]] = []
    position = 0
    for line in text.splitlines(keepends=True):
        end = position + len(line)
        lines.append((position, end, line.rstrip("\r\n")))
        position = end
    if position < len(text):
        lines.append((position, len(text), text[position:]))
    return lines


def _parse_experiment_heading(line: str) -> _ExperimentHeading | None:
    """Recognize conservative experiment-heading appearances."""
    stripped = line.strip()
    if (
        not stripped
        or len(stripped) > 120
        or stripped.startswith(("#", "*", "-", "=", "_", "|"))
        or stripped.endswith(("?", "!"))
    ):
        return None

    match = _EXPERIMENT_HEADING_PATTERN.fullmatch(stripped)
    if match is None:
        return None

    title = (match.group("title") or "").strip()
    if (
        match.group("after") == "."
        and title
        and (
            not title[0].isupper()
            or _SENTENCE_LIKE_EXPERIMENT_TITLE.search(title) is not None
        )
    ):
        return None
    if title and (
        title.endswith((".", "!", "?"))
        or len(title.split()) > 12
    ):
        return None

    number = match.group("number")
    heading = f"Experiment {number}: {title}" if title else f"Experiment {number}"
    prefix_separator = match.group("between").strip()
    trailing_separator = match.group("after")
    is_outer = prefix_separator in {":", "-", "–", "—"} or (
        not title
        and trailing_separator is None
        and prefix_separator in {"", "."}
    )
    return _ExperimentHeading(
        number=number,
        title=title or None,
        heading=heading,
        is_outer=is_outer,
    )


def _section_heading(line: str) -> str | None:
    """Return a short recognized subsection heading, if the line is one."""
    match = _SECTION_PATTERN.fullmatch(line)
    if match is not None:
        return match.group("label").capitalize()

    generic_match = _GENERIC_COLON_HEADING.fullmatch(line)
    if generic_match is not None:
        title = generic_match.group("title").strip()
        if (
            not title[0].isupper()
            or _GENERIC_COMMAND_START.match(title) is not None
            or _GENERIC_SENTENCE_START.match(title) is not None
            or _GENERIC_CLAUSE_VERB.search(title) is not None
            or _GENERIC_CODE_MARKER.search(title) is not None
            or title.casefold() in _GENERIC_CODE_LABELS
        ):
            return None
        return title
    return None


def _find_units(text: str) -> list[_LogicalUnit]:
    """Find logical unit spans without relying on structure-chunk boundaries."""
    if not text:
        return []

    heading_events = [
        (line_start, heading)
        for line_start, _, line in _read_lines(text)
        if (heading := _parse_experiment_heading(line)) is not None
    ]
    units: list[_LogicalUnit] = []
    current_start = 0
    current_metadata = _UnitMetadata(
        title="Unclassified content",
        experiment_number=None,
        parent_heading=None,
        unit_type="unclassified",
    )
    active_outer: _ExperimentHeading | None = None
    child_started = False

    def finish(end: int, metadata: _UnitMetadata | None = None) -> None:
        if current_start < end:
            units.append(
                _LogicalUnit(
                    start=current_start,
                    end=end,
                    metadata=metadata or current_metadata,
                )
            )

    for line_start, heading in heading_events:
        if heading.is_outer:
            if current_start < line_start:
                if (
                    current_metadata.unit_type == "unclassified"
                    and active_outer is None
                ):
                    current_metadata = _UnitMetadata(
                        title="Preamble",
                        experiment_number=None,
                        parent_heading=None,
                        unit_type="preamble",
                    )
                finish(line_start)
            current_start = line_start
            current_metadata = _UnitMetadata(
                title=heading.heading,
                experiment_number=heading.number,
                parent_heading=None,
                unit_type="experiment",
            )
            active_outer = heading
            child_started = False
            continue

        if active_outer is not None:
            if not child_started:
                context_metadata = _UnitMetadata(
                    title=active_outer.heading,
                    experiment_number=active_outer.number,
                    parent_heading=None,
                    unit_type="experiment_context",
                )
                finish(line_start, context_metadata)
                current_start = line_start
                child_started = True
            else:
                finish(line_start)
                current_start = line_start

            child_metadata = _UnitMetadata(
                title=heading.heading,
                experiment_number=heading.number,
                parent_heading=active_outer.heading,
                unit_type="experiment",
            )
            current_metadata = child_metadata
            continue

        if current_start < line_start:
            if current_metadata.unit_type == "unclassified":
                current_metadata = _UnitMetadata(
                    title="Preamble",
                    experiment_number=None,
                    parent_heading=None,
                    unit_type="preamble",
                )
            finish(line_start)
        current_start = line_start
        current_metadata = _UnitMetadata(
            title=heading.heading,
            experiment_number=heading.number,
            parent_heading=None,
            unit_type="experiment",
        )

    finish(len(text))
    return units


def _sentence_boundaries(text: str) -> list[int]:
    """Find conservative sentence ends, including their following whitespace."""
    boundaries: list[int] = []
    for index, character in enumerate(text):
        if character not in ".!?":
            continue

        if character == ".":
            previous_is_digit = index > 0 and text[index - 1].isdigit()
            next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
            if (
                (previous_is_digit and next_is_digit)
                or _is_abbreviation_period(text, index)
            ):
                continue

        end = index + 1
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        if end < len(text) and text[end].isspace():
            while end < len(text) and text[end].isspace():
                end += 1
            boundaries.append(end)
    return boundaries


def _is_abbreviation_period(text: str, period_index: int) -> bool:
    """Check only a bounded suffix for abbreviations and measurement units."""
    end = period_index + 1
    start = max(0, end - _MAX_ABBREVIATION_LENGTH)
    for abbreviation in _ABBREVIATIONS:
        abbreviation_start = end - len(abbreviation)
        if abbreviation_start < start:
            continue
        if text[abbreviation_start:end].casefold() != abbreviation:
            continue
        if abbreviation_start and text[abbreviation_start - 1].isalnum():
            continue
        if abbreviation == "ms.":
            preceding = text[max(0, abbreviation_start - 32) : abbreviation_start]
            if re.search(r"\d+(?:\.\d+)?\s+$", preceding):
                return False
        return True
    return False


def _section_boundaries(text: str) -> tuple[list[int], list[tuple[int, str]]]:
    boundaries: list[int] = []
    headings: list[tuple[int, str]] = []
    for line_start, _, line in _read_lines(text):
        heading = _section_heading(line)
        if heading is not None:
            boundaries.append(line_start)
            headings.append((line_start, heading))
    return boundaries, headings


def _latest_boundary(
    boundaries: list[int],
    start: int,
    limit: int,
) -> int | None:
    candidates = [position for position in boundaries if start < position <= limit]
    return max(candidates) if candidates else None


def _split_unit(
    text: str,
    maximum_size: int,
) -> list[tuple[int, int, str]]:
    """Split a unit at the best available natural boundary, without loss."""
    if len(text) <= maximum_size:
        return [(0, len(text), "unit_start")]

    section_positions, _ = _section_boundaries(text)
    paragraph_positions = [
        match.end() for match in _PARAGRAPH_BREAK.finditer(text)
    ]
    sentence_positions = _sentence_boundaries(text)
    line_positions = [match.end() for match in _LINE_ENDING.finditer(text)]
    word_positions = [
        match.end()
        for match in re.finditer(r"\s+", text)
        if not any(
            section_start < match.end()
            and _SECTION_LABEL_PREFIX.fullmatch(
                text[section_start : match.end()]
            )
            for section_start in section_positions
        )
    ]

    chunks: list[tuple[int, int, str]] = []
    start = 0
    reason_before = "unit_start"
    while start < len(text):
        limit = min(start + maximum_size, len(text))
        if limit == len(text):
            chunks.append((start, limit, reason_before))
            break

        chosen: tuple[int, str] | None = None
        for reason, boundaries in (
            ("subsection", section_positions),
            ("paragraph", paragraph_positions),
            ("sentence", sentence_positions),
            ("line", line_positions),
            ("word", word_positions),
        ):
            boundary = _latest_boundary(boundaries, start, limit)
            # Use the latest fitting natural boundary; the order above breaks
            # ties in favor of the more meaningful boundary type.
            if boundary is not None and (
                chosen is None or boundary > chosen[0]
            ):
                chosen = (boundary, reason)

        if chosen is None:
            chosen = (limit, "character")
        end, next_reason = chosen
        if (
            end < len(text)
            and len(text) - end <= maximum_size
            and text[end:].isspace()
        ):
            final_chunk_start = len(text) - maximum_size
            alternatives = [
                (position, reason)
                for reason, boundaries in (
                    ("subsection", section_positions),
                    ("paragraph", paragraph_positions),
                    ("sentence", sentence_positions),
                    ("line", line_positions),
                    ("word", word_positions),
                )
                for position in boundaries
                if start < position < end and position >= final_chunk_start
            ]
            if alternatives:
                end, next_reason = max(
                    alternatives,
                    key=lambda candidate: candidate[0],
                )
            else:
                end = final_chunk_start
                next_reason = "character"
        chunks.append((start, end, reason_before))
        start = end
        reason_before = next_reason

    return chunks


def _page_numbers_for_span(
    start: int,
    end: int,
    page_break_offsets: tuple[int, ...],
) -> tuple[int, ...]:
    if not page_break_offsets or end <= start:
        return ()

    page_starts = tuple(offset + 1 for offset in page_break_offsets)
    first_page = bisect_right(page_starts, start)
    last_page = bisect_right(page_starts, end - 1)
    return tuple(range(first_page + 1, last_page + 2))


def _validate_chunks(
    chunks: list[LogicalChunk],
    source_text: str,
    max_chunk_size: int,
    source_document: str | None,
) -> None:
    if not isinstance(source_text, str):
        raise RuntimeError("The logical-unit source must be text.")
    if (
        not isinstance(max_chunk_size, int)
        or isinstance(max_chunk_size, bool)
        or max_chunk_size <= 0
    ):
        raise RuntimeError("The logical-unit maximum must be a positive integer.")
    if any(not isinstance(chunk, LogicalChunk) for chunk in chunks):
        raise RuntimeError("Logical-unit output contains an invalid chunk record.")
    if any(not isinstance(chunk.text, str) for chunk in chunks):
        raise RuntimeError("Logical-unit chunk text must be a string.")
    if "".join(chunk.text for chunk in chunks) != source_text:
        raise RuntimeError("Logical-unit chunking did not reconstruct the source text.")
    if any(not chunk.text for chunk in chunks):
        raise RuntimeError("Logical-unit chunking produced an empty chunk.")
    if any(
        not isinstance(chunk.character_count, int)
        or isinstance(chunk.character_count, bool)
        or chunk.character_count != len(chunk.text)
        for chunk in chunks
    ):
        raise RuntimeError("Logical-unit chunk character counts are inconsistent.")
    if any(len(chunk.text) > max_chunk_size for chunk in chunks):
        raise RuntimeError("A logical-unit chunk exceeds the configured maximum.")
    if any(
        not isinstance(chunk.chunk_index, int)
        or isinstance(chunk.chunk_index, bool)
        for chunk in chunks
    ) or [chunk.chunk_index for chunk in chunks] != list(
        range(1, len(chunks) + 1)
    ):
        raise RuntimeError("Logical-unit chunk indices are not sequential.")

    closed_ids: set[str] = set()
    previous_id: str | None = None
    unit_metadata: dict[str, tuple[object, ...]] = {}
    unit_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        if (
            not isinstance(chunk.logical_unit_id, str)
            or not chunk.logical_unit_id.strip()
            or not isinstance(chunk.logical_unit_title, str)
            or not chunk.logical_unit_title.strip()
        ):
            raise RuntimeError("Logical-unit identifiers and titles must be present.")
        if (
            not isinstance(chunk.logical_unit_type, str)
            or chunk.logical_unit_type not in _LOGICAL_UNIT_TYPES
        ):
            raise RuntimeError("A logical-unit chunk has an invalid unit type.")
        if chunk.logical_unit_type in {"experiment", "experiment_context"}:
            if (
                not isinstance(chunk.experiment_number, str)
                or not chunk.experiment_number.isdigit()
            ):
                raise RuntimeError("Experiment chunks must include a numeric identifier.")
            if chunk.logical_unit_type == "experiment_context":
                if chunk.parent_heading is not None:
                    raise RuntimeError(
                        "An outer experiment context cannot have a parent heading."
                    )
            elif chunk.parent_heading is not None and (
                not isinstance(chunk.parent_heading, str)
                or not chunk.parent_heading.strip()
            ):
                raise RuntimeError("A child experiment parent heading must be non-empty.")
        elif chunk.experiment_number is not None or chunk.parent_heading is not None:
            raise RuntimeError(
                "Preamble and unclassified chunks cannot have experiment metadata."
            )
        if chunk.section_heading is not None and (
            not isinstance(chunk.section_heading, str)
            or not chunk.section_heading.strip()
        ):
            raise RuntimeError("A section heading must be None or non-empty text.")
        if not isinstance(chunk.is_split, bool):
            raise RuntimeError("Logical-unit split status must be boolean.")
        if (
            not isinstance(chunk.boundary_reason, str)
            or chunk.boundary_reason not in _BOUNDARY_REASONS
        ):
            raise RuntimeError("Every chunk must have a recognized boundary reason.")
        if chunk.source_document is not None and (
            not isinstance(chunk.source_document, str)
            or not chunk.source_document.strip()
        ):
            raise RuntimeError("Source-document metadata must be non-empty text.")
        if chunk.source_document != source_document:
            raise RuntimeError("Source-document metadata is inconsistent.")
        if not isinstance(chunk.source_page_numbers, tuple) or any(
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
            for page_number in chunk.source_page_numbers
        ):
            raise RuntimeError("Source page numbers must be positive integers.")
        if tuple(sorted(set(chunk.source_page_numbers))) != chunk.source_page_numbers:
            raise RuntimeError("Source page numbers must be ordered and unique.")

        if chunk.logical_unit_id != previous_id:
            if previous_id is not None:
                closed_ids.add(previous_id)
            if chunk.logical_unit_id in closed_ids:
                raise RuntimeError("A logical-unit ID was reused after another unit.")
            if chunk.boundary_reason != "unit_start":
                raise RuntimeError("The first chunk of a unit must mark its start.")
            previous_id = chunk.logical_unit_id
        elif chunk.boundary_reason == "unit_start":
            raise RuntimeError("A split unit cannot restart its unit-start boundary.")

        metadata = (
            chunk.logical_unit_title,
            chunk.experiment_number,
            chunk.parent_heading,
            chunk.logical_unit_type,
            chunk.is_split,
            chunk.source_document,
        )
        previous_metadata = unit_metadata.setdefault(
            chunk.logical_unit_id,
            metadata,
        )
        if previous_metadata != metadata:
            raise RuntimeError(
                "Chunks belonging to one logical unit have inconsistent metadata."
            )
        unit_chunk_counts[chunk.logical_unit_id] = (
            unit_chunk_counts.get(chunk.logical_unit_id, 0) + 1
        )

    for logical_unit_id, chunk_count in unit_chunk_counts.items():
        if unit_metadata[logical_unit_id][4] != (chunk_count > 1):
            raise RuntimeError("Logical-unit split status does not match its chunks.")


def logical_unit_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    *,
    page_break_offsets: tuple[int, ...] | list[int] = (),
    source_document: str | None = None,
) -> list[LogicalChunk]:
    """Group each experiment appearance as a logical unit, then size-limit it.

    Outer experiment headings establish parent context. A titled experiment
    heading within that context starts a child logical unit. Every appearance
    receives a sequence-based ID, so repeated experiment numbers stay distinct.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if (
        not isinstance(max_chunk_size, int)
        or isinstance(max_chunk_size, bool)
        or max_chunk_size <= 0
    ):
        raise ValueError("max_chunk_size must be a positive integer.")
    if source_document is not None and (
        not isinstance(source_document, str) or not source_document.strip()
    ):
        raise ValueError("source_document must be None or a non-empty string.")
    if isinstance(page_break_offsets, (str, bytes)):
        raise TypeError("page_break_offsets must be a sequence of integer offsets.")

    offsets = tuple(page_break_offsets)
    if any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets):
        raise TypeError("page break offsets must be integers.")
    if len(set(offsets)) != len(offsets):
        raise ValueError("page_break_offsets must be unique.")
    if tuple(sorted(offsets)) != offsets:
        raise ValueError("page_break_offsets must be in ascending order.")
    for offset in offsets:
        if offset < 0 or offset >= len(text) or text[offset] != "\n":
            raise ValueError(
                "each page break offset must point to a newline."
            )

    if not text:
        return []

    units = _find_units(text)
    chunks: list[LogicalChunk] = []
    for unit_number, unit in enumerate(units, start=1):
        unit_text = text[unit.start : unit.end]
        split_ranges = _split_unit(unit_text, max_chunk_size)
        is_split = len(split_ranges) > 1
        _, section_headings = _section_boundaries(unit_text)
        section_positions = [position for position, _ in section_headings]

        for relative_start, relative_end, boundary_reason in split_ranges:
            absolute_start = unit.start + relative_start
            absolute_end = unit.start + relative_end
            section_index = bisect_right(section_positions, relative_start) - 1
            section_heading = (
                section_headings[section_index][1]
                if section_index >= 0
                else None
            )
            chunk_text = text[absolute_start:absolute_end]
            chunks.append(
                LogicalChunk(
                    chunk_index=len(chunks) + 1,
                    character_count=len(chunk_text),
                    text=chunk_text,
                    logical_unit_id=f"logical-unit-{unit_number:04d}",
                    logical_unit_title=unit.metadata.title,
                    experiment_number=unit.metadata.experiment_number,
                    parent_heading=unit.metadata.parent_heading,
                    logical_unit_type=unit.metadata.unit_type,
                    section_heading=section_heading,
                    is_split=is_split,
                    boundary_reason=boundary_reason,
                    source_document=source_document,
                    source_page_numbers=_page_numbers_for_span(
                        absolute_start,
                        absolute_end,
                        offsets,
                    ),
                )
            )

    _validate_chunks(chunks, text, max_chunk_size, source_document)
    return chunks
