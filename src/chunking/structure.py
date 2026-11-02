"""Structure-aware chunking for laboratory-manual text."""

import re
from dataclasses import dataclass


_HEADING_LABELS = (
    "Experiment",
    "Expt",
    "Aim",
    "Objective",
    "Theory",
    "Algorithm",
    "Procedure",
    "Program",
    "Output",
    "Result",
    "Observation",
    "Analysis",
    "Discussion",
    "Conclusion",
    "Summary",
    "Method",
    "Implementation",
    "Hypothesis",
    "Task",
    "Requirement",
    "Requirements",
)
_HEADING_PATTERN = "|".join(re.escape(label) for label in _HEADING_LABELS)
_EXPERIMENT_HEADING_PATTERN = re.compile(
    r"^(?:Experiment|Experimet|Expt)(?:\s*[\.\-–]\s*|\s*)"
    r"(?P<number>\d+)(?:\s*[:\-–]\s*(?P<title>.*?))?$",
    re.IGNORECASE,
)
_TOPIC_HEADINGS = (
    "AODV (Ad-hoc On-Demand Distance Vector)",
    "TCP Socket Programming",
    "UDP Socket Programming",
    "DNS (Domain Name Server)",
    "TCP Congestion Control",
)


@dataclass(frozen=True)
class StructureChunk:
    """One chunk with its structural heading metadata."""

    chunk_index: int
    text: str
    heading: str | None
    heading_level: int
    parent_heading: str | None
    section_path: tuple[str, ...]


def _normalized_heading_text(label: str) -> str:
    """Normalize a heading label while preserving the source line text."""
    cleaned = label.strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1]
    return cleaned.strip()


def _detect_heading(line: str) -> dict[str, int | str | None] | None:
    """Return heading metadata for a conservative line-level heading match."""
    if not isinstance(line, str):
        raise TypeError("Each line must be a string.")

    stripped = line.strip()
    if not stripped:
        return None
    if len(stripped) > 120:
        return None
    if stripped.startswith(("#", "*", "-", "=", "_", "|")):
        return None
    if stripped.endswith(("?", "!")):
        return None

    experiment_match = _EXPERIMENT_HEADING_PATTERN.fullmatch(stripped)
    if experiment_match is not None:
        title = (experiment_match.group("title") or "").strip()
        if title:
            title = re.sub(r"[!?]+\s*$", "", title).strip()
            if (
                not title
                or title.endswith(".")
                or len(title.split()) > 12
            ):
                return None
            heading = f"Experiment {experiment_match.group('number')}: {title}"
        else:
            heading = f"Experiment {experiment_match.group('number')}"
        return {
            "heading": heading,
            "heading_level": 1,
        }

    for topic_heading in _TOPIC_HEADINGS:
        if stripped.casefold() == topic_heading.casefold():
            return {
                "heading": topic_heading,
                "heading_level": 2,
            }

    match = re.match(
        rf"^(?:\d+\s*[\.\)]\s*|\d+\s*[:\-–]\s*|[•\-*]\s*)?"
        rf"(?P<label>{_HEADING_PATTERN})"
        rf"(?:\s+(?P<id>\d+|[A-Za-z0-9._-]+))?"
        rf"(?:\s*[:\-–]\s*(?P<title>.*?))?"
        rf"\s*$",
        stripped,
        re.IGNORECASE,
    )
    if match is None:
        return None

    label = match.group("label")
    identifier = match.group("id")
    title = (match.group("title") or "").strip()
    if title:
        title = re.sub(r"[.!?]+\s*$", "", title).strip()

    if label.lower() in {"experiment", "expt"}:
        if not title and not identifier:
            normalized = label
        elif identifier and title:
            normalized = f"{label} {identifier}: {title}"
        elif identifier:
            normalized = f"{label} {identifier}"
        else:
            normalized = f"{label}: {title}"
    elif identifier:
        normalized = f"{label} {identifier}"
    else:
        normalized = label

    normalized = _normalized_heading_text(normalized)
    if not normalized:
        return None
    if len(normalized.split()) > 12:
        return None
    if normalized.lower().startswith("experiment") and len(normalized) > 80:
        return None
    if re.search(r"[.!?]", normalized):
        return None

    heading_level = 1 if label.lower() in {"experiment", "expt"} else 2
    return {
        "heading": normalized,
        "heading_level": heading_level,
    }


def _split_section_text(text: str, maximum_size: int) -> list[str]:
    """Split a section into smaller chunks while preserving exact text."""
    if len(text) <= maximum_size:
        return [text]

    paragraph_parts = re.split(r"(\n\s*\n+)", text)
    chunks: list[str] = []
    current = ""

    for part in paragraph_parts:
        if not part:
            continue
        if re.fullmatch(r"\n\s*\n+", part):
            if current and len(current) + len(part) > maximum_size:
                chunks.append(current)
                current = part
            else:
                current += part
            continue

        if current and len(current) + len(part) > maximum_size and current.strip():
            chunks.append(current)
            current = part
        else:
            current += part

    if current:
        chunks.append(current)

    if all(len(chunk) <= maximum_size for chunk in chunks):
        return chunks or [text]

    line_parts = text.splitlines(keepends=True)
    fallback_chunks: list[str] = []
    current = ""
    for line in line_parts:
        if current and len(current) + len(line) > maximum_size and current.strip():
            fallback_chunks.append(current)
            current = line
        else:
            current += line
    if current:
        fallback_chunks.append(current)

    if all(len(chunk) <= maximum_size for chunk in fallback_chunks):
        return fallback_chunks

    char_chunks = [
        text[start : start + maximum_size]
        for start in range(0, len(text), maximum_size)
    ]
    if any(len(chunk) > maximum_size for chunk in char_chunks):
        raise RuntimeError(
            "Unable to enforce the configured maximum chunk size without dropping text."
        )
    return char_chunks


def structure_chunk(
    text: str,
    max_chunk_size: int | None = None,
) -> list[StructureChunk]:
    """Chunk text using conservative lab-manual structure detection."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if max_chunk_size is not None:
        if (
            not isinstance(max_chunk_size, int)
            or isinstance(max_chunk_size, bool)
            or max_chunk_size <= 0
        ):
            raise ValueError("max_chunk_size must be a positive integer or None.")
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    sections: list[StructureChunk] = []
    current_heading: str | None = None
    current_parent: str | None = None
    current_path: tuple[str, ...] = ()
    current_level = 0
    current_lines: list[str] = []
    recent_level_one_heading: str | None = None

    def finalize_section(
        heading: str | None,
        parent: str | None,
        path: tuple[str, ...],
        level: int,
        lines_for_section: list[str],
    ) -> None:
        if not lines_for_section:
            return
        # Keep heading-only sections as valid chunks so consecutive headings do not
        # silently lose observable section text while preserving exact reconstruction.
        section_text = "".join(lines_for_section)
        if not section_text.strip():
            return

        chunks: list[tuple[str, str | None, int, str | None, tuple[str, ...]]] = []
        if max_chunk_size is None:
            chunks.append((section_text, heading, level, parent, path))
        else:
            section_parts = _split_section_text(section_text, max_chunk_size)
            if any(len(part) > max_chunk_size for part in section_parts):
                raise RuntimeError(
                    "The configured maximum chunk size could not be enforced without dropping characters."
                )
            for part in section_parts:
                chunks.append((part, heading, level, parent, path))

        for chunk_text, chunk_heading, chunk_level, chunk_parent, chunk_path in chunks:
            sections.append(
                StructureChunk(
                    chunk_index=len(sections) + 1,
                    text=chunk_text,
                    heading=chunk_heading,
                    heading_level=chunk_level,
                    parent_heading=chunk_parent,
                    section_path=chunk_path,
                )
            )

    for line in lines:
        heading_info = _detect_heading(line)
        if heading_info is not None:
            if current_lines:
                finalize_section(
                    current_heading,
                    current_parent,
                    current_path,
                    current_level,
                    current_lines,
                )

            heading_label = str(heading_info["heading"])
            heading_level = int(heading_info["heading_level"])
            if heading_level == 1:
                recent_level_one_heading = heading_label
                current_parent = None
                current_path = (heading_label,)
            else:
                current_parent = recent_level_one_heading
                if recent_level_one_heading is not None:
                    current_path = (recent_level_one_heading, heading_label)
                else:
                    current_path = (heading_label,)

            current_heading = heading_label
            current_level = heading_level
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        finalize_section(
            current_heading,
            current_parent,
            current_path,
            current_level,
            current_lines,
        )

    if not sections:
        return []

    reconstructed = "".join(chunk.text for chunk in sections)
    if reconstructed != text:
        raise RuntimeError("Structure-based chunking did not preserve the source text exactly.")

    for index, chunk in enumerate(sections, start=1):
        if chunk.chunk_index != index:
            raise RuntimeError("Structure-based chunk indices are inconsistent.")

    return sections
