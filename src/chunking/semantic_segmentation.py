"""Alternative paragraph-aware segmentation for semantic chunking studies."""

import re
from dataclasses import dataclass
from typing import Sequence


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
_SECTION_LABEL = re.compile(
    r"^\s*(?:aim|theory|procedure|algorithm|program|output|result|"
    r"observation|conclusion)\s*(?:\s*[:\-]\s*)+",
    re.IGNORECASE,
)
_EXPERIMENT_HEADING = re.compile(
    r"^\s*experi(?:ment|met)\s*[- ]?\s*\d+\b",
    re.IGNORECASE,
)
_FIGURE_HEADING = re.compile(r"^\s*(?:figure|fig\.?)\s*\.?\s*\d+", re.IGNORECASE)
_COLON_HEADING = re.compile(r"^\s*[^.!?]{1,80}:\s*$")
_LIST_MARKER = re.compile(
    r"^\s*(?:\d+[.)]|[a-z][.)]|[●•▪◦➢❖])\s+",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(r"\s*\d{1,3}\s*")
_CODE_START = re.compile(
    r"^\s*(?:"
    r"#include\s*[<\"]|"
    r"\bint\s+main\s*\(|"
    r"\breturn\s+0\s*;|"
    r"\b(?:if|for|while)\s*\(|"
    r"\b(?:def|class|import|from|print|raise|pass|break|continue)\b|"
    r"[\w.]+\s*(?:\+=|-=|\*=|/=|:=|=(?!=))|"
    r"(?:std::)?cout\s*<<"
    r")"
)
_STANDALONE_COMMAND = re.compile(
    r"^\s*(?:sudo\s+|(?:ping|ifconfig|traceroute|nslookup|host|dig|nmap|"
    r"tcpdump|gcc|g\+\+|iptables)\s+|\.\/[\w.-]+(?:\s|$))",
    re.IGNORECASE,
)
_PAGE_CONTINUATION_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class _Line:
    text: str
    content: str
    start: int
    page_start: bool
    kind: str


def _abbreviation_before(text: str, index: int) -> bool:
    preceding_text = text[: index + 1].lower()
    if preceding_text.endswith("ms.") and re.search(
        r"\d(?:\.\d+)?\s+ms\.$",
        preceding_text,
    ):
        return False
    return any(preceding_text.endswith(item) for item in _ABBREVIATIONS)


def _is_sentence_ending(text: str) -> bool:
    ending = text.rstrip()
    while ending and ending[-1] in _CLOSING_PUNCTUATION:
        ending = ending[:-1]
    if not ending or ending[-1] not in ".!?":
        return False
    if ending[-1] == "." and _abbreviation_before(ending, len(ending) - 1):
        return False
    if (
        ending[-1] == "."
        and len(ending) >= 2
        and ending[-2].isdigit()
    ):
        return False
    return True


def _is_code_like(content: str) -> bool:
    stripped = content.strip()
    indentation = len(content) - len(content.lstrip(" \t"))
    if _STANDALONE_COMMAND.match(stripped):
        return True
    if _CODE_START.match(stripped):
        return True
    return bool(
        indentation
        and re.match(
            r"(?:"
            r"#|"
            r"(?:from|import|def|class|return|print|raise|pass|break|continue)\b|"
            r"(?:if|elif|else|for|while|try|except|with)\b.*:|"
            r"[\w.]+\s*(?:\+=|-=|\*=|/=|:=|=(?!=))|"
            r"[{}\[\]();]"
            r")",
            stripped,
        )
    )


def _line_kind(content: str, page_start: bool) -> str:
    stripped = content.strip()
    if not stripped:
        return "blank"
    if page_start and _PAGE_NUMBER.fullmatch(content):
        return "page_number"
    if _EXPERIMENT_HEADING.match(content) or _FIGURE_HEADING.match(content):
        return "heading"
    section_match = _SECTION_LABEL.match(content)
    if section_match:
        return "section" if content[section_match.end() :].strip() else "section_label"
    if _LIST_MARKER.match(content):
        return "list"
    if _is_code_like(content):
        return "code"
    if _COLON_HEADING.match(content):
        return "heading"
    return "prose"


def _read_lines(text: str, page_break_offsets: set[int]) -> list[_Line]:
    page_starts = {0, *(offset + 1 for offset in page_break_offsets)}
    lines: list[_Line] = []
    start = 0
    while start < len(text):
        newline_index = text.find("\n", start)
        end = len(text) if newline_index == -1 else newline_index + 1
        content_end = end - 1 if newline_index != -1 else end
        content = text[start:content_end]
        page_start = start in page_starts
        lines.append(
            _Line(
                text=text[start:end],
                content=content,
                start=start,
                page_start=page_start,
                kind=_line_kind(content, page_start),
            )
        )
        start = end
    return lines


def _is_clear_page_continuation(previous: _Line, following: _Line) -> bool:
    previous_text = previous.content.rstrip()
    following_text = following.content.lstrip()
    if not previous_text or not following_text or _is_sentence_ending(previous_text):
        return False
    if following.kind != "prose":
        return False
    first_character = following_text[0]
    last_word_match = re.search(r"[A-Za-z]+$", previous_text)
    last_word = last_word_match.group(0).lower() if last_word_match else ""
    return first_character.islower() or last_word in _PAGE_CONTINUATION_WORDS


def _can_continue_page(lines: list[_Line], index: int) -> bool:
    return (
        index == 0
        or not lines[index].page_start
        or _is_clear_page_continuation(lines[index - 1], lines[index])
    )


def _collect_prose(lines: list[_Line], start: int) -> tuple[str, int]:
    pieces = [lines[start].text]
    index = start + 1
    while (
        index < len(lines)
        and lines[index].kind == "prose"
        and _can_continue_page(lines, index)
    ):
        pieces.append(lines[index].text)
        index += 1
    return "".join(pieces), index


def _collect_list_item(lines: list[_Line], start: int) -> tuple[str, int]:
    pieces = [lines[start].text]
    index = start + 1
    while (
        index < len(lines)
        and lines[index].kind in {"prose", "code"}
        and _can_continue_page(lines, index)
    ):
        pieces.append(lines[index].text)
        index += 1
    return "".join(pieces), index


def _collect_code(lines: list[_Line], start: int) -> tuple[str, int]:
    pieces = [lines[start].text]
    index = start + 1
    while index < len(lines) and lines[index].kind == "code":
        pieces.append(lines[index].text)
        index += 1
    return "".join(pieces), index


def _collect_section_label(lines: list[_Line], start: int) -> tuple[str, int, str]:
    if start + 1 >= len(lines):
        return lines[start].text, start + 1, "section_label"
    following = lines[start + 1]
    if following.kind in {"blank", "heading", "section", "section_label", "page_number"}:
        return lines[start].text, start + 1, "section_label"
    if not _can_continue_page(lines, start + 1):
        return lines[start].text, start + 1, "section_label"

    if following.kind == "list":
        body, end = _collect_list_item(lines, start + 1)
        return lines[start].text + body, end, "list"
    if following.kind == "code":
        body, end = _collect_code(lines, start + 1)
        return lines[start].text + body, end, "code"
    body, end = _collect_prose(lines, start + 1)
    return lines[start].text + body, end, "prose"


def _split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character not in ".!?":
            index += 1
            continue

        if character == ".":
            previous_is_digit = index > 0 and text[index - 1].isdigit()
            next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
            if (
                (previous_is_digit and next_is_digit)
                or _abbreviation_before(text, index)
            ):
                index += 1
                continue

        boundary_end = index + 1
        while (
            boundary_end < len(text)
            and text[boundary_end] in _CLOSING_PUNCTUATION
        ):
            boundary_end += 1
        if boundary_end < len(text) and text[boundary_end].isspace():
            while boundary_end < len(text) and text[boundary_end].isspace():
                boundary_end += 1
            pieces.append(text[start:boundary_end])
            start = boundary_end
            index = boundary_end
        else:
            index += 1

    if start < len(text):
        pieces.append(text[start:])
    return pieces


def segment_text_paragraph_aware(
    text: str,
    page_break_offsets: Sequence[int] = (),
) -> list[str]:
    """Segment wrapped prose while preserving structural lines and all text.

    Each page-break offset is the index of the newline inserted between two
    cleaned page texts. The returned pieces are contiguous source spans, so
    joining them reconstructs ``text`` exactly.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text:
        if page_break_offsets:
            raise ValueError("page_break_offsets cannot refer to empty text.")
        return []

    normalized_offsets: set[int] = set()
    for offset in page_break_offsets:
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError("page break offsets must be integers.")
        if offset < 0 or offset >= len(text) or text[offset] != "\n":
            raise ValueError("each page break offset must point to a newline.")
        if offset in normalized_offsets:
            raise ValueError("page break offsets must be unique.")
        normalized_offsets.add(offset)

    lines = _read_lines(text, normalized_offsets)
    segments: list[str] = []
    pending_whitespace = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.kind == "blank":
            if segments:
                segments[-1] += line.text
            else:
                pending_whitespace += line.text
            index += 1
            continue

        prefix = pending_whitespace
        pending_whitespace = ""
        if line.kind == "page_number" or line.kind == "heading":
            block, end, split_sentences = line.text, index + 1, False
        elif line.kind == "section_label":
            block, end, block_kind = _collect_section_label(lines, index)
            split_sentences = block_kind == "prose"
        elif line.kind == "section":
            block, end = _collect_prose(lines, index)
            split_sentences = True
        elif line.kind == "list":
            block, end = _collect_list_item(lines, index)
            split_sentences = False
        elif line.kind == "code":
            block, end = _collect_code(lines, index)
            split_sentences = False
        else:
            block, end = _collect_prose(lines, index)
            split_sentences = True

        pieces = _split_sentences(block) if split_sentences else [block]
        if pieces:
            segments.append(prefix + pieces[0])
            segments.extend(piece for piece in pieces[1:] if piece)
        index = end

    if pending_whitespace:
        if segments:
            segments[-1] += pending_whitespace
        else:
            segments.append(pending_whitespace)

    if "".join(segments) != text:
        raise RuntimeError("Paragraph-aware segmentation changed the source text.")
    return segments
