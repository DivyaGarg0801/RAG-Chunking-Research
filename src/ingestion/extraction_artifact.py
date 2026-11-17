"""Create and load reusable, page-preserving cleaned-text artifacts."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.ingestion.pdf_loader import PdfPage, extract_pdf_pages
from src.preprocessing.text_cleaner import clean_text


PAGE_SEPARATOR = "\n"
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "extracted"
    / "extracted_pages.json"
)


@dataclass(frozen=True)
class PageSpan:
    """Character offsets occupied by one source page in the joined text."""

    page_number: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class PagePreservingText:
    """Joined chunker input alongside the page records and their offsets."""

    source_document: str
    pages: tuple[PdfPage, ...]
    text: str
    page_spans: tuple[PageSpan, ...]
    page_break_offsets: tuple[int, ...]
    empty_pages: tuple[int, ...]

    def page_numbers_for_range(self, start: int, end: int) -> tuple[int, ...]:
        """Return source pages whose text overlaps a character range."""
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
            or end > len(self.text)
        ):
            raise ValueError("The source-text range is invalid.")

        return tuple(
            span.page_number
            for span in self.page_spans
            if span.start_offset < end and span.end_offset > start
        )

    def page_numbers_for_chunks(
        self,
        chunks: Sequence[str],
        overlap_lengths: Sequence[int] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Map ordered, source-preserving chunks to their source pages.

        ``overlap_lengths`` contains the repeated prefix length for each chunk.
        It is useful for chunkers that intentionally repeat text between chunks.
        """
        if overlap_lengths is None:
            overlaps = (0,) * len(chunks)
        else:
            overlaps = tuple(overlap_lengths)
            if len(overlaps) != len(chunks):
                raise ValueError("Each chunk must have one overlap length.")

        source_end = 0
        page_numbers: list[tuple[int, ...]] = []
        for index, (chunk, overlap) in enumerate(zip(chunks, overlaps)):
            if not isinstance(chunk, str):
                raise TypeError("Each chunk must be a string.")
            if (
                not isinstance(overlap, int)
                or isinstance(overlap, bool)
                or overlap < 0
                or (index == 0 and overlap != 0)
                or overlap > source_end
            ):
                raise ValueError("A chunk has an invalid overlap length.")

            start = source_end - overlap
            end = start + len(chunk)
            if end > len(self.text) or self.text[start:end] != chunk:
                raise ValueError(
                    "Chunks do not match the ordered source text and overlaps."
                )
            page_numbers.append(self.page_numbers_for_range(start, end))
            source_end = end

        if source_end != len(self.text):
            raise ValueError("Chunks do not cover the complete source text.")
        return tuple(page_numbers)


def page_preserving_text(
    source_document: str,
    pages: Sequence[PdfPage],
) -> PagePreservingText:
    """Build the existing newline-joined input without discarding page offsets."""
    page_records = tuple(pages)
    if not isinstance(source_document, str) or not source_document.strip():
        raise ValueError("source_document must be a non-empty string.")

    page_spans: list[PageSpan] = []
    page_break_offsets: list[int] = []
    position = 0
    for index, page in enumerate(page_records):
        if page.page_number != index + 1:
            raise ValueError("Page numbers must be consecutive and start at one.")
        start = position
        position += len(page.text)
        page_spans.append(
            PageSpan(
                page_number=page.page_number,
                start_offset=start,
                end_offset=position,
            )
        )
        if index < len(page_records) - 1:
            page_break_offsets.append(position)
            position += len(PAGE_SEPARATOR)

    return PagePreservingText(
        source_document=source_document,
        pages=page_records,
        text=PAGE_SEPARATOR.join(page.text for page in page_records),
        page_spans=tuple(page_spans),
        page_break_offsets=tuple(page_break_offsets),
        empty_pages=tuple(
            page.page_number for page in page_records if not page.text.strip()
        ),
    )


def create_extraction_artifact(
    pdf_path: str | Path,
    artifact_path: str | Path,
    *,
    overwrite: bool = False,
) -> PagePreservingText:
    """Extract and clean each PDF page, then save it as a JSON artifact.

    Existing artifacts are left untouched unless ``overwrite`` is explicitly
    enabled. PDF extraction errors from :func:`extract_pdf_pages` are surfaced.
    """
    source_path = Path(pdf_path)
    destination = Path(artifact_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Extraction artifact already exists and was not changed: {destination}"
        )

    extraction = extract_pdf_pages(source_path)
    cleaned_pages = tuple(
        PdfPage(
            source_filename=source_path.name,
            page_number=page.page_number,
            text=clean_text(page.text),
        )
        for page in extraction.pages
    )
    document = page_preserving_text(source_path.name, cleaned_pages)
    payload = {
        "source_document": document.source_document,
        "pages": [
            {"page_number": page.page_number, "text": page.text}
            for page in document.pages
        ],
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                output_file.write(serialized)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    else:
        with destination.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(serialized)

    return document


def load_extraction_artifact(artifact_path: str | Path) -> PagePreservingText:
    """Load and validate a page-preserving extraction JSON artifact."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Extraction artifact does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Extraction artifact contains invalid JSON: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("Extraction artifact root must be a JSON object.")
    source_document = payload.get("source_document")
    raw_pages = payload.get("pages")
    if not isinstance(source_document, str) or not source_document.strip():
        raise ValueError(
            "Extraction artifact source_document must be a non-empty string."
        )
    if not isinstance(raw_pages, list):
        raise ValueError("Extraction artifact pages must be a JSON array.")

    pages: list[PdfPage] = []
    for expected_number, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            raise ValueError(f"Page {expected_number} must be a JSON object.")
        page_number = raw_page.get("page_number")
        text = raw_page.get("text")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number != expected_number
        ):
            raise ValueError(
                "Artifact page numbers must be consecutive, one-based integers."
            )
        if not isinstance(text, str):
            raise ValueError(f"Page {expected_number} text must be a string.")
        pages.append(
            PdfPage(
                source_filename=source_document,
                page_number=page_number,
                text=text,
            )
        )

    return page_preserving_text(source_document, pages)


