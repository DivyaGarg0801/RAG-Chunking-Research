"""Extract text from PDF documents one page at a time."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PdfPage:
    """Text and source information for one page in a PDF."""

    source_filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class PdfExtractionResult:
    """Page-level text and page numbers with no extractable text."""

    pages: tuple[PdfPage, ...]
    empty_pages: tuple[int, ...]


def extract_pdf_pages(pdf_path: str | Path) -> PdfExtractionResult:
    """Extract text from each page, reporting pages with no text.

    Args:
        pdf_path: Path to the PDF file to read.

    Raises:
        FileNotFoundError: If the given path is not a file.
        ValueError: If the file cannot be opened as a readable PDF.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    pages: list[PdfPage] = []
    empty_pages: list[int] = []

    try:
        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text()
                pages.append(
                    PdfPage(
                        source_filename=path.name,
                        page_number=page_number,
                        text=text,
                    )
                )
                if not text.strip():
                    empty_pages.append(page_number)
    except (pymupdf.FileDataError, OSError) as error:
        raise ValueError(
            f"Could not read a valid PDF from '{path}': {error}"
        ) from error

    return PdfExtractionResult(
        pages=tuple(pages),
        empty_pages=tuple(empty_pages),
    )
