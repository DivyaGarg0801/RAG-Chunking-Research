"""Tests for page-by-page PDF text extraction."""

import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.ingestion.pdf_loader import extract_pdf_pages


class ExtractPdfPagesTests(unittest.TestCase):
    """Check page text, metadata, and PDF errors."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.pdf_path = Path(self.temporary_directory.name) / "sample_manual.pdf"

    def create_pdf(self, page_texts: list[str | None]) -> Path:
        """Create a small PDF for a test, leaving None pages text-free."""
        with pymupdf.open() as document:
            for page_text in page_texts:
                page = document.new_page()
                if page_text is not None:
                    page.insert_text((72, 72), page_text)
            document.save(self.pdf_path)

        return self.pdf_path

    def test_extracts_text_and_metadata_from_multiple_pages(self) -> None:
        self.create_pdf(["First page text", "Second page text"])

        result = extract_pdf_pages(self.pdf_path)

        self.assertEqual(len(result.pages), 2)
        self.assertEqual(
            [page.source_filename for page in result.pages],
            ["sample_manual.pdf", "sample_manual.pdf"],
        )
        self.assertEqual([page.page_number for page in result.pages], [1, 2])
        self.assertEqual(result.pages[0].text, "First page text\n")
        self.assertEqual(result.pages[1].text, "Second page text\n")
        self.assertEqual(result.empty_pages, ())

    def test_reports_a_text_free_page_without_dropping_its_record(self) -> None:
        self.create_pdf(["Page with text", None])

        result = extract_pdf_pages(self.pdf_path)

        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.pages[1].page_number, 2)
        self.assertEqual(result.pages[1].text, "")
        self.assertEqual(result.empty_pages, (2,))

    def test_missing_pdf_raises_file_not_found_error(self) -> None:
        missing_path = Path(self.temporary_directory.name) / "missing.pdf"

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            extract_pdf_pages(missing_path)

    def test_invalid_pdf_raises_clear_value_error(self) -> None:
        self.pdf_path.write_bytes(b"This is not a PDF.")

        with self.assertRaisesRegex(ValueError, "Could not read a valid PDF"):
            extract_pdf_pages(self.pdf_path)


if __name__ == "__main__":
    unittest.main()
