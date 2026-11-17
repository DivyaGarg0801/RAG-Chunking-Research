"""Tests for the shared, page-preserving cleaned-text artifact."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.extraction_artifact import (
    create_extraction_artifact,
    load_extraction_artifact,
)
from src.ingestion.pdf_loader import PdfExtractionResult, PdfPage


class ExtractionArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source_pdf = self.root / "manual.pdf"
        self.artifact = self.root / "extracted_pages.json"

    def test_creation_cleans_text_and_keeps_empty_page_records(self) -> None:
        extraction = PdfExtractionResult(
            pages=(
                PdfPage("manual.pdf", 1, "\n  First page.\r\n\r\n"),
                PdfPage("manual.pdf", 2, ""),
                PdfPage("manual.pdf", 3, "Third page.\n"),
            ),
            empty_pages=(2,),
        )
        with patch(
            "src.ingestion.extraction_artifact.extract_pdf_pages",
            return_value=extraction,
        ) as extract_pages:
            document = create_extraction_artifact(self.source_pdf, self.artifact)

        extract_pages.assert_called_once_with(self.source_pdf)
        payload = json.loads(self.artifact.read_text(encoding="utf-8"))
        self.assertEqual(
            payload,
            {
                "source_document": "manual.pdf",
                "pages": [
                    {"page_number": 1, "text": "  First page."},
                    {"page_number": 2, "text": ""},
                    {"page_number": 3, "text": "Third page."},
                ],
            },
        )
        self.assertEqual([page.page_number for page in document.pages], [1, 2, 3])
        self.assertEqual(document.empty_pages, (2,))
        self.assertEqual(document.text, "  First page.\n\nThird page.")
        self.assertEqual(document.page_break_offsets, (13, 14))

    def test_loading_preserves_page_offsets_and_page_references(self) -> None:
        self.artifact.write_text(
            json.dumps(
                {
                    "source_document": "manual.pdf",
                    "pages": [
                        {"page_number": 1, "text": "Alpha"},
                        {"page_number": 2, "text": ""},
                        {"page_number": 3, "text": "Beta"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        document = load_extraction_artifact(self.artifact)

        self.assertEqual(document.text, "Alpha\n\nBeta")
        self.assertEqual(document.empty_pages, (2,))
        self.assertEqual(
            document.page_numbers_for_chunks(["Alpha\n", "\nBeta"]),
            ((1,), (3,)),
        )

    def test_existing_artifact_is_not_overwritten_without_explicit_permission(self) -> None:
        self.artifact.write_text("keep this", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            create_extraction_artifact(self.source_pdf, self.artifact)
        self.assertEqual(self.artifact.read_text(encoding="utf-8"), "keep this")

    def test_explicit_overwrite_replaces_existing_artifact(self) -> None:
        self.artifact.write_text("old", encoding="utf-8")
        extraction = PdfExtractionResult(
            pages=(PdfPage("manual.pdf", 1, "New text.\n"),),
            empty_pages=(),
        )
        with patch(
            "src.ingestion.extraction_artifact.extract_pdf_pages",
            return_value=extraction,
        ):
            create_extraction_artifact(
                self.source_pdf,
                self.artifact,
                overwrite=True,
            )

        loaded = load_extraction_artifact(self.artifact)
        self.assertEqual(loaded.text, "New text.")

    def test_missing_artifact_raises_file_not_found(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_extraction_artifact(self.artifact)

    def test_invalid_json_raises_clear_value_error(self) -> None:
        self.artifact.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            load_extraction_artifact(self.artifact)

    def test_invalid_utf8_artifact_raises_clear_value_error(self) -> None:
        self.artifact.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            load_extraction_artifact(self.artifact)

    def test_invalid_page_numbers_or_text_are_rejected(self) -> None:
        for pages in (
            [{"page_number": 2, "text": "wrong number"}],
            [{"page_number": 1, "text": None}],
        ):
            with self.subTest(pages=pages):
                self.artifact.write_text(
                    json.dumps({"source_document": "manual.pdf", "pages": pages}),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    load_extraction_artifact(self.artifact)

    def test_pdf_extraction_errors_are_not_hidden(self) -> None:
        with patch(
            "src.ingestion.extraction_artifact.extract_pdf_pages",
            side_effect=ValueError("invalid PDF"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid PDF"):
                create_extraction_artifact(self.source_pdf, self.artifact)
        self.assertFalse(self.artifact.exists())


if __name__ == "__main__":
    unittest.main()
