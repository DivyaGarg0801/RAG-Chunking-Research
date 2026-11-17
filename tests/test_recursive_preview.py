"""Integration tests for the recursive preview using synthetic artifact data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import preview_recursive
from src.ingestion.extraction_artifact import page_preserving_text
from src.ingestion.pdf_loader import PdfPage


class RecursivePreviewTests(unittest.TestCase):
    def test_preview_preserves_source_text_and_page_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "recursive_preview"
            artifact = page_preserving_text(
                "manual.pdf",
                (
                    PdfPage("manual.pdf", 1, "First page has text.\n"),
                    PdfPage("manual.pdf", 2, "Second page has text."),
                ),
            )
            with (
                patch.object(
                    preview_recursive,
                    "EXTRACTION_ARTIFACT",
                    root / "extracted_pages.json",
                ),
                patch.object(
                    preview_recursive,
                    "load_extraction_artifact",
                    return_value=artifact,
                ),
                patch.object(preview_recursive, "CHUNK_SIZE", 18),
                patch.object(preview_recursive, "OVERLAP", 4),
            ):
                preview_recursive.main(output_directory)

            metadata = json.loads(
                (output_directory / "chunk_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "".join(
                    item["text"][item["overlap_length"] :]
                    for item in metadata
                ),
                artifact.text,
            )
            self.assertIn(1, metadata[0]["source_page_numbers"])
            self.assertIn(2, metadata[-1]["source_page_numbers"])
            self.assertTrue(summary["text_preserved_exactly"])
            self.assertTrue(summary["page_level_provenance_available"])


if __name__ == "__main__":
    unittest.main()
