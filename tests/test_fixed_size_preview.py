"""Integration tests for the fixed-size preview using synthetic artifact data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import preview_fixed_size
from src.ingestion.extraction_artifact import page_preserving_text
from src.ingestion.pdf_loader import PdfPage


class FixedSizePreviewTests(unittest.TestCase):
    def test_preview_reads_pages_and_records_source_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "fixed_preview"
            artifact = page_preserving_text(
                "manual.pdf",
                (
                    PdfPage("manual.pdf", 1, "abcdefgh"),
                    PdfPage("manual.pdf", 2, "ijklmnop"),
                ),
            )
            with (
                patch.object(
                    preview_fixed_size,
                    "EXTRACTION_ARTIFACT",
                    root / "input.json",
                ),
                patch.object(
                    preview_fixed_size,
                    "load_extraction_artifact",
                    return_value=artifact,
                ),
                patch.object(preview_fixed_size, "CHUNK_SIZE", 10),
                patch.object(preview_fixed_size, "OVERLAP", 2),
            ):
                preview_fixed_size.main(output_directory)

            metadata = json.loads(
                (output_directory / "chunk_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["source_page_numbers"] for item in metadata],
                [[1, 2], [2]],
            )
            self.assertEqual(
                "".join(
                    item["text"] if index == 0 else item["text"][2:]
                    for index, item in enumerate(metadata)
                ),
                artifact.text,
            )
            self.assertTrue(summary["page_level_chunk_provenance_available"])

    def test_existing_preview_stops_before_loading_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "fixed_preview"
            output_directory.mkdir()
            existing = output_directory / "chunks.txt"
            existing.write_text("keep", encoding="utf-8")
            with (
                patch.object(preview_fixed_size, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_fixed_size, "load_extraction_artifact"
                ) as load_document,
            ):
                with self.assertRaises(FileExistsError):
                    preview_fixed_size.main()

            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
            load_document.assert_not_called()


if __name__ == "__main__":
    unittest.main()
