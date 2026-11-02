"""Integration tests for the semantic preview script using synthetic input."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import preview_semantic


class FakeEmbeddingModel:
    max_seq_length = 512

    class Tokenizer:
        def encode(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
        ) -> list[int]:
            if truncation:
                raise AssertionError("token counting must not enable truncation")
            count = len(re.findall(r"\w+|[^\w\s]", text))
            return list(range(count + (2 if add_special_tokens else 0)))

    tokenizer = Tokenizer()

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.encode_kwargs = kwargs
        return np.asarray(
            [
                [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]
                for index, _ in enumerate(sentences)
            ],
            dtype=float,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 2


class SemanticPreviewTests(unittest.TestCase):
    def test_synthetic_pipeline_writes_chunks_and_summary_without_real_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "semantic_preview"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text="Aim: First sentence. Second sentence.")],
                empty_pages=(),
            )

            with (
                patch.object(preview_semantic, "SOURCE_PDF", source_pdf),
                patch.object(
                    preview_semantic, "OUTPUT_DIRECTORY", output_directory
                ),
                patch.object(
                    preview_semantic,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_semantic, "clean_text", side_effect=lambda text: text),
                patch.object(
                    preview_semantic,
                    "load_sentence_model",
                    return_value=FakeEmbeddingModel(),
                ) as load_model,
                patch.object(
                    preview_semantic,
                    "_runtime_versions",
                    return_value={"python": "test"},
                ),
                patch.object(preview_semantic, "MAX_CHUNK_SIZE", 100),
                patch.object(preview_semantic, "SIMILARITY_THRESHOLD", 0.5),
                patch.object(preview_semantic, "MODEL_REVISION", "test-revision"),
            ):
                preview_semantic.main()

            load_model.assert_called_once_with(
                preview_semantic.MODEL_NAME,
                preview_semantic.DEVICE,
                revision="test-revision",
            )
            chunks = json.loads(
                (output_directory / "chunk_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            expected_text = "Aim: First sentence. Second sentence."

            self.assertEqual("".join(chunk["text"] for chunk in chunks), expected_text)
            self.assertEqual(summary["number_of_pages_extracted"], 1)
            self.assertEqual(summary["number_of_original_segments"], 2)
            self.assertEqual(summary["number_of_embedding_inputs"], 2)
            self.assertEqual(summary["number_of_chunks"], len(chunks))
            self.assertEqual(
                summary["maximum_chunk_length_characters"],
                max(len(chunk["text"]) for chunk in chunks),
            )
            self.assertTrue(summary["chunks_may_cross_page_boundaries"])
            self.assertFalse(summary["page_level_chunk_provenance_available"])
            self.assertEqual(summary["chunking"]["runtime_versions"], {"python": "test"})
            self.assertEqual(
                summary["chunking"]["requested_model_revision"],
                "test-revision",
            )
            self.assertIsNone(summary["chunking"]["resolved_model_revision"])

    def test_whitespace_only_input_does_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "semantic_preview"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text=" \n\t ")],
                empty_pages=(),
            )

            with (
                patch.object(preview_semantic, "SOURCE_PDF", source_pdf),
                patch.object(
                    preview_semantic, "OUTPUT_DIRECTORY", output_directory
                ),
                patch.object(
                    preview_semantic,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_semantic, "clean_text", side_effect=lambda text: text),
                patch.object(
                    preview_semantic,
                    "load_sentence_model",
                ) as load_model,
                patch.object(
                    preview_semantic,
                    "_runtime_versions",
                    return_value={"python": "test"},
                ),
            ):
                preview_semantic.main()

            load_model.assert_not_called()
            metadata = json.loads(
                (output_directory / "chunk_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("".join(chunk["text"] for chunk in metadata), " \n\t ")

    def test_partial_output_files_are_removed_when_writing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "semantic_preview"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text="One sentence.")],
                empty_pages=(),
            )
            original_open = Path.open

            class FailingWriter:
                def __init__(
                    self,
                    path: Path,
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    self.file = original_open(path, *args, **kwargs)

                def __enter__(self) -> "FailingWriter":
                    self.file.__enter__()
                    return self

                def __exit__(
                    self,
                    exc_type: object,
                    exc_value: object,
                    traceback: object,
                ) -> object:
                    return self.file.__exit__(exc_type, exc_value, traceback)

                def write(self, content: str) -> None:
                    self.file.write(content[:12])
                    raise OSError("simulated write failure")

            def fail_metadata_write(
                path: Path,
                *args: object,
                **kwargs: object,
            ) -> object:
                if path.name == "chunk_metadata.json":
                    return FailingWriter(path, *args, **kwargs)
                return original_open(path, *args, **kwargs)

            with (
                patch.object(preview_semantic, "SOURCE_PDF", source_pdf),
                patch.object(
                    preview_semantic, "OUTPUT_DIRECTORY", output_directory
                ),
                patch.object(
                    preview_semantic,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_semantic, "clean_text", side_effect=lambda text: text),
                patch.object(
                    preview_semantic,
                    "load_sentence_model",
                    return_value=FakeEmbeddingModel(),
                ),
                patch.object(
                    preview_semantic,
                    "_runtime_versions",
                    return_value={"python": "test"},
                ),
                patch.object(Path, "open", new=fail_metadata_write),
            ):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    preview_semantic.main()

            self.assertFalse((output_directory / "chunks.txt").exists())
            self.assertFalse((output_directory / "summary.json").exists())
            self.assertFalse((output_directory / "chunk_metadata.json").exists())

    def test_resolved_model_revision_comes_from_transformers_config(self) -> None:
        model = FakeEmbeddingModel()
        model._first_module = lambda: SimpleNamespace(  # type: ignore[method-assign]
            auto_model=SimpleNamespace(
                config=SimpleNamespace(_commit_hash="0123456789abcdef")
            )
        )

        self.assertEqual(
            preview_semantic._resolved_model_revision(model),
            "0123456789abcdef",
        )

    def test_existing_outputs_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "semantic_preview"
            output_directory.mkdir()
            existing_file = output_directory / "summary.json"
            existing_file.write_text("keep this", encoding="utf-8")

            with (
                patch.object(preview_semantic, "SOURCE_PDF", source_pdf),
                patch.object(
                    preview_semantic, "OUTPUT_DIRECTORY", output_directory
                ),
                patch.object(preview_semantic, "extract_pdf_pages") as extract_pages,
            ):
                with self.assertRaises(FileExistsError):
                    preview_semantic.main()

            self.assertEqual(existing_file.read_text(encoding="utf-8"), "keep this")
            extract_pages.assert_not_called()

    def test_missing_source_pdf_is_reported_before_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_pdf = Path(temporary_directory) / "missing.pdf"

            with (
                patch.object(preview_semantic, "SOURCE_PDF", missing_pdf),
                patch.object(preview_semantic, "extract_pdf_pages") as extract_pages,
            ):
                with self.assertRaises(FileNotFoundError):
                    preview_semantic.main()

            extract_pages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
