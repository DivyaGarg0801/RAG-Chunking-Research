"""Tests for structure-aware chunking."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import preview_structure
from src.chunking.structure import StructureChunk, structure_chunk


class StructureChunkTests(unittest.TestCase):
    """Check heading detection, metadata, and exact reconstruction."""

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(structure_chunk(""), [])

    def test_detects_experiment_and_subsection_headings(self) -> None:
        text = (
            "Experiment 1: Packet loss\n"
            "Aim: Measure the drop rate.\n"
            "Theory: Packets may be dropped by the router.\n"
            "Result: Loss falls below 5%.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].heading, "Experiment 1: Packet loss")
        self.assertEqual(chunks[0].heading_level, 1)
        self.assertIsNone(chunks[0].parent_heading)
        self.assertEqual(chunks[1].heading, "Aim")
        self.assertEqual(chunks[1].parent_heading, "Experiment 1: Packet loss")
        self.assertEqual(chunks[2].heading, "Theory")
        self.assertEqual(chunks[3].heading, "Result")
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_detects_experiment_heading_ocr_and_punctuation_variants(self) -> None:
        headings = (
            "Experiment: 1",
            "Experiment 1",
            "Experiment.2:",
            "Experiment1:",
            "Experimet-10",
        )
        for heading in headings:
            with self.subTest(heading=heading):
                chunks = structure_chunk(f"{heading}\n")
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].heading_level, 1)
                self.assertIsNone(chunks[0].parent_heading)
                self.assertEqual(chunks[0].section_path, (chunks[0].heading,))
                self.assertEqual(chunks[0].text, f"{heading}\n")

    def test_nested_experiment_titles_and_their_subsections_use_context(self) -> None:
        text = (
            "Experiment: 1\n"
            "Experiment 1: Packet loss\n"
            "Aim: Measure packet loss.\n"
            "Procedure: Run the test.\n"
            "Experiment.2: Throughput\n"
            "Result: Throughput increased.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.heading_level for chunk in chunks],
            [1, 2, 3, 3, 2, 3],
        )
        self.assertEqual(
            [chunk.parent_heading for chunk in chunks],
            [
                None,
                "Experiment 1",
                "Experiment 1: Packet loss",
                "Experiment 1: Packet loss",
                "Experiment 1",
                "Experiment 2: Throughput",
            ],
        )
        self.assertEqual(
            [chunk.section_path for chunk in chunks],
            [
                ("Experiment 1",),
                ("Experiment 1", "Experiment 1: Packet loss"),
                ("Experiment 1", "Experiment 1: Packet loss", "Aim"),
                ("Experiment 1", "Experiment 1: Packet loss", "Procedure"),
                ("Experiment 1", "Experiment 2: Throughput"),
                ("Experiment 1", "Experiment 2: Throughput", "Result"),
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_experiment_heading_starts_new_hierarchy(self) -> None:
        text = (
            "Experiment 1: Queueing\n"
            "Aim: Measure delay\n"
            "Procedure: Run the test\n"
            "Experiment: 2\n"
            "Aim: Measure packet loss\n"
            "Result: RED reduces queue drops\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.heading for chunk in chunks],
            [
                "Experiment 1: Queueing",
                "Aim",
                "Procedure",
                "Experiment 2",
                "Aim",
                "Result",
            ],
        )
        self.assertEqual(
            [chunk.parent_heading for chunk in chunks],
            [
                None,
                "Experiment 1: Queueing",
                "Experiment 1: Queueing",
                None,
                "Experiment 2",
                "Experiment 2",
            ],
        )
        self.assertEqual(
            [chunk.section_path for chunk in chunks],
            [
                ("Experiment 1: Queueing",),
                ("Experiment 1: Queueing", "Aim"),
                ("Experiment 1: Queueing", "Procedure"),
                ("Experiment 2",),
                ("Experiment 2", "Aim"),
                ("Experiment 2", "Result"),
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_detects_configured_standalone_topic_headings_conservatively(self) -> None:
        headings = (
            "AODV (Ad-hoc On-Demand Distance Vector)",
            "TCP Socket Programming",
            "DNS (Domain Name Server)",
        )
        text = (
            f"{headings[0]}\n"
            "AODV selects routes on demand.\n"
            f"{headings[1]}\n"
            "TCP sockets provide a stream interface.\n"
            f"{headings[2]}\n"
            "DNS maps names to addresses.\n"
            "The DNS server is configured in this paragraph.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.heading for chunk in chunks],
            list(headings),
        )
        self.assertTrue(all(chunk.heading_level == 2 for chunk in chunks))
        self.assertIn("AODV selects routes on demand.", chunks[0].text)
        self.assertIn("The DNS server is configured in this paragraph.", chunks[2].text)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_conservative_heading_detection_avoids_paragraph_text(self) -> None:
        text = (
            "The aim of this experiment is to study the signal.\n"
            "The theory explains timing jitter in the circuit.\n"
            "The result was stable and reproducible.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].heading)
        self.assertEqual(chunks[0].text, text)

    def test_numbered_and_colon_variants_are_recognized(self) -> None:
        text = (
            "1. Aim\n"
            "This paragraph is part of the aim.\n"
            "2. Procedure\n"
            "Collect data and record it.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading, "Aim")
        self.assertEqual(chunks[1].heading, "Procedure")
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_handles_unheaded_document_by_returning_one_chunk(self) -> None:
        text = "Background note\n\nThis is a normal paragraph.\n"

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].heading)
        self.assertEqual(chunks[0].text, text)

    def test_max_chunk_size_splits_only_large_sections(self) -> None:
        text = (
            "Experiment 2: Delay measurement\n"
            "Aim: " + ("Measure delay across the network. " * 4) + "\n"
            "Theory: " + ("Queue length affects delay. " * 4) + "\n"
            "Algorithm: Step 1. Step 2. Step 3.\n"
        )
        maximum_size = 60
        self.assertGreater(
            max(
                len(section)
                for section in (
                    "Experiment 2: Delay measurement\n",
                    "Aim: " + ("Measure delay across the network. " * 4) + "\n",
                    "Theory: " + ("Queue length affects delay. " * 4) + "\n",
                    "Algorithm: Step 1. Step 2. Step 3.\n",
                )
            ),
            maximum_size,
        )

        chunks = structure_chunk(text, max_chunk_size=maximum_size)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= maximum_size for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(1, len(chunks) + 1)),
        )

    def test_double_newlines_and_indentation_are_preserved(self) -> None:
        text = (
            "Experiment 3: Control loop\n\n"
            "    Aim:\n"
            "    Measure the response time.\n\n"
            "    Result:\n"
            "    Stable output was observed.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertIn("    Aim:\n", chunks[1].text)
        self.assertIn("    Result:\n", chunks[2].text)

    def test_invalid_max_chunk_size_is_rejected(self) -> None:
        for invalid_size in (0, -1, True, 10.5, "100"):
            with self.subTest(max_chunk_size=invalid_size):
                with self.assertRaisesRegex(ValueError, "max_chunk_size"):
                    structure_chunk(
                        "Experiment 1: Demo\nText\n",
                        max_chunk_size=invalid_size,
                    )

        with self.assertRaisesRegex(TypeError, "text"):
            structure_chunk(None)  # type: ignore[arg-type]

    def test_multiple_level_two_headings_share_the_same_experiment_parent(self) -> None:
        text = (
            "Experiment 1: Queueing\n"
            "Aim: Measure delay.\n"
            "Procedure: Run the test.\n"
            "Result: Delay was low.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(chunks[0].heading, "Experiment 1: Queueing")
        self.assertEqual(chunks[1].parent_heading, "Experiment 1: Queueing")
        self.assertEqual(chunks[2].parent_heading, "Experiment 1: Queueing")
        self.assertEqual(chunks[3].parent_heading, "Experiment 1: Queueing")

    def test_repeated_aim_procedure_and_result_keep_the_nearest_experiment_parent(
        self,
    ) -> None:
        text = (
            "Experiment: 1\n"
            "Aim: First aim.\n"
            "Procedure: First procedure.\n"
            "Result: First result.\n"
            "Aim: Second aim.\n"
            "Procedure: Second procedure.\n"
            "Result: Second result.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.parent_heading for chunk in chunks[1:]],
            ["Experiment 1"] * 6,
        )
        self.assertEqual(
            [chunk.section_path for chunk in chunks[1:]],
            [("Experiment 1", chunk.heading) for chunk in chunks[1:]],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_section_path_and_heading_level_follow_hierarchy(self) -> None:
        text = (
            "Experiment 1: Queueing\n"
            "Aim: Measure delay.\n"
            "Experiment: 2\n"
            "Result: Routes converge.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.heading_level for chunk in chunks],
            [1, 2, 1, 2],
        )
        self.assertEqual(
            [chunk.section_path for chunk in chunks],
            [
                ("Experiment 1: Queueing",),
                ("Experiment 1: Queueing", "Aim"),
                ("Experiment 2",),
                ("Experiment 2", "Result"),
            ],
        )

    def test_repeated_heading_names_do_not_corrupt_parent_metadata(self) -> None:
        text = (
            "Experiment 1: Queueing\n"
            "Aim: Measure delay.\n"
            "Aim: Measure throughput.\n"
            "Experiment: 2\n"
            "Aim: Measure packet loss.\n"
            "Aim: Measure route convergence.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.heading for chunk in chunks],
            [
                "Experiment 1: Queueing",
                "Aim",
                "Aim",
                "Experiment 2",
                "Aim",
                "Aim",
            ],
        )
        self.assertEqual(
            [chunk.parent_heading for chunk in chunks],
            [
                None,
                "Experiment 1: Queueing",
                "Experiment 1: Queueing",
                None,
                "Experiment 2",
                "Experiment 2",
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_new_experiment_resets_hierarchy(self) -> None:
        text = (
            "Experiment 1: Queueing\n"
            "Aim: Measure delay.\n"
            "Experiment: 2\n"
            "Aim: Measure throughput.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(chunks[0].heading, "Experiment 1: Queueing")
        self.assertEqual(chunks[1].parent_heading, "Experiment 1: Queueing")
        self.assertEqual(chunks[2].heading, "Experiment 2")
        self.assertEqual(chunks[3].parent_heading, "Experiment 2")

    def test_consecutive_headings_preserve_all_heading_text(self) -> None:
        text = "Aim: Measure delay.\nResult: Stable.\n"

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk.heading for chunk in chunks], ["Aim", "Result"])
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_max_chunk_size_is_enforced_for_every_returned_chunk(self) -> None:
        text = (
            "Experiment 1: Queueing\n"
            "Aim: " + ("word " * 80) + "\n"
            "Result: Stable.\n"
        )

        chunks = structure_chunk(text, max_chunk_size=40)

        self.assertTrue(all(len(chunk.text) <= 40 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_long_single_line_content_is_split_without_bypassing_max_size(self) -> None:
        text = "Procedure: " + ("x" * 200) + "\n"

        chunks = structure_chunk(text, max_chunk_size=50)

        self.assertTrue(all(len(chunk.text) <= 50 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_long_whitespace_only_line_is_split_without_losing_whitespace(self) -> None:
        text = "Start\n" + (" " * 200) + "\nEnd\n"

        chunks = structure_chunk(text, max_chunk_size=32)

        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 32 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_long_whitespace_only_run_is_split_without_losing_whitespace(self) -> None:
        text = (
            "Start"
            + (" \t\n" * 40)
            + "Paragraph content follows the whitespace run.\n"
        )

        chunks = structure_chunk(text, max_chunk_size=24)

        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 24 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_multiple_blank_lines_before_oversized_paragraph_are_preserved(self) -> None:
        text = (
            "Intro\n\n\n"
            + ("A long paragraph with preserved words. " * 20)
            + "\n\n\nOutro\n\n"
        )

        chunks = structure_chunk(text, max_chunk_size=45)

        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 45 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_oversized_section_chunks_retain_the_same_section_metadata(self) -> None:
        text = (
            "Experiment 1: Networking\n"
            "Aim: measure packet loss\n"
            + ("across the network " * 30)
            + "\n"
        )

        chunks = structure_chunk(text, max_chunk_size=50)
        aim_chunks = [chunk for chunk in chunks if chunk.heading == "Aim"]

        self.assertGreater(len(aim_chunks), 1)
        self.assertTrue(
            all(
                (
                    chunk.heading,
                    chunk.heading_level,
                    chunk.parent_heading,
                    chunk.section_path,
                    chunk.heading_occurrence,
                )
                == (
                    aim_chunks[0].heading,
                    aim_chunks[0].heading_level,
                    aim_chunks[0].parent_heading,
                    aim_chunks[0].section_path,
                    aim_chunks[0].heading_occurrence,
                )
                for chunk in aim_chunks
            )
        )
        self.assertEqual(aim_chunks[0].heading_level, 2)
        self.assertEqual(aim_chunks[0].parent_heading, "Experiment 1: Networking")
        self.assertEqual(
            aim_chunks[0].section_path,
            ("Experiment 1: Networking", "Aim"),
        )
        self.assertTrue(all(len(chunk.text) <= 50 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_whitespace_only_input_returns_no_chunks(self) -> None:
        self.assertEqual(structure_chunk(" \n\t\r\n "), [])

    def test_false_positive_heading_detection_is_rejected(self) -> None:
        text = (
            "The aim of this experiment is to study the signal.\n"
            "The theory explains the observed timing jitter.\n"
            "The procedure was revised after the first trial.\n"
            "The result was stable and reproducible.\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].heading)
        self.assertEqual(chunks[0].text, text)

    def test_numbered_ordinary_sentence_starting_with_heading_keyword_is_not_heading(
        self,
    ) -> None:
        text = "1. Aim of this experiment is to study networks.\n"

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].heading)
        self.assertEqual(chunks[0].text, text)

    def test_lowercase_heading_titles_are_recognized(self) -> None:
        text = (
            "Experiment 1: study of networking\n"
            "Aim: measure packet loss\n"
            "Theory: analysis of routing behavior\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual(
            [chunk.heading for chunk in chunks],
            [
                "Experiment 1: study of networking",
                "Aim",
                "Theory",
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_level_two_heading_before_experiment_has_no_parent(self) -> None:
        text = "Aim: Measure the delay.\n"

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading, "Aim")
        self.assertEqual(chunks[0].heading_level, 2)
        self.assertIsNone(chunks[0].parent_heading)
        self.assertEqual(chunks[0].section_path, ("Aim",))

    def test_heading_metadata_is_normalized_but_chunk_text_is_exact(self) -> None:
        text = "  experiment 1 : Networking  \r\n"

        chunks = structure_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading, "Experiment 1: Networking")
        self.assertEqual(chunks[0].text, text)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_exact_reconstruction_with_blank_lines_crlf_and_trailing_newlines(self) -> None:
        text = (
            "Experiment 1: Queueing\r\n"
            "\r\n"
            "Aim:\r\n"
            "Measure the queue delay.\r\n\r\n"
            "Result:\r\n"
            "Stable.\r\n"
        )

        chunks = structure_chunk(text)

        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertTrue(all(chunk.text for chunk in chunks))


class StructurePreviewTests(unittest.TestCase):
    """Integration tests for the structure preview script."""

    def test_preview_rejects_invalid_max_chunk_size_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_pdf = Path(temporary_directory) / "manual.pdf"
            source_pdf.touch()

            for invalid_size in (True, 0, -1, 1.5):
                with self.subTest(max_chunk_size=invalid_size):
                    with (
                        patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                        patch.object(
                            preview_structure,
                            "MAX_CHUNK_SIZE",
                            invalid_size,
                        ),
                        patch.object(
                            preview_structure,
                            "extract_pdf_pages",
                        ) as extract_pages,
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "MAX_CHUNK_SIZE must be a positive integer",
                        ):
                            preview_structure.main()

                    extract_pages.assert_not_called()

    def test_chunk_metadata_consistency_is_validated(self) -> None:
        invalid_chunks = (
            StructureChunk(
                chunk_index=1,
                text="Text",
                heading=None,
                heading_level=1,
                parent_heading=None,
                section_path=(),
            ),
            StructureChunk(
                chunk_index=1,
                text="Text",
                heading="Aim",
                heading_level=4,
                parent_heading=None,
                section_path=("Aim",),
            ),
            StructureChunk(
                chunk_index=1,
                text="Text",
                heading="Aim",
                heading_level=2,
                parent_heading=None,
                section_path=(),
            ),
            StructureChunk(
                chunk_index=1,
                text="Text",
                heading="Aim",
                heading_level=3,
                parent_heading="Experiment",
                section_path=("Wrong experiment", "Aim"),
            ),
        )

        for chunk in invalid_chunks:
            with self.subTest(chunk=chunk):
                with self.assertRaises(RuntimeError):
                    preview_structure._validate_chunks([chunk], "Text", 10)

    def test_preview_rejects_oversized_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "structure_preview"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text="Experiment 1: Demo\nAim: Observe.\n")],
                empty_pages=(),
            )

            oversized_text = "A" * 80
            oversized_chunks = [
                StructureChunk(
                    chunk_index=1,
                    text=oversized_text,
                    heading="Experiment 1: Demo",
                    heading_level=1,
                    parent_heading=None,
                    section_path=("Experiment 1: Demo",),
                )
            ]

            with (
                patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                patch.object(preview_structure, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_structure,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_structure, "clean_text", side_effect=lambda text: oversized_text),
                patch.object(
                    preview_structure,
                    "structure_chunk",
                    return_value=oversized_chunks,
                ),
                patch.object(preview_structure, "MAX_CHUNK_SIZE", 64),
            ):
                with self.assertRaisesRegex(RuntimeError, "size limit"):
                    preview_structure.main()

    def test_preview_rejects_empty_extracted_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "structure_preview"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text="\n\t\n")],
                empty_pages=(),
            )

            with (
                patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                patch.object(preview_structure, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_structure,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_structure, "clean_text", side_effect=lambda text: ""),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "No usable text was extracted after cleaning",
                ):
                    preview_structure.main()

    def test_preview_validates_exact_reconstruction_and_heading_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "structure_preview"
            text = "Experiment 1: Demo\nAim: Measure output.\nResult: Stable.\n"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text=text)],
                empty_pages=(),
            )

            with (
                patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                patch.object(preview_structure, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_structure,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_structure, "clean_text", side_effect=lambda value: value),
            ):
                preview_structure.main()

            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (output_directory / "chunk_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue((output_directory / "chunks.txt").is_file())
            self.assertTrue((output_directory / "summary.json").is_file())
            self.assertTrue((output_directory / "chunk_metadata.json").is_file())
            reconstructed = "".join(chunk["text"] for chunk in metadata)
            self.assertEqual(reconstructed, text)
            self.assertEqual(summary["chunk_count_by_heading"]["Experiment 1: Demo"], 1)
            self.assertEqual(summary["chunk_count_by_heading"]["Aim"], 1)
            self.assertEqual(summary["chunk_count_by_heading"]["Result"], 1)
            self.assertEqual(
                set(metadata[0]),
                {
                    "chunk_index",
                    "heading",
                    "heading_level",
                    "parent_heading",
                    "section_path",
                    "heading_occurrence",
                    "character_count",
                    "text",
                },
            )
            self.assertEqual(
                metadata[0]["section_path"],
                ["Experiment 1: Demo"],
            )
            self.assertEqual(metadata[0]["heading_level"], 1)
            self.assertEqual(summary["number_of_chunks"], 3)
            self.assertEqual(summary["number_of_unique_headings"], 3)
            self.assertEqual(summary["number_of_heading_occurrences"], 3)
            self.assertEqual(summary["number_of_unheaded_chunks"], 0)
            self.assertEqual(
                summary["heading_occurrence_count_by_heading"],
                {"Experiment 1: Demo": 1, "Aim": 1, "Result": 1},
            )
            self.assertEqual(
                summary["chunk_size_statistics_characters"],
                {
                    "average": len(text) / 3,
                    "minimum": len("Result: Stable.\n"),
                    "maximum": len("Aim: Measure output.\n"),
                },
            )

    def test_preview_counts_heading_occurrences_separately_from_split_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "structure_preview"
            text = (
                "Experiment: 1\n"
                "Aim: measure the network.\n"
                + ("Network details follow.\n" * 8)
                + "Aim: Confirm the result.\n"
            )
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text=text)],
                empty_pages=(),
            )

            with (
                patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                patch.object(preview_structure, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_structure,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(
                    preview_structure,
                    "clean_text",
                    side_effect=lambda value: value,
                ),
                patch.object(preview_structure, "MAX_CHUNK_SIZE", 30),
            ):
                preview_structure.main()

            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (output_directory / "chunk_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            aim_chunks = [item for item in metadata if item["heading"] == "Aim"]

            self.assertGreater(len(aim_chunks), 2)
            self.assertEqual(summary["number_of_heading_occurrences"], 3)
            self.assertEqual(
                summary["heading_occurrence_count_by_heading"]["Aim"],
                2,
            )
            self.assertEqual(
                len({item["heading_occurrence"] for item in aim_chunks}),
                2,
            )
            self.assertEqual(
                summary["chunk_count_by_heading"]["Aim"],
                len(aim_chunks),
            )
            self.assertTrue(
                all(item["character_count"] <= 30 for item in metadata)
            )
            self.assertEqual("".join(item["text"] for item in metadata), text)

    def test_existing_outputs_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "structure_preview"
            output_directory.mkdir()
            existing_file = output_directory / "summary.json"
            existing_file.write_text("keep this", encoding="utf-8")

            with (
                patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                patch.object(preview_structure, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_structure,
                    "extract_pdf_pages",
                ) as extract_pages,
            ):
                with self.assertRaises(FileExistsError):
                    preview_structure.main()

            self.assertEqual(existing_file.read_text(encoding="utf-8"), "keep this")
            extract_pages.assert_not_called()

    def test_partial_output_is_cleaned_after_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "manual.pdf"
            source_pdf.touch()
            output_directory = root / "structure_preview"
            text = "Experiment 1: Demo\nAim: Measure output.\n"
            extraction = SimpleNamespace(
                pages=[SimpleNamespace(text=text)],
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
                    self.file.write(content[:10])
                    raise OSError("simulated write failure")

            def fail_summary_write(
                path: Path,
                *args: object,
                **kwargs: object,
            ) -> object:
                if path.name == "summary.json":
                    return FailingWriter(path, *args, **kwargs)
                return original_open(path, *args, **kwargs)

            with (
                patch.object(preview_structure, "SOURCE_PDF", source_pdf),
                patch.object(preview_structure, "OUTPUT_DIRECTORY", output_directory),
                patch.object(
                    preview_structure,
                    "extract_pdf_pages",
                    return_value=extraction,
                ),
                patch.object(preview_structure, "clean_text", side_effect=lambda value: value),
                patch.object(Path, "open", new=fail_summary_write),
            ):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    preview_structure.main()

            self.assertFalse((output_directory / "chunks.txt").exists())
            self.assertFalse((output_directory / "summary.json").exists())
            self.assertFalse((output_directory / "chunk_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
