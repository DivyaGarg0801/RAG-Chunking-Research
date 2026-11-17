"""Tests for logical-unit-based chunking and its PDF preview."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts import preview_logical
from src.chunking.logical import (
    LogicalChunk,
    _page_numbers_for_span,
    _section_heading,
    _sentence_boundaries,
    _validate_chunks,
    logical_unit_chunk,
)
from src.ingestion.pdf_loader import PdfPage
from src.ingestion.extraction_artifact import page_preserving_text


class LogicalChunkTests(unittest.TestCase):
    """Check experiment grouping, metadata, size limits, and preservation."""

    def test_complete_experiment_stays_one_logical_chunk(self) -> None:
        text = (
            "Experiment 1: Packet delay\n"
            "Aim: Measure delay.\n"
            "Theory: Queue length affects delay.\n"
            "Procedure: Run the test.\n"
            "Result: Delay was low.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].logical_unit_type, "experiment")
        self.assertEqual(chunks[0].logical_unit_title, "Experiment 1: Packet delay")
        self.assertEqual(chunks[0].experiment_number, "1")
        self.assertIsNone(chunks[0].parent_heading)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_experiment_with_missing_sections_is_still_one_unit(self) -> None:
        text = "Experiment 3: Route test\nProcedure: Run ping.\n"

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].logical_unit_title, "Experiment 3: Route test")
        self.assertEqual(chunks[0].logical_unit_id, "logical-unit-0001")

    def test_multiple_experiments_receive_separate_units_in_source_order(self) -> None:
        text = (
            "Experiment 1: Ping\nAim: Check connectivity.\n"
            "Experiment 2: DNS\nAim: Resolve a name.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(
            [chunk.logical_unit_title for chunk in chunks],
            ["Experiment 1: Ping", "Experiment 2: DNS"],
        )
        self.assertEqual(
            [chunk.logical_unit_id for chunk in chunks],
            ["logical-unit-0001", "logical-unit-0002"],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_nested_experiment_headings_start_child_units_with_parent_context(
        self,
    ) -> None:
        text = (
            "Experiment-4\n"
            "Experiment 1: Point-to-point link\nAim: Connect two nodes.\n"
            "Experiment 2: Broadcasting\nAim: Send to all nodes.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "Experiment-4\n")
        self.assertEqual(chunks[0].logical_unit_type, "experiment_context")
        self.assertEqual(
            [chunk.logical_unit_title for chunk in chunks],
            [
                "Experiment 4",
                "Experiment 1: Point-to-point link",
                "Experiment 2: Broadcasting",
            ],
        )
        self.assertEqual(
            [chunk.parent_heading for chunk in chunks],
            [None, "Experiment 4", "Experiment 4"],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_outer_heading_and_pre_child_text_form_a_separate_context_unit(
        self,
    ) -> None:
        context = "Experiment-4\nOverview and shared setup.\n"
        child = "Experiment 1: Point-to-point link\nAim: Connect two nodes.\n"
        text = context + child

        chunks = logical_unit_chunk(text)

        self.assertEqual([chunk.text for chunk in chunks], [context, child])
        self.assertEqual(
            [chunk.logical_unit_type for chunk in chunks],
            ["experiment_context", "experiment"],
        )
        self.assertEqual(chunks[0].logical_unit_title, "Experiment 4")
        self.assertEqual(chunks[1].parent_heading, "Experiment 4")
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_pre_child_context_stays_separate_with_multiple_child_experiments(
        self,
    ) -> None:
        context = "Experiment-5\nShared topology setup.\n"
        child_one = "Experiment 1: Wi-Fi\nAim: Learn access points.\n"
        child_two = "Experiment 2: Extending the topology\nAim: Add a node.\n"
        text = context + child_one + child_two

        chunks = logical_unit_chunk(text)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [context, child_one, child_two],
        )
        self.assertEqual(
            [chunk.logical_unit_type for chunk in chunks],
            ["experiment_context", "experiment", "experiment"],
        )
        self.assertEqual(
            [chunk.parent_heading for chunk in chunks],
            [None, "Experiment 5", "Experiment 5"],
        )
        self.assertEqual(len({chunk.logical_unit_id for chunk in chunks}), 3)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_outer_heading_without_nested_children_remains_an_experiment_unit(
        self,
    ) -> None:
        text = "Experiment-6\nAim: Study TCP.\nResult: Stable.\n"

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].logical_unit_type, "experiment")
        self.assertEqual(chunks[0].logical_unit_title, "Experiment 6")

    def test_repeated_experiment_numbers_get_distinct_appearance_ids(self) -> None:
        text = (
            "Experiment 1: First appearance\nAim: First.\n"
            "Experiment 1: Second appearance\nAim: Second.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [chunk.experiment_number for chunk in chunks],
            ["1", "1"],
        )
        self.assertNotEqual(
            chunks[0].logical_unit_id,
            chunks[1].logical_unit_id,
        )

    def test_numbered_and_lettered_procedures_are_preserved(self) -> None:
        text = (
            "Experiment 2: Capture\n"
            "Procedure:\n"
            "1. Start Wireshark.\n"
            "2. Capture packets.\n"
            "a) Filter TCP packets.\n"
            "b) Record the result.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertIn("1. Start Wireshark.\n2. Capture packets.\n", chunks[0].text)
        self.assertIn("a) Filter TCP packets.\nb) Record the result.\n", chunks[0].text)

    def test_commands_and_code_examples_remain_in_the_experiment(self) -> None:
        text = (
            "Experiment 2: TCP server\n"
            "Procedure:\n"
            "    gcc -o server tcp_server.c\n"
            "    ./server\n"
            "Program:\n"
            "    int main() { return 0; }\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertIn("    gcc -o server tcp_server.c\n", chunks[0].text)
        self.assertIn("    int main() { return 0; }\n", chunks[0].text)
        self.assertEqual(chunks[0].section_heading, None)

    def test_content_before_first_experiment_is_preserved_as_preamble(self) -> None:
        text = (
            "Computer Networks Laboratory\n\n"
            "Experiment 1: Ping\nAim: Check reachability.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(chunks[0].logical_unit_type, "preamble")
        self.assertEqual(chunks[0].logical_unit_title, "Preamble")
        self.assertEqual(chunks[1].logical_unit_type, "experiment")
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_content_between_experiments_stays_with_preceding_unit(self) -> None:
        text = (
            "Experiment 1: First\nAim: Start.\n"
            "Additional notes before the next experiment.\n"
            "Experiment 2: Second\nAim: Finish.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertIn("Additional notes", chunks[0].text)
        self.assertTrue(chunks[1].text.startswith("Experiment 2: Second"))

    def test_missing_or_inconsistent_experiment_headings_are_not_guessed(self) -> None:
        text = (
            "Experiment 1 was run on the test network.\n"
            "Experiment two: this heading has no numeric identifier.\n"
        )

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].logical_unit_type, "unclassified")
        self.assertEqual(chunks[0].experiment_number, None)
        self.assertEqual(chunks[0].text, text)

    def test_ocr_and_punctuation_variants_are_recognized(self) -> None:
        for heading in ("Experimet-10", "Experiment.2:", "Experlment5"):
            with self.subTest(heading=heading):
                chunks = logical_unit_chunk(f"{heading}\nAim: Test.\n")
                self.assertEqual(chunks[0].logical_unit_type, "experiment")
                self.assertEqual(chunks[0].experiment_number, "10" if "10" in heading else (
                    "2" if "2" in heading else "5"
                ))

    def test_period_after_experiment_number_separates_title(self) -> None:
        text = "Experiment 1. Packet Delay\nAim: Measure packet delay.\n"

        chunks = logical_unit_chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].logical_unit_title, "Experiment 1: Packet Delay")
        self.assertEqual(chunks[0].experiment_number, "1")
        self.assertEqual(chunks[0].logical_unit_type, "experiment")
        self.assertEqual(chunks[0].text, text)

    def test_period_delimiter_does_not_turn_numbered_sentences_into_headings(
        self,
    ) -> None:
        for line in (
            "Experiment 1. The packets were transmitted over the network.",
            "Experiment 2. Packet delay increased during the test.",
            "Experiment 4. Packet delay depends on network load",
            "Experiment 5. Packets travel across the network",
            "Experiment 3. 4 packets were received.",
        ):
            with self.subTest(line=line):
                chunks = logical_unit_chunk(line)
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].logical_unit_type, "unclassified")

    def test_generic_colon_heading_accepts_concise_section_titles(self) -> None:
        self.assertEqual(_section_heading("Network topology:"), "Network topology")
        self.assertEqual(_section_heading("TCP/IP Addressing:"), "TCP/IP Addressing")

    def test_generic_colon_heading_rejects_prose_commands_and_code_labels(
        self,
    ) -> None:
        non_headings = (
            "Find the IP addresses of the hosts:",
            "An Internet socket is characterized by a unique combination:",
            "Packets travel across the network:",
            "In ns-3, the visualization will appear below:",
            "$ ping 127.0.0.1:",
            "Ping -c 4:",
            "Sudo apt-get install wireshark:",
            "./server:",
            "Main:",
            "if timeout > 0:",
            "numbers 172-174 of the code:",
        )

        for line in non_headings:
            with self.subTest(line=line):
                self.assertIsNone(_section_heading(line))

    def test_oversized_experiment_splits_at_subsection_and_keeps_unit_id(self) -> None:
        text = (
            "Experiment 1: Large experiment\n"
            "Aim: "
            + ("A" * 850)
            + "\n\nTheory: "
            + ("B" * 850)
            + "\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=1500)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.character_count <= 1500 for chunk in chunks))
        self.assertTrue(all(chunk.is_split for chunk in chunks))
        self.assertEqual(len({chunk.logical_unit_id for chunk in chunks}), 1)
        self.assertEqual(chunks[1].boundary_reason, "subsection")
        self.assertEqual(chunks[1].section_heading, "Theory")
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_splitter_uses_later_meaningful_boundary_to_avoid_tiny_chunk(
        self,
    ) -> None:
        text = (
            "Experiment 1: Large\n"
            "Aim: "
            + ("A" * 50)
            + "\n\n"
            + ("B" * 60)
            + "\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=100)

        self.assertGreater(chunks[0].character_count, len("Experiment 1: Large\n"))
        self.assertEqual(chunks[1].boundary_reason, "paragraph")
        self.assertTrue(all(chunk.character_count <= 100 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_oversized_paragraph_splits_at_sentence_boundaries(self) -> None:
        text = "Experiment 1: Long\n" + ("First sentence. " * 160)

        chunks = logical_unit_chunk(text, max_chunk_size=300)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.character_count <= 300 for chunk in chunks))
        self.assertTrue(
            any(chunk.boundary_reason == "sentence" for chunk in chunks[1:])
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_sentence_boundaries_preserve_technical_abbreviations(self) -> None:
        text = "See e.g. TCP/IP, i.e. packet data. Next reading."

        boundaries = _sentence_boundaries(text)

        self.assertEqual(len(boundaries), 1)
        self.assertEqual(text[: boundaries[0]], "See e.g. TCP/IP, i.e. packet data. ")

    def test_sentence_boundaries_handle_decimal_measurements(self) -> None:
        text = "Measured 3.14 ms. Next reading."

        boundaries = _sentence_boundaries(text)

        self.assertEqual(boundaries, [len("Measured 3.14 ms. ")])
        self.assertEqual(text[boundaries[0] - 1], " ")

    def test_sentence_boundaries_remain_correct_for_long_text(self) -> None:
        text = "Dr. Rao recorded 3.14 ms. Next reading. " * 500

        boundaries = _sentence_boundaries(text)

        self.assertEqual(len(boundaries), 1000)
        self.assertTrue(all(text[position - 1].isspace() for position in boundaries))

    def test_oversized_unbroken_line_uses_safe_character_fallback(self) -> None:
        text = "Experiment 1: Long\n" + ("x" * 1700)

        chunks = logical_unit_chunk(text, max_chunk_size=80)

        self.assertTrue(all(0 < chunk.character_count <= 80 for chunk in chunks))
        self.assertIn("character", [chunk.boundary_reason for chunk in chunks])
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_oversized_line_with_words_prefers_word_boundaries(self) -> None:
        text = "Experiment 1: Long\n" + ("network packets " * 80)

        chunks = logical_unit_chunk(text, max_chunk_size=70)

        self.assertTrue(all(chunk.character_count <= 70 for chunk in chunks))
        self.assertIn("word", [chunk.boundary_reason for chunk in chunks])
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_whitespace_only_input_is_preserved_and_size_limited(self) -> None:
        text = " \t\n" * 40

        chunks = logical_unit_chunk(text, max_chunk_size=3)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.logical_unit_type == "unclassified" for chunk in chunks))
        self.assertTrue(all(chunk.character_count <= 3 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_leading_and_trailing_whitespace_are_preserved(self) -> None:
        text = " \t\nExperiment 1: Demo\nAim: Measure delay.\n \t"

        chunks = logical_unit_chunk(text, max_chunk_size=10)

        self.assertTrue(chunks[0].text.startswith(" \t\n"))
        self.assertTrue(chunks[-1].text.endswith("\n \t"))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(logical_unit_chunk(""), [])

    def test_chunk_indices_and_unit_metadata_are_consistent(self) -> None:
        text = (
            "Experiment 1: Long\n"
            "Aim: " + ("measure delay " * 20) + "\n"
            "Experiment 2: Next\nAim: Complete.\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=40)

        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(1, len(chunks) + 1)),
        )
        for logical_unit_id in {chunk.logical_unit_id for chunk in chunks}:
            unit_chunks = [
                chunk for chunk in chunks if chunk.logical_unit_id == logical_unit_id
            ]
            self.assertEqual(len({chunk.logical_unit_title for chunk in unit_chunks}), 1)
            self.assertEqual(len({chunk.is_split for chunk in unit_chunks}), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_page_and_source_document_metadata_are_recorded(self) -> None:
        page_one = "Experiment 1: Demo"
        text = page_one + "\n" + "Aim: Measure delay.\n"

        chunks = logical_unit_chunk(
            text,
            page_break_offsets=[len(page_one)],
            source_document="manual.pdf",
        )

        self.assertEqual(chunks[0].source_document, "manual.pdf")
        self.assertEqual(chunks[0].source_page_numbers, (1, 2))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_page_provenance_handles_single_pages_and_exact_page_ends(self) -> None:
        offsets = (3, 7)

        self.assertEqual(_page_numbers_for_span(0, 3, offsets), (1,))
        self.assertEqual(_page_numbers_for_span(4, 7, offsets), (2,))
        self.assertEqual(_page_numbers_for_span(0, 11, offsets), (1, 2, 3))
        self.assertEqual(_page_numbers_for_span(0, 4, offsets), (1,))
        self.assertEqual(_page_numbers_for_span(4, 8, offsets), (2,))

    def test_page_provenance_includes_empty_intermediate_page_in_order(self) -> None:
        cleaned_pages = ["one", "", "six"]
        text = "\n".join(cleaned_pages)
        offsets = preview_logical._page_break_offsets(cleaned_pages)

        self.assertEqual(offsets, (3, 4))
        self.assertEqual(_page_numbers_for_span(0, len(text), offsets), (1, 2, 3))
        self.assertEqual(_page_numbers_for_span(3, 4, offsets), (1,))
        self.assertEqual(_page_numbers_for_span(4, 5, offsets), (2,))
        self.assertEqual(_page_numbers_for_span(5, 8, offsets), (3,))

    def test_leading_and_trailing_empty_pages_have_defined_provenance(self) -> None:
        cleaned_pages = ["", "two", ""]
        text = "\n".join(cleaned_pages)
        offsets = preview_logical._page_break_offsets(cleaned_pages)

        self.assertEqual(offsets, (0, 4))
        self.assertEqual(_page_numbers_for_span(0, 1, offsets), (1,))
        self.assertEqual(_page_numbers_for_span(1, 4, offsets), (2,))
        self.assertEqual(_page_numbers_for_span(4, 5, offsets), (2,))
        self.assertEqual(_page_numbers_for_span(0, len(text), offsets), (1, 2))

    def test_chunk_ending_exactly_at_page_boundary_uses_preceding_page(self) -> None:
        first_page = "Experiment 1: Demo"
        second_page = "unbroken-second-page-content"
        text = first_page + "\n" + second_page

        chunks = logical_unit_chunk(
            text,
            max_chunk_size=len(first_page) + 1,
            page_break_offsets=[len(first_page)],
        )

        self.assertEqual(chunks[0].text, first_page + "\n")
        self.assertEqual(chunks[0].source_page_numbers, (1,))
        self.assertEqual(chunks[1].source_page_numbers, (2,))
        self.assertEqual(
            chunks[0].source_page_numbers,
            tuple(sorted(set(chunks[0].source_page_numbers))),
        )

    def test_invalid_settings_and_page_offsets_are_rejected(self) -> None:
        for maximum in (0, -1, True, 2.5):
            with self.subTest(max_chunk_size=maximum):
                with self.assertRaisesRegex(ValueError, "max_chunk_size"):
                    logical_unit_chunk("text", max_chunk_size=maximum)

        with self.assertRaisesRegex(TypeError, "text"):
            logical_unit_chunk(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "page break offset"):
            logical_unit_chunk("text", page_break_offsets=[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            logical_unit_chunk("a\nb", page_break_offsets=[1, 1])
        with self.assertRaisesRegex(ValueError, "ascending"):
            logical_unit_chunk("a\nb\nc", page_break_offsets=[3, 1])

    def test_core_validator_rejects_invalid_chunk_metadata(self) -> None:
        source = "Experiment 1: Demo\n"
        valid = logical_unit_chunk(source, source_document="manual.pdf")[0]
        invalid_chunks = (
            (replace(valid, character_count=1), source, 1500, "manual.pdf", "character counts"),
            (replace(valid, logical_unit_id="  "), source, 1500, "manual.pdf", "identifiers"),
            (replace(valid, logical_unit_title=" "), source, 1500, "manual.pdf", "titles"),
            (replace(valid, experiment_number="one"), source, 1500, "manual.pdf", "numeric"),
            (replace(valid, source_document="other.pdf"), source, 1500, "manual.pdf", "Source-document"),
            (replace(valid, source_page_numbers=(2, 1)), source, 1500, "manual.pdf", "ordered and unique"),
            (replace(valid, boundary_reason="unknown"), source, 1500, "manual.pdf", "boundary reason"),
        )

        for chunk, text, maximum, source_document, error in invalid_chunks:
            with self.subTest(chunk=chunk, error=error):
                with self.assertRaisesRegex(RuntimeError, error):
                    _validate_chunks([chunk], text, maximum, source_document)

    def test_core_validator_rejects_nonsequential_indices_and_duplicate_pages(
        self,
    ) -> None:
        source = "Experiment 1: Demo\n"
        valid = logical_unit_chunk(
            source,
            page_break_offsets=[len(source) - 1],
            source_document="manual.pdf",
        )[0]

        with self.assertRaisesRegex(RuntimeError, "indices are not sequential"):
            _validate_chunks(
                [replace(valid, chunk_index=0)],
                source,
                1500,
                "manual.pdf",
            )
        with self.assertRaisesRegex(RuntimeError, "ordered and unique"):
            _validate_chunks(
                [replace(valid, source_page_numbers=(1, 1))],
                source,
                1500,
                "manual.pdf",
            )

    def test_core_validator_checks_actual_text_size_and_unit_reappearance(self) -> None:
        too_large_source = "Experiment 1: A\n"
        too_large_chunk = logical_unit_chunk(too_large_source)[0]
        with self.assertRaisesRegex(RuntimeError, "maximum"):
            _validate_chunks([too_large_chunk], too_large_source, 3, None)

        source = (
            "Experiment 1: First\n"
            "Experiment 2: Second\n"
            "Experiment 1: Third\n"
        )
        chunks = logical_unit_chunk(source)
        reused_id = replace(chunks[2], logical_unit_id=chunks[0].logical_unit_id)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            _validate_chunks([chunks[0], chunks[1], reused_id], source, 1500, None)

    def test_core_validator_rejects_inconsistent_split_metadata(self) -> None:
        source = "Experiment 1: Long\n" + ("text " * 20)
        chunks = logical_unit_chunk(source, max_chunk_size=20)
        changed_chunk = replace(chunks[1], parent_heading="Unrelated parent")

        with self.assertRaisesRegex(RuntimeError, "inconsistent metadata"):
            _validate_chunks(
                [chunks[0], changed_chunk, *chunks[2:]],
                source,
                20,
                None,
            )

    def test_preview_validator_checks_counts_documents_and_page_range(self) -> None:
        text = "Experiment 1: Demo\n"
        valid = logical_unit_chunk(text, source_document="manual.pdf")[0]

        with self.assertRaisesRegex(RuntimeError, "character counts"):
            preview_logical._validate_chunks(
                [replace(valid, character_count=1)],
                text,
                1500,
                source_document="manual.pdf",
                page_count=1,
            )
        with self.assertRaisesRegex(RuntimeError, "Source-document"):
            preview_logical._validate_chunks(
                [valid],
                text,
                1500,
                source_document="different.pdf",
                page_count=1,
            )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            preview_logical._validate_chunks(
                [replace(valid, source_page_numbers=(2,))],
                text,
                1500,
                source_document="manual.pdf",
                page_count=1,
            )


class LogicalPreviewTests(unittest.TestCase):
    """Check preview output, refusal to overwrite, and partial-write cleanup."""

    def _run_preview(
        self,
        output_directory: Path,
        selected_output_directory: Path | None = None,
    ) -> None:
        artifact_path = output_directory.parent / "extracted_pages.json"
        document = page_preserving_text(
            "manual.pdf",
            (
                PdfPage("manual.pdf", 1, "Experiment 1: Ping\nAim: Check it.\n"),
                PdfPage("manual.pdf", 2, "Experiment 2: DNS\nAim: Resolve it.\n"),
            ),
        )
        with (
            patch.object(preview_logical, "EXTRACTION_ARTIFACT", artifact_path),
            patch.object(preview_logical, "OUTPUT_DIRECTORY", output_directory),
            patch.object(
                preview_logical,
                "load_extraction_artifact",
                return_value=document,
            ),
            redirect_stdout(io.StringIO()),
        ):
            preview_logical.main(selected_output_directory)

    def test_preview_writes_summary_and_chunk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "logical_preview"

            self._run_preview(output_directory)

            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            chunks = json.loads(
                (output_directory / "chunk_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["number_of_pages_extracted"], 2)
            self.assertEqual(summary["number_of_logical_units"], 2)
            self.assertTrue(summary["text_preserved_exactly"])
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["source_page_numbers"], [1])
            self.assertEqual(chunks[1]["source_page_numbers"], [2])

    def test_preview_can_use_a_fresh_directory_without_touching_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            default_directory = Path(temporary_directory) / "logical_preview"
            default_directory.mkdir()
            sentinel = default_directory / "preserve.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            fresh_directory = Path(temporary_directory) / "logical_preview_reviewed"

            self._run_preview(default_directory, fresh_directory)

            self.assertTrue((fresh_directory / "summary.json").is_file())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_preview_refuses_existing_outputs_without_modifying_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "logical_preview"
            output_directory.mkdir()
            existing_file = output_directory / "chunks.txt"
            existing_file.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                self._run_preview(output_directory)

            self.assertEqual(existing_file.read_text(encoding="utf-8"), "keep")

    def test_preview_cleans_partial_outputs_after_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "logical_preview"
            original_open = Path.open

            def fail_on_summary(
                path: Path,
                mode: str = "r",
                *args: object,
                **kwargs: object,
            ) -> object:
                if path.name == "summary.json" and "x" in mode:
                    raise OSError("synthetic write failure")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", new=fail_on_summary):
                with self.assertRaisesRegex(OSError, "synthetic write failure"):
                    self._run_preview(output_directory)

            self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
