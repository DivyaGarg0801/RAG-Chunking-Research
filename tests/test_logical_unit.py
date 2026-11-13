"""Tests for experiment-aware logical-unit chunking."""

import unittest

from src.chunking.logical_unit import logical_unit_chunk


class LogicalUnitChunkTests(unittest.TestCase):
    """Exercise the public logical-unit API and document the current contract."""

    def assert_valid_reconstruction(
        self,
        original_text: str,
        chunks: list,
        max_chunk_size: int | None = None,
    ) -> None:
        self.assertEqual(
            "".join(chunk.text for chunk in chunks),
            original_text,
            "Logical-unit chunk texts did not reconstruct the original text exactly.",
        )
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(1, len(chunks) + 1)),
            "Logical-unit chunk indexes must be sequential and start at 1.",
        )
        if max_chunk_size is not None:
            self.assertTrue(
                all(len(chunk.text) <= max_chunk_size for chunk in chunks),
                "Every logical-unit chunk must respect the configured maximum size: "
                f"{[len(chunk.text) for chunk in chunks]!r}",
            )

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(logical_unit_chunk("", max_chunk_size=500), [])

    def test_whitespace_only_input_is_preserved_as_a_fallback_block(self) -> None:
        text = " \n\t \n"
        chunks = logical_unit_chunk(text, max_chunk_size=10)
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].experiment_id)
        self.assertIsNone(chunks[0].experiment_title)
        self.assertEqual(chunks[0].text, text)
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=10)

    def test_non_string_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            logical_unit_chunk(None)

    def test_invalid_max_chunk_sizes_raise_value_error(self) -> None:
        for max_chunk_size in (0, -1, True):
            with self.subTest(max_chunk_size=max_chunk_size):
                with self.assertRaises(ValueError):
                    logical_unit_chunk("Experiment 1: Sample\nResult: OK\n", max_chunk_size=max_chunk_size)

    def test_max_chunk_size_of_one_is_supported(self) -> None:
        text = "ABC\nDEF\n"
        chunks = logical_unit_chunk(text, max_chunk_size=1)
        self.assertTrue(all(len(chunk.text) <= 1 for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=1)

    def test_leading_and_trailing_whitespace_are_preserved(self) -> None:
        text = " \nExperiment 1: Sample\nResult: OK\n \n"
        chunks = logical_unit_chunk(text, max_chunk_size=50)
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=50)

    def test_single_experiment_is_kept_as_one_chunk_when_it_fits(self) -> None:
        text = (
            "Experiment 1: Packet loss\n"
            "Aim: Measure the drop rate.\n"
            "Result: Loss stayed below 5%.\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=500)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].experiment_id, "Experiment 1")
        self.assertEqual(chunks[0].experiment_title, "Packet loss")
        self.assertEqual(chunks[0].section, "Experiment 1: Packet loss")
        self.assertEqual(chunks[0].text, text)
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=500)

    def test_large_experiment_is_split_without_losing_metadata(self) -> None:
        text = (
            "Experiment 3: Queueing\n"
            "Aim: Measure delay.\n"
            "Theory: Bursts fill the queue.\n"
            "Procedure: Run the test under load.\n"
            "Result: Delay remains lower than expected.\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.experiment_id == "Experiment 3" for chunk in chunks))
        self.assertTrue(all(chunk.experiment_title == "Queueing" for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=80)

    def test_repeated_experiment_numbers_are_distinct_occurrences_when_not_nested(self) -> None:
        text = (
            "Experiment 1: First run\n"
            "Result: Alpha\n"
            "Experiment 1\n"
            "Result: Beta\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=500)

        self.assertEqual([chunk.experiment_id for chunk in chunks], ["Experiment 1", "Experiment 1"])
        self.assertEqual([chunk.experiment_title for chunk in chunks], ["First run", None])
        self.assertEqual([chunk.experiment_occurrence_id for chunk in chunks], [1, 2])
        self.assertTrue(all(chunk.parent_experiment_id is None for chunk in chunks))
        self.assertTrue(all(chunk.parent_experiment_occurrence_id is None for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=500)

    def test_nested_experiments_keep_parent_relationships(self) -> None:
        text = (
            "Experiment 1: Outer\n"
            "Aim: Outer aim\n"
            "Experiment 1: Inner\n"
            "Result: Inner result\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=500)

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk.experiment_id for chunk in chunks], ["Experiment 1", "Experiment 1"])
        self.assertEqual([chunk.experiment_title for chunk in chunks], ["Outer", "Inner"])
        self.assertEqual([chunk.experiment_occurrence_id for chunk in chunks], [1, 2])
        self.assertEqual([chunk.parent_experiment_id for chunk in chunks], [None, "Experiment 1"])
        self.assertEqual([chunk.parent_experiment_occurrence_id for chunk in chunks], [None, 1])
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=500)

    def test_section_metadata_matches_the_current_contract(self) -> None:
        complete = (
            "Experiment 1: Packet loss\n"
            "Aim: Measure the drop rate.\n"
            "Theory: Congestion may occur.\n"
            "Procedure: Run the test.\n"
        )
        chunks = logical_unit_chunk(complete, max_chunk_size=500)
        self.assertEqual(chunks[0].section, "Experiment 1: Packet loss")

        oversized = (
            "Experiment 1: Packet loss\n"
            "Aim: Measure the drop rate.\n"
            "Theory: Congestion may occur.\n"
            "Procedure: Run the test.\n"
            "Result: Loss stayed below 5%.\n"
        )
        split_chunks = logical_unit_chunk(oversized, max_chunk_size=30)
        self.assertTrue(all(chunk.section == "Experiment 1: Packet loss" for chunk in split_chunks))

        for label, value in (
            ("Aim", "Aim: Evaluate the system.\n"),
            ("Theory", "Theory: Latency matters.\n"),
            ("Requirements", "Requirements: Python 3.13.\n"),
            ("Procedure", "Procedure: Run the benchmark.\n"),
            ("Result", "Result: The benchmark passed.\n"),
        ):
            with self.subTest(section=label):
                fallback = logical_unit_chunk(value, max_chunk_size=500)
                self.assertTrue(all(chunk.experiment_id is None for chunk in fallback))
                self.assertTrue(all(chunk.experiment_title is None for chunk in fallback))
                self.assertEqual(fallback[0].section, label)
                self.assert_valid_reconstruction(value, fallback, max_chunk_size=500)

        bare_fallback = logical_unit_chunk("This is a plain sentence with no detectable heading.\n", max_chunk_size=500)
        self.assertTrue(all(chunk.section is None for chunk in bare_fallback))
        self.assert_valid_reconstruction("This is a plain sentence with no detectable heading.\n", bare_fallback, max_chunk_size=500)

    def test_non_experiment_text_is_preserved_as_a_fallback_block(self) -> None:
        text = "This is a short preamble.\nIt is not labelled as an experiment.\n"
        chunks = logical_unit_chunk(text, max_chunk_size=50)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.experiment_id is None for chunk in chunks))
        self.assertTrue(all(chunk.experiment_title is None for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 50 for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=50)

    def test_consecutive_non_experiment_sections_are_handled_consistently(self) -> None:
        text = (
            "Preface\n"
            "Aim: Not an experiment heading.\n"
            "Theory: Still non-experiment context.\n"
            "Experiment 1: Packet loss\n"
            "Result: See the result.\n"
        )
        chunks = logical_unit_chunk(text, max_chunk_size=500)

        self.assertTrue(any(chunk.experiment_id is None for chunk in chunks))
        self.assertTrue(any(chunk.experiment_id == "Experiment 1" for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=500)

    def test_metadata_is_consistent_across_split_experiment_chunks(self) -> None:
        text = "Experiment 1: Sample\n" + ("Aim: Detail\n" * 50)
        chunks = logical_unit_chunk(text, max_chunk_size=40)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.experiment_id == "Experiment 1" for chunk in chunks))
        self.assertTrue(all(chunk.experiment_title == "Sample" for chunk in chunks))
        self.assertTrue(len({chunk.experiment_occurrence_id for chunk in chunks}) == 1)
        self.assertTrue(all(chunk.parent_experiment_id is None for chunk in chunks))
        self.assertTrue(all(chunk.parent_experiment_occurrence_id is None for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=40)

    def test_source_pages_are_empty_or_preserved_per_document_contract(self) -> None:
        text = "Experiment 1: Packet loss\nResult: Loss stayed below 5%.\n"
        chunks = logical_unit_chunk(text)
        self.assertTrue(all(chunk.source_pages == () for chunk in chunks))

        explicit_pages = (2, 5, 9)
        chunks_with_pages = logical_unit_chunk(text, source_pages=explicit_pages)
        self.assertTrue(all(chunk.source_pages == explicit_pages for chunk in chunks_with_pages))
        self.assert_valid_reconstruction(text, chunks_with_pages)

    def test_max_chunk_size_is_enforced_for_every_output_chunk(self) -> None:
        text = (
            "This is a long leading paragraph with many repeated words. " * 40
            + "\nExperiment 1: Throughput\nAim: Show the improvement.\n"
            + "\nThis is a trailing paragraph. " * 40
        )
        chunks = logical_unit_chunk(text, max_chunk_size=64)
        self.assertTrue(all(len(chunk.text) <= 64 for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=64)

    def test_oversized_paragraph_and_line_are_split_without_loss(self) -> None:
        text = ("A" * 5000) + "\nExperiment 1: Queueing\nResult: Done.\n"
        chunks = logical_unit_chunk(text, max_chunk_size=80)
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=80)

    def test_multiple_experiments_remain_separate(self) -> None:
        text = (
            "Experiment 1: Packet loss\n"
            "Aim: Measure the drop rate.\n"
            "Experiment 2: Throughput\n"
            "Result: Throughput increased.\n"
        )

        chunks = logical_unit_chunk(text, max_chunk_size=500)

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk.experiment_id for chunk in chunks], ["Experiment 1", "Experiment 2"])
        self.assertEqual([chunk.experiment_title for chunk in chunks], ["Packet loss", "Throughput"])
        self.assert_valid_reconstruction(text, chunks, max_chunk_size=500)


if __name__ == "__main__":
    unittest.main()
