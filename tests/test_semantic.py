"""Tests for semantic chunking based on sentence similarity."""

import unittest

from src.chunking.semantic import semantic_chunk


class SemanticChunkTests(unittest.TestCase):
    """Check semantic splitting, parameter validation, and conservative boundaries."""

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(semantic_chunk("", chunk_size=100), [])

    def test_short_text_returns_one_chunk(self) -> None:
        text = "The experiment compares packet delivery times."
        self.assertEqual(semantic_chunk(text, chunk_size=200), [text])

    def test_invalid_chunk_size_and_threshold_are_rejected(self) -> None:
        invalid_chunk_sizes = (0, -1)
        for chunk_size in invalid_chunk_sizes:
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    semantic_chunk("Example sentence.", chunk_size=chunk_size)

        for threshold in (-0.1, 1.1, True):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "similarity_threshold"):
                    semantic_chunk("Example sentence.", chunk_size=10, similarity_threshold=threshold)

    def test_topic_shift_creates_multiple_chunks(self) -> None:
        text = (
            "The aim is to compare network throughput. "
            "This theory explains congestion and packet loss. "
            "The algorithm measures average latency across each run. "
            "The moon landing happened in 1969."
        )

        chunks = semantic_chunk(text, chunk_size=120, similarity_threshold=0.18)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), " ".join(text.split()))

    def test_chunk_size_enforces_a_boundary(self) -> None:
        text = (
            "Sentence one explains the experiment. "
            "Sentence two adds the algorithm. "
            "Sentence three describes the output. "
            "Sentence four adds the conclusion."
        )

        chunks = semantic_chunk(text, chunk_size=70, similarity_threshold=0.05)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 70 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
