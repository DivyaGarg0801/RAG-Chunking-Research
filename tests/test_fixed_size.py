"""Tests for character-based fixed-size chunking."""

import unittest

from src.chunking.fixed_size import fixed_size_chunk


class FixedSizeChunkTests(unittest.TestCase):
    """Check chunk boundaries, validation, and source-text preservation."""

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(fixed_size_chunk("", chunk_size=4), [])

    def test_input_shorter_than_chunk_size_returns_one_chunk(self) -> None:
        self.assertEqual(fixed_size_chunk("abc", chunk_size=4), ["abc"])

    def test_input_equal_to_chunk_size_returns_one_chunk(self) -> None:
        self.assertEqual(fixed_size_chunk("abcd", chunk_size=4), ["abcd"])

    def test_input_longer_than_chunk_size_is_split(self) -> None:
        self.assertEqual(
            fixed_size_chunk("abcdefghij", chunk_size=4),
            ["abcd", "efgh", "ij"],
        )

    def test_adjacent_chunks_share_the_configured_overlap(self) -> None:
        chunks = fixed_size_chunk("abcdefgh", chunk_size=4, overlap=2)

        self.assertEqual(chunks, ["abcd", "cdef", "efgh"])
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous[-2:], following[:2])

    def test_zero_overlap_produces_consecutive_nonoverlapping_chunks(self) -> None:
        self.assertEqual(
            fixed_size_chunk("abcdef", chunk_size=3, overlap=0),
            ["abc", "def"],
        )

    def test_zero_or_negative_chunk_size_is_rejected(self) -> None:
        for chunk_size in (0, -1):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    fixed_size_chunk("text", chunk_size=chunk_size)

    def test_negative_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            fixed_size_chunk("text", chunk_size=3, overlap=-1)

    def test_overlap_equal_to_chunk_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            fixed_size_chunk("text", chunk_size=3, overlap=3)

    def test_overlap_greater_than_chunk_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            fixed_size_chunk("text", chunk_size=3, overlap=4)

    def test_preserves_punctuation_newlines_symbols_and_code_indentation(self) -> None:
        text = "Program:\n    x = α + 2!\nResult: ∑x²."
        chunks = fixed_size_chunk(text, chunk_size=9, overlap=3)

        reconstructed = chunks[0] + "".join(chunk[3:] for chunk in chunks[1:])
        self.assertEqual(reconstructed, text)

    def test_reconstruction_preserves_order_and_covers_every_character(self) -> None:
        text = "Aim:\n1. Start\n2. Stop."
        chunks = fixed_size_chunk(text, chunk_size=7, overlap=2)

        reconstructed = chunks[0] + "".join(chunk[2:] for chunk in chunks[1:])

        self.assertEqual(reconstructed, text)

    def test_does_not_emit_trailing_chunk_when_text_is_already_covered(self) -> None:
        chunks = fixed_size_chunk("abcdef", chunk_size=4, overlap=2)

        self.assertEqual(chunks, ["abcd", "cdef"])


if __name__ == "__main__":
    unittest.main()
