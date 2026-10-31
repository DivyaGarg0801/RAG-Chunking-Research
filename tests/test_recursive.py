"""Tests for recursive chunking and structural boundary preservation."""

import re
import unittest

from src.chunking.fixed_size import fixed_size_chunk
from src.chunking.recursive import RecursiveChunk, recursive_chunk


def reconstruct_chunks(chunks: list[RecursiveChunk]) -> str:
    """Remove recorded overlap and reconstruct the original source text."""
    if not chunks:
        return ""

    reconstructed = chunks[0].text
    for previous, chunk in zip(chunks, chunks[1:]):
        overlap = chunk.overlap_length
        if overlap:
            if chunk.text[:overlap] != previous.text[-overlap:]:
                raise AssertionError("Recorded overlap does not match chunk text.")
        reconstructed += chunk.text[overlap:]

    return reconstructed


def count_non_boundary_ends(
    chunks: list[str],
    valid_boundaries: set[int],
) -> int:
    """Count chunk ends that do not coincide with a chosen text boundary."""
    position = 0
    invalid_ends = 0

    for chunk in chunks[:-1]:
        position += len(chunk)
        if position not in valid_boundaries:
            invalid_ends += 1

    return invalid_ends


class RecursiveChunkTests(unittest.TestCase):
    """Check recursive splitting, overlap, and preservation behavior."""

    def test_empty_input_returns_no_chunks(self) -> None:
        self.assertEqual(recursive_chunk("", chunk_size=10), [])

    def test_invalid_chunk_size_and_overlap_are_rejected(self) -> None:
        invalid_settings = (
            (0, 0),
            (-1, 0),
            (10, -1),
            (10, 10),
            (10, 11),
            (10.5, 0),
            (10, 2.5),
            (True, 0),
        )
        for chunk_size, overlap in invalid_settings:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    recursive_chunk("some text", chunk_size, overlap)

    def test_no_chunk_exceeds_the_maximum_size(self) -> None:
        text = (
            "First paragraph has several words.\n\n"
            "Second paragraph also has several words.\n\n"
            "Third paragraph ends the example."
        )

        chunks = recursive_chunk(text, chunk_size=24, overlap=5)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= 24 for chunk in chunks))

    def test_content_and_order_are_preserved_with_overlap(self) -> None:
        text = "Aim:\n1. Keep punctuation, α symbols, and order!\n\nResult: complete."

        chunks = recursive_chunk(text, chunk_size=16, overlap=5)

        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_overlap_prefers_a_word_boundary_and_records_actual_length(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta"

        chunks = recursive_chunk(text, chunk_size=18, overlap=7)

        self.assertGreater(len(chunks), 1)
        previous, chunk = chunks[0], chunks[1]
        self.assertLess(chunk.overlap_length, 7)
        self.assertEqual(
            chunk.text[: chunk.overlap_length],
            previous.text[-chunk.overlap_length :],
        )
        source_position = len(previous.text) - previous.overlap_length
        overlap_start = source_position - chunk.overlap_length
        self.assertTrue(text[overlap_start - 1].isspace())
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_overlap_falls_back_to_exact_characters_without_boundaries(self) -> None:
        chunks = recursive_chunk("abcdefghijklmno", chunk_size=6, overlap=2)

        self.assertEqual(chunks[1].overlap_length, 2)
        self.assertEqual(chunks[1].text[:2], chunks[0].text[-2:])
        self.assertEqual(reconstruct_chunks(chunks), "abcdefghijklmno")

    def test_every_adjacent_overlap_matches_previous_chunk_suffix(self) -> None:
        text = (
            "Paragraph one contains several words and ends here.\n\n"
            "Paragraph two contains more words and ends there.\n\n"
            "Paragraph three finishes the source."
        )

        chunks = recursive_chunk(text, chunk_size=27, overlap=9)

        self.assertGreater(len(chunks), 2)
        for previous, chunk in zip(chunks, chunks[1:]):
            overlap = chunk.overlap_length
            self.assertLessEqual(overlap, len(previous.text))
            self.assertEqual(
                chunk.text[:overlap] if overlap else "",
                previous.text[-overlap:] if overlap else "",
            )
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_natural_overlap_can_equal_requested_maximum(self) -> None:
        text = "abcdef xy abcdefghijklmnopqrstuvwxyz"

        chunks = recursive_chunk(text, chunk_size=10, overlap=3)

        self.assertEqual(chunks[1].overlap_length, 3)
        self.assertEqual(chunks[0].text[-3:], "xy ")
        self.assertEqual(chunks[1].text[:3], "xy ")
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_preserves_abbreviations_decimals_and_closing_quotes(self) -> None:
        text = 'Dr. Smith measured 3.14 seconds. He said "stable." Next reading: 2.71.'

        chunks = recursive_chunk(text, chunk_size=22, overlap=4)

        self.assertTrue(all(len(chunk.text) <= 22 for chunk in chunks))
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_preserves_leading_and_trailing_whitespace(self) -> None:
        text = "  leading text and trailing text  \n"

        chunks = recursive_chunk(text, chunk_size=12, overlap=3)

        self.assertEqual(reconstruct_chunks(chunks), text)
        self.assertTrue(chunks[0].text.startswith("  "))
        self.assertTrue(chunks[-1].text.endswith("  \n"))

    def test_single_character_input_fits_chunk_size_one(self) -> None:
        chunks = recursive_chunk("x", chunk_size=1)

        self.assertEqual(chunks, [RecursiveChunk(text="x", overlap_length=0)])

    def test_zero_overlap_preserves_multiple_chunks(self) -> None:
        text = "abcdefghij klmnopqrst uvwxyz"

        chunks = recursive_chunk(text, chunk_size=10, overlap=0)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.overlap_length == 0 for chunk in chunks))
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_long_paragraph_uses_smaller_boundaries(self) -> None:
        text = (
            "A long paragraph contains a first sentence. "
            "It contains a second sentence, followed by additional words "
            "that make the paragraph longer than one chunk."
        )

        chunks = recursive_chunk(text, chunk_size=45)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 45 for chunk in chunks))
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_unbroken_text_uses_character_fallback(self) -> None:
        text = "x" * 53

        chunks = recursive_chunk(text, chunk_size=12)

        self.assertTrue(all(len(chunk.text) <= 12 for chunk in chunks))
        self.assertEqual(reconstruct_chunks(chunks), text)

    def test_synthetic_structural_comparison_with_fixed_size(self) -> None:
        paragraphs = [
            f"Experiment {number} examines routing behavior. "
            "It records packet delay and loss."
            for number in range(1, 5)
        ]
        text = "\n\n".join(paragraphs)
        chunk_size = 100

        fixed_chunks = fixed_size_chunk(text, chunk_size, overlap=0)
        recursive_chunks = recursive_chunk(text, chunk_size, overlap=0)
        paragraph_and_sentence_boundaries = {
            match.end()
            for pattern in (re.compile(r"\n\n"), re.compile(r"(?<=[.!?])\s+"))
            for match in pattern.finditer(text)
        }
        fixed_non_boundary_ends = count_non_boundary_ends(
            fixed_chunks,
            paragraph_and_sentence_boundaries,
        )
        recursive_non_boundary_ends = count_non_boundary_ends(
            [chunk.text for chunk in recursive_chunks],
            paragraph_and_sentence_boundaries,
        )

        self.assertLess(recursive_non_boundary_ends, fixed_non_boundary_ends)


if __name__ == "__main__":
    unittest.main()
