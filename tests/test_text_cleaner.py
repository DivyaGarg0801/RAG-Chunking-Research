"""Tests for conservative extracted-text cleaning."""

import unittest

from src.preprocessing.text_cleaner import clean_text


class CleanTextTests(unittest.TestCase):
    """Check that only unnecessary surrounding and repeated blank lines change."""

    def test_removes_leading_and_trailing_blank_lines(self) -> None:
        self.assertEqual(clean_text("\n  \nManual title\nEnd\n \n"), "Manual title\nEnd")

    def test_reduces_excessive_blank_lines_to_one(self) -> None:
        self.assertEqual(
            clean_text("First paragraph\n\n\n\nSecond paragraph"),
            "First paragraph\n\nSecond paragraph",
        )

    def test_preserves_headings_and_paragraph_boundaries(self) -> None:
        text = "Aim\nTo study networks.\n\nTheory\nA network connects devices."

        self.assertEqual(clean_text(text), text)

    def test_preserves_numbered_algorithm_steps(self) -> None:
        text = "Algorithm\n1. Start\n2. Configure the router\n3. Stop"

        self.assertEqual(clean_text(text), text)

    def test_preserves_code_indentation_and_horizontal_whitespace(self) -> None:
        text = "Program\n    if ready:\n\tprint(  'yes'  )\n"

        self.assertEqual(
            clean_text(text),
            "Program\n    if ready:\n\tprint(  'yes'  )",
        )

    def test_empty_and_whitespace_only_input_return_empty_text(self) -> None:
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(" \t\n \r\n\t "), "")

    def test_preserves_punctuation_and_mathematical_symbols(self) -> None:
        text = "Result: x <= 5, y != 0; sum = a + b - c. α → β, ∑x²."

        self.assertEqual(clean_text(text), text)

    def test_normalizes_windows_and_old_mac_line_endings(self) -> None:
        self.assertEqual(clean_text("Aim\r\nFirst line\rSecond line"), "Aim\nFirst line\nSecond line")


if __name__ == "__main__":
    unittest.main()
