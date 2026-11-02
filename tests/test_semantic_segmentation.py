"""Tests for the alternative paragraph-aware semantic segmenter."""

import unittest

from src.chunking.semantic_segmentation import segment_text_paragraph_aware


class ParagraphAwareSegmentationTests(unittest.TestCase):
    def test_joins_wrapped_prose_without_removing_line_breaks(self) -> None:
        text = "The sentence continues\non the next physical line.\n"

        segments = segment_text_paragraph_aware(text)

        self.assertEqual(segments, [text])
        self.assertEqual("".join(segments), text)

    def test_splits_sentences_across_soft_wrapped_lines(self) -> None:
        text = "First sentence ends.\nSecond sentence continues\nonto this line.\n"

        segments = segment_text_paragraph_aware(text)

        self.assertEqual(
            segments,
            [
                "First sentence ends.\n",
                "Second sentence continues\nonto this line.\n",
            ],
        )
        self.assertEqual("".join(segments), text)

    def test_blank_line_keeps_paragraphs_separate(self) -> None:
        text = "First paragraph wraps\nacross a line.\n\nSecond paragraph.\n"

        segments = segment_text_paragraph_aware(text)

        self.assertEqual(
            segments,
            [
                "First paragraph wraps\nacross a line.\n\n",
                "Second paragraph.\n",
            ],
        )

    def test_experiment_headings_and_inline_section_labels_are_preserved(self) -> None:
        text = (
            "Experiment-1 \n\n"
            "AIM:- To learn about the network layer\n"
            "and its tools.\n"
            "Experiment 2: Compare two methods\n"
        )

        segments = segment_text_paragraph_aware(text)

        self.assertEqual(
            segments,
            [
                "Experiment-1 \n\n",
                "AIM:- To learn about the network layer\nand its tools.\n",
                "Experiment 2: Compare two methods\n",
            ],
        )
        self.assertEqual("".join(segments), text)

    def test_colon_terminated_topic_heading_stays_separate(self) -> None:
        text = "TCP Flow Control:\nThe explanation continues\non this line.\n"

        self.assertEqual(
            segment_text_paragraph_aware(text),
            [
                "TCP Flow Control:\n",
                "The explanation continues\non this line.\n",
            ],
        )

    def test_label_only_procedure_prefix_joins_its_instruction(self) -> None:
        text = "Procedure:\nRun the following commands.\n\n"

        self.assertEqual(
            segment_text_paragraph_aware(text),
            ["Procedure:\nRun the following commands.\n\n"],
        )

    def test_section_label_does_not_cross_a_blank_line(self) -> None:
        text = "Procedure:\n\nRun the following commands.\n"

        self.assertEqual(
            segment_text_paragraph_aware(text),
            ["Procedure:\n\n", "Run the following commands.\n"],
        )

    def test_numbered_and_lettered_steps_stay_separate(self) -> None:
        text = (
            "1) Troubleshooting hosts\n"
            "a) ping <address>\n"
            "The explanation continues on this line.\n"
            "b) use another command\n"
            "2. Record the result.\n"
        )

        segments = segment_text_paragraph_aware(text)

        self.assertEqual(
            segments,
            [
                "1) Troubleshooting hosts\n",
                "a) ping <address>\nThe explanation continues on this line.\n",
                "b) use another command\n",
                "2. Record the result.\n",
            ],
        )
        self.assertEqual("".join(segments), text)

    def test_indented_code_lines_remain_in_one_code_block(self) -> None:
        text = (
            "Program:\n"
            "    print('first. second')\n"
            "    value = 2\n"
            "Result:\n"
            "The program finished.\n"
        )

        segments = segment_text_paragraph_aware(text)

        self.assertEqual(
            segments,
            [
                "Program:\n    print('first. second')\n    value = 2\n",
                "Result:\nThe program finished.\n",
            ],
        )
        self.assertEqual("".join(segments), text)

    def test_inline_command_in_prose_is_not_classified_as_a_code_line(self) -> None:
        text = "Run sudo apt-get install ifplugd on the terminal.\n"

        self.assertEqual(segment_text_paragraph_aware(text), [text])

    def test_page_number_is_preserved_as_a_hard_boundary(self) -> None:
        first_page = "The packet is individually addressed and"
        second_page = "4\nrouted to the destination.\n"
        text = first_page + "\n" + second_page

        segments = segment_text_paragraph_aware(
            text,
            page_break_offsets=(len(first_page),),
        )

        self.assertEqual(
            segments,
            [
                "The packet is individually addressed and\n",
                "4\n",
                "routed to the destination.\n",
            ],
        )
        self.assertEqual("".join(segments), text)

    def test_clear_page_continuation_joins_when_no_number_intervenes(self) -> None:
        first_page = "The packet is addressed and"
        second_page = "routed to the destination.\n"
        text = first_page + "\n" + second_page

        self.assertEqual(
            segment_text_paragraph_aware(
                text,
                page_break_offsets=(len(first_page),),
            ),
            [text],
        )

    def test_completed_sentence_does_not_join_to_new_page(self) -> None:
        first_page = "The first sentence is complete."
        second_page = "A new page begins here.\n"
        text = first_page + "\n" + second_page

        self.assertEqual(
            segment_text_paragraph_aware(
                text,
                page_break_offsets=(len(first_page),),
            ),
            ["The first sentence is complete.\n", "A new page begins here.\n"],
        )

    def test_abbreviations_decimals_and_closing_quotes(self) -> None:
        text = 'Dr. Rao measured 3.14 ms. She wrote "done." Next step.\n'

        self.assertEqual(
            segment_text_paragraph_aware(text),
            [
                "Dr. Rao measured 3.14 ms. ",
                'She wrote "done." ',
                "Next step.\n",
            ],
        )

    def test_leading_trailing_whitespace_and_indentation_are_preserved(self) -> None:
        text = "  First sentence.\n    Second sentence.  \n"

        segments = segment_text_paragraph_aware(text)

        self.assertEqual("".join(segments), text)
        self.assertEqual(
            segments,
            ["  First sentence.\n    ", "Second sentence.  \n"],
        )

    def test_empty_and_whitespace_only_input(self) -> None:
        self.assertEqual(segment_text_paragraph_aware(""), [])
        self.assertEqual(segment_text_paragraph_aware(" \n\t"), [" \n\t"])

    def test_rejects_invalid_page_break_offsets(self) -> None:
        with self.assertRaises(ValueError):
            segment_text_paragraph_aware("abc", (1,))
        with self.assertRaises(ValueError):
            segment_text_paragraph_aware("a\nb", (1, 1))
        with self.assertRaises(TypeError):
            segment_text_paragraph_aware("a\nb", (True,))


if __name__ == "__main__":
    unittest.main()
