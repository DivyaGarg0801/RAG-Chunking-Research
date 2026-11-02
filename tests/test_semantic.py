"""Tests for sentence-embedding-based semantic chunking."""

import builtins
import re
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.chunking.semantic import (
    SemanticChunk,
    _fitting_prefix_length,
    load_sentence_model,
    segment_text,
    semantic_chunk,
)


class FakeTokenizer:
    """Count simple word and punctuation tokens and retain tokenizer options."""

    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, dict[str, object]]] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        self.encode_calls.append(
            (
                text,
                {
                    "add_special_tokens": add_special_tokens,
                    "truncation": truncation,
                },
            )
        )
        count = len(re.findall(r"\w+|[^\w\s]", text))
        if add_special_tokens:
            count += 2
        return list(range(count))


class NonMonotonicTokenizer:
    """Provide deterministic prefix counts where longer text can use fewer tokens."""

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.token_counts = {
            "abcdefgh": 8,
            "abcd": 5,
            "ab": 2,
            "abc": 1,
        }

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        self.encode_calls.append(text)
        if truncation:
            raise AssertionError("token counting must not enable truncation")
        token_count = self.token_counts.get(text, 9)
        if not add_special_tokens:
            token_count = max(0, token_count - 2)
        return list(range(token_count))


class FakeEmbeddingModel:
    """Return controlled vectors without downloading a real model."""

    def __init__(
        self,
        embeddings: list[list[float]] | np.ndarray,
        declared_dimension: int | None = None,
        max_seq_length: int = 512,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=float)
        if declared_dimension is not None:
            self.declared_dimension = declared_dimension
        elif self.embeddings.ndim == 2:
            self.declared_dimension = self.embeddings.shape[1]
        else:
            self.declared_dimension = None
        self.max_seq_length = max_seq_length
        self.tokenizer = FakeTokenizer()
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.encode_calls.append((sentences, kwargs))
        return self.embeddings

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.declared_dimension


class DynamicEmbeddingModel(FakeEmbeddingModel):
    """Produce one deterministic embedding for each supplied model input."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        for sentence in sentences:
            if (
                len(
                    self.tokenizer.encode(
                        sentence,
                        add_special_tokens=True,
                        truncation=False,
                    )
                )
                > self.max_seq_length
            ):
                raise AssertionError("an over-limit segment reached model.encode")
        self.encode_calls.append((sentences, kwargs))
        return np.tile(
            np.asarray([[1.0, 0.0]], dtype=float),
            (len(sentences), 1),
        )


class SemanticSegmentationTests(unittest.TestCase):
    """Check sentence-like segment boundaries and exact text retention."""

    def test_splits_normal_sentences_and_keeps_delimiter_whitespace(self) -> None:
        text = "First sentence. Second sentence? Third sentence!"

        segments = segment_text(text)

        self.assertEqual(
            segments,
            ["First sentence. ", "Second sentence? ", "Third sentence!"],
        )
        self.assertEqual("".join(segments), text)

    def test_preserves_paragraph_and_line_boundaries(self) -> None:
        text = "Aim:\nFirst sentence.\n\nTheory:\nSecond sentence."

        segments = segment_text(text)

        self.assertEqual("".join(segments), text)
        self.assertTrue(any("\n\n" in segment for segment in segments))

    def test_preserves_decimal_numbers_and_common_abbreviations(self) -> None:
        text = "Dr. Rao measured 3.14 ms. Next, see e.g. Figure 2. Done."

        segments = segment_text(text)

        self.assertEqual("".join(segments), text)
        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[0].startswith("Dr. Rao"))
        self.assertIn("3.14 ms.", segments[0])

    def test_keeps_numbered_procedure_lines_intact(self) -> None:
        text = "1. Open the capture. Check the ACK flag.\n2. Record the result.\n"

        segments = segment_text(text)

        self.assertEqual(segments, [text.splitlines(keepends=True)[0], text.splitlines(keepends=True)[1]])
        self.assertEqual("".join(segments), text)

    def test_splits_ordinary_indented_prose_and_preserves_indentation(self) -> None:
        text = "  First sentence. Second sentence.  "

        segments = segment_text(text)

        self.assertEqual(
            segments,
            ["  First sentence. ", "Second sentence.  "],
        )
        self.assertEqual("".join(segments), text)

    def test_splits_indented_prose_on_a_line_with_newline(self) -> None:
        text = "    First sentence. Second sentence.\n"

        segments = segment_text(text)

        self.assertEqual(
            segments,
            ["    First sentence. ", "Second sentence.\n"],
        )
        self.assertEqual("".join(segments), text)

    def test_preserves_indented_code_lines(self) -> None:
        text = "Program:\n    print('first. second')\nResult."

        segments = segment_text(text)

        self.assertEqual("".join(segments), text)
        self.assertIn("    print('first. second')\n", segments)

    def test_long_sentence_remains_a_segment_until_size_enforcement(self) -> None:
        text = "This is one very long sentence without an ending punctuation mark"

        self.assertEqual(segment_text(text), [text])

    def test_empty_text_has_no_segments_and_no_empty_segments(self) -> None:
        self.assertEqual(segment_text(""), [])
        self.assertNotIn("", segment_text("A sentence. Another sentence."))

    def test_preserves_leading_and_trailing_whitespace(self) -> None:
        text = "  First sentence. Second sentence.  "

        segments = segment_text(text)

        self.assertEqual("".join(segments), text)
        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[0].startswith("  "))
        self.assertTrue(segments[-1].endswith("  "))

    def test_segments_realistic_laboratory_manual_prose(self) -> None:
        text = (
            "Experiment 4: TCP Flow Control\n"
            "Aim: To observe the TCP window size.\n"
            "Fig. 4 shows the ACK sequence. The receiver advertises a 4 KB window.\n"
            "1. Start Wireshark. Capture the TCP packets.\n"
        )

        segments = segment_text(text)

        self.assertEqual("".join(segments), text)
        self.assertIn("Fig. 4 shows the ACK sequence. ", segments)
        self.assertIn("The receiver advertises a 4 KB window.\n", segments)
        self.assertIn("1. Start Wireshark. Capture the TCP packets.\n", segments)

    def test_sentence_ending_before_closing_quote_is_preserved(self) -> None:
        text = 'He said "finished." Next step.'

        segments = segment_text(text)

        self.assertEqual(segments, ['He said "finished." ', "Next step."])
        self.assertEqual("".join(segments), text)


class SemanticChunkTests(unittest.TestCase):
    """Check validation, embeddings, boundaries, size limits, and metadata."""

    def test_empty_input_returns_no_chunks_without_loading_a_model(self) -> None:
        with patch("src.chunking.semantic.load_sentence_model") as load_model:
            result = semantic_chunk("", chunk_size=10)

        self.assertEqual(result.chunks, ())
        self.assertEqual(result.metadata.original_segment_count, 0)
        self.assertEqual(result.metadata.embedding_input_count, 0)
        load_model.assert_not_called()

    def test_whitespace_only_input_is_preserved_without_embedding(self) -> None:
        text = " \n\t  "

        with patch("src.chunking.semantic.load_sentence_model") as load_model:
            result = semantic_chunk(text, chunk_size=3)

        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)
        self.assertTrue(all(chunk.character_count <= 3 for chunk in result.chunks))
        self.assertEqual(result.metadata.original_segment_count, 1)
        self.assertEqual(result.metadata.embedding_input_count, 0)
        load_model.assert_not_called()

    def test_rejects_invalid_chunk_sizes(self) -> None:
        for chunk_size in (0, -1, 1.5, True):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    semantic_chunk("A sentence.", chunk_size=chunk_size)  # type: ignore[arg-type]

    def test_rejects_invalid_similarity_thresholds(self) -> None:
        for threshold in (-0.01, 1.01, float("nan"), float("inf"), "0.5"):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "similarity_threshold"):
                    semantic_chunk(
                        "A sentence.",
                        similarity_threshold=threshold,  # type: ignore[arg-type]
                    )

    def test_rejects_invalid_batch_sizes(self) -> None:
        for batch_size in (0, -1, 1.5, True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    semantic_chunk(
                        "A sentence.",
                        batch_size=batch_size,  # type: ignore[arg-type]
                    )

    def test_rejects_invalid_model_name_and_device(self) -> None:
        for kwargs in ({"model_name": ""}, {"device": " "}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    semantic_chunk("A sentence.", **kwargs)

    def test_rejects_model_without_sequence_limit_or_tokenizer(self) -> None:
        model = FakeEmbeddingModel([[1, 0]])
        del model.max_seq_length

        with self.assertRaisesRegex(ValueError, "max_seq_length"):
            semantic_chunk("One sentence.", model=model)

    def test_one_batched_encode_call_receives_configured_batch_size(self) -> None:
        model = FakeEmbeddingModel([[1, 0], [1, 0]])

        result = semantic_chunk(
            "First sentence. Second sentence.",
            batch_size=7,
            model=model,
        )

        self.assertEqual(len(model.encode_calls), 1)
        sentences, kwargs = model.encode_calls[0]
        self.assertEqual(len(sentences), 2)
        self.assertEqual(kwargs["batch_size"], 7)
        self.assertTrue(kwargs["convert_to_numpy"])
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(result.metadata.original_segment_count, 2)
        self.assertEqual(result.metadata.embedding_input_count, 2)

    def test_prefix_selection_directly_verifies_non_monotonic_token_counts(self) -> None:
        tokenizer = NonMonotonicTokenizer()
        text = "abcdefgh"
        total_token_count = len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )

        prefix_length = _fitting_prefix_length(
            text,
            tokenizer,
            maximum_tokens=4,
            total_tokens=total_token_count,
        )
        selected_prefix = text[:prefix_length]

        self.assertEqual(selected_prefix, "ab")
        self.assertLessEqual(
            len(
                tokenizer.encode(
                    selected_prefix,
                    add_special_tokens=True,
                    truncation=False,
                )
            ),
            4,
        )
        self.assertLessEqual(len(tokenizer.encode_calls), 4)
        self.assertLess(
            len(
                tokenizer.encode(
                    "abc",
                    add_special_tokens=True,
                    truncation=False,
                )
            ),
            len(
                tokenizer.encode(
                    "abcd",
                    add_special_tokens=True,
                    truncation=False,
                )
            ),
        )

    def test_model_loader_is_called_once_when_no_model_is_passed(self) -> None:
        model = FakeEmbeddingModel([[1, 0]])
        with patch(
            "src.chunking.semantic.load_sentence_model",
            return_value=model,
        ) as load_model:
            result = semantic_chunk("One sentence.", model_name="local/model")

        load_model.assert_called_once_with("local/model", "cpu")
        self.assertTrue(result.metadata.model_loaded_during_call)

    def test_model_loader_passes_an_optional_revision(self) -> None:
        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                self.model_name = model_name
                self.kwargs = kwargs

        fake_module = types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        )
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            model = load_sentence_model("local/model", revision="abc123")

        self.assertEqual(model.model_name, "local/model")
        self.assertEqual(
            model.kwargs,
            {"device": "cpu", "revision": "abc123"},
        )

    def test_model_loader_rejects_blank_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "revision"):
            load_sentence_model(revision="  ")

    def test_existing_model_instance_is_reused(self) -> None:
        model = FakeEmbeddingModel([[1, 0]])
        with patch("src.chunking.semantic.load_sentence_model") as load_model:
            result = semantic_chunk("One sentence.", model=model)

        load_model.assert_not_called()
        self.assertFalse(result.metadata.model_loaded_during_call)

    def test_loader_reports_missing_sentence_transformers_dependency(self) -> None:
        original_import = builtins.__import__

        def import_without_sentence_transformers(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                raise ImportError("dependency missing")
            return original_import(name, *args, **kwargs)

        with patch(
            "builtins.__import__",
            side_effect=import_without_sentence_transformers,
        ):
            with self.assertRaisesRegex(ImportError, "requires Sentence Transformers"):
                load_sentence_model()

    def test_loader_reports_model_initialization_failure(self) -> None:
        class FailingSentenceTransformer:
            def __init__(self, _model_name: str, *, device: str) -> None:
                raise OSError(f"{_model_name} files unavailable on {device}")

        fake_module = types.SimpleNamespace(
            SentenceTransformer=FailingSentenceTransformer
        )
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "Could not load embedding model"):
                load_sentence_model("invalid/model")

    def test_similarity_groups_related_segments(self) -> None:
        model = FakeEmbeddingModel([[1, 0], [1, 0]])

        result = semantic_chunk(
            "First sentence. Related sentence.",
            chunk_size=100,
            similarity_threshold=0.5,
            model=model,
        )

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].boundary_before, "document_start")

    def test_caller_supplied_segments_use_the_existing_chunking_pipeline(self) -> None:
        text = "First wrapped line\ncontinues here."
        model = FakeEmbeddingModel([[1, 0], [0, 1]])

        result = semantic_chunk(
            text,
            similarity_threshold=0.5,
            model=model,
            input_segments=["First wrapped line\n", "continues here."],
            segmentation_approach="test_segmenter",
        )

        self.assertEqual(len(model.encode_calls), 1)
        self.assertEqual(
            model.encode_calls[0][0],
            ["First wrapped line\n", "continues here."],
        )
        self.assertEqual(result.metadata.original_segment_count, 2)
        self.assertEqual(result.metadata.embedding_input_count, 2)
        self.assertEqual(
            result.metadata.sentence_segmentation_approach,
            "test_segmenter",
        )
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)

    def test_caller_supplied_segments_must_reconstruct_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly match"):
            semantic_chunk(
                "Expected source.",
                model=FakeEmbeddingModel([[1, 0]]),
                input_segments=["Different source."],
            )

    def test_default_segmenter_metadata_is_unchanged(self) -> None:
        result = semantic_chunk(
            "One sentence.",
            model=FakeEmbeddingModel([[1, 0]]),
        )

        self.assertEqual(
            result.metadata.sentence_segmentation_approach,
            "rule_based_line_and_sentence_v1",
        )

    def test_caller_supplied_segments_reject_empty_and_non_string_items(self) -> None:
        for segments, error in (([""], ValueError), (["text", 1], TypeError)):
            with self.subTest(segments=segments):
                with self.assertRaises(error):
                    semantic_chunk(
                        "text",
                        model=FakeEmbeddingModel([[1, 0]]),
                        input_segments=segments,
                    )

    def test_segmentation_label_requires_caller_supplied_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "only be set"):
            semantic_chunk(
                "One sentence.",
                model=FakeEmbeddingModel([[1, 0]]),
                segmentation_approach="custom",
            )

    def test_boundary_can_be_created_between_first_and_second_segments(self) -> None:
        model = FakeEmbeddingModel([[1, 0], [0, 1]])

        result = semantic_chunk(
            "Topic one. Topic two.",
            similarity_threshold=0.5,
            model=model,
        )

        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.chunks[1].boundary_before, "semantic")

    def test_multiple_topic_changes_create_multiple_semantic_boundaries(self) -> None:
        model = FakeEmbeddingModel(
            [[1, 0], [1, 0], [0, 1], [0, 1]]
        )

        result = semantic_chunk(
            "A one. A two. B one. B two.",
            similarity_threshold=0.8,
            model=model,
        )

        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.chunks[1].boundary_before, "semantic")

    def test_threshold_zero_and_one_are_valid_edge_values(self) -> None:
        same_embeddings = [[1, 0], [1, 0]]
        text = "First sentence. Second sentence."

        at_zero = semantic_chunk(
            text,
            similarity_threshold=0,
            model=FakeEmbeddingModel(same_embeddings),
        )
        at_one = semantic_chunk(
            text,
            similarity_threshold=1,
            model=FakeEmbeddingModel(same_embeddings),
        )

        self.assertEqual(len(at_zero.chunks), 1)
        self.assertEqual(len(at_one.chunks), 1)

    def test_similarity_below_zero_threshold_creates_boundary(self) -> None:
        model = FakeEmbeddingModel([[1, 0], [-1, 0]])

        result = semantic_chunk(
            "First sentence. Second sentence.",
            similarity_threshold=0,
            model=model,
        )

        self.assertEqual(len(result.chunks), 2)

    def test_wrong_embedding_count_is_rejected(self) -> None:
        model = FakeEmbeddingModel([[1, 0]])

        with self.assertRaisesRegex(ValueError, "one non-empty embedding"):
            semantic_chunk("First. Second.", model=model)

    def test_embedding_dimension_mismatch_is_rejected(self) -> None:
        model = FakeEmbeddingModel([[1, 0], [0, 1]], declared_dimension=3)

        with self.assertRaisesRegex(ValueError, "embedding dimension"):
            semantic_chunk("First. Second.", model=model)

    def test_non_matrix_embedding_output_is_rejected(self) -> None:
        model = FakeEmbeddingModel([1, 0])

        with self.assertRaisesRegex(ValueError, "one non-empty embedding"):
            semantic_chunk("One sentence.", model=model)

    def test_zero_and_non_finite_embeddings_are_rejected(self) -> None:
        for vectors in ([[0, 0]], [[float("nan"), 1]]):
            with self.subTest(vectors=vectors):
                with self.assertRaises(ValueError):
                    semantic_chunk("One sentence.", model=FakeEmbeddingModel(vectors))

    def test_chunk_size_splits_long_sentence_at_word_boundaries(self) -> None:
        text = "alpha beta gamma delta"

        result = semantic_chunk(
            text,
            chunk_size=10,
            model=FakeEmbeddingModel([[1, 0]]),
        )

        self.assertEqual(
            [chunk.text for chunk in result.chunks],
            ["alpha ", "beta ", "gamma ", "delta"],
        )
        self.assertTrue(all(chunk.character_count <= 10 for chunk in result.chunks))
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)
        self.assertEqual(result.chunks[1].boundary_before, "size")

    def test_oversized_numbered_line_is_split_before_embedding_without_truncation(self) -> None:
        text = "1. Configure router and set address then verify connection."
        model = DynamicEmbeddingModel([], max_seq_length=6)

        result = semantic_chunk(
            text,
            chunk_size=100,
            model=model,
        )

        encoded_segments = model.encode_calls[0][0]
        self.assertGreater(len(encoded_segments), 1)
        self.assertEqual("".join(encoded_segments), text)
        self.assertTrue(
            all(
                len(model.tokenizer.encode(
                    segment,
                    add_special_tokens=True,
                    truncation=False,
                ))
                <= model.max_seq_length
                for segment in encoded_segments
            )
        )
        self.assertTrue(
            all(
                options["truncation"] is False
                for _, options in model.tokenizer.encode_calls
            )
        )
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)
        self.assertEqual(result.metadata.original_segment_count, 1)
        self.assertGreater(result.metadata.embedding_input_count, 1)

    def test_oversized_indented_code_line_is_split_by_source_spans(self) -> None:
        text = "    print('alpha beta gamma delta epsilon')"
        model = DynamicEmbeddingModel([], max_seq_length=5)

        result = semantic_chunk(text, chunk_size=100, model=model)

        encoded_segments = model.encode_calls[0][0]
        self.assertGreater(len(encoded_segments), 1)
        self.assertEqual("".join(encoded_segments), text)
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)
        self.assertTrue(
            all(
                len(model.tokenizer.encode(
                    segment,
                    add_special_tokens=True,
                    truncation=False,
                ))
                <= model.max_seq_length
                for segment in encoded_segments
            )
        )

    def test_artificial_token_split_pieces_do_not_create_semantic_boundaries(self) -> None:
        text = "1. Configure router and set address then verify connection."
        model = DynamicEmbeddingModel([], max_seq_length=6)

        result = semantic_chunk(
            text,
            chunk_size=len(text) + 1,
            similarity_threshold=0.9,
            model=model,
        )

        self.assertGreater(result.metadata.embedding_input_count, 1)
        self.assertEqual(result.metadata.original_segment_count, 1)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].boundary_before, "document_start")
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)

    def test_final_character_limit_still_splits_pieces_from_one_source_segment(self) -> None:
        text = "1. Configure router and set address then verify connection."
        chunk_size = 18
        model = DynamicEmbeddingModel([], max_seq_length=6)

        result = semantic_chunk(
            text,
            chunk_size=chunk_size,
            similarity_threshold=0.9,
            model=model,
        )

        self.assertGreater(result.metadata.embedding_input_count, 1)
        self.assertEqual(result.metadata.original_segment_count, 1)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(
            all(chunk.character_count <= chunk_size for chunk in result.chunks)
        )
        self.assertTrue(
            all(chunk.boundary_before == "size" for chunk in result.chunks[1:])
        )
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)

    def test_character_fallback_handles_long_unbroken_segment(self) -> None:
        text = "x" * 19

        result = semantic_chunk(
            text,
            chunk_size=5,
            model=FakeEmbeddingModel([[1, 0]]),
        )

        self.assertEqual([len(chunk.text) for chunk in result.chunks], [5, 5, 5, 4])
        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)

    def test_chunk_exactly_at_maximum_size_is_allowed(self) -> None:
        text = "exact size"
        result = semantic_chunk(
            text,
            chunk_size=len(text),
            model=FakeEmbeddingModel([[1, 0]]),
        )

        self.assertEqual(result.chunks[0].character_count, len(text))

    def test_content_is_preserved_once_in_order_without_overlap(self) -> None:
        text = " Aim:\n1. Keep α and punctuation!\n\nResult: complete. "
        model = FakeEmbeddingModel([[1, 0], [0, 1], [1, 0]])

        result = semantic_chunk(
            text,
            chunk_size=15,
            similarity_threshold=0.8,
            model=model,
        )

        self.assertEqual("".join(chunk.text for chunk in result.chunks), text)
        self.assertTrue(all(chunk.text for chunk in result.chunks))
        self.assertTrue(all(chunk.character_count == len(chunk.text) for chunk in result.chunks))
        self.assertTrue(all(chunk.character_count <= 15 for chunk in result.chunks))
        self.assertEqual(
            [chunk.chunk_index for chunk in result.chunks],
            list(range(1, len(result.chunks) + 1)),
        )

    def test_result_includes_reproducibility_metadata(self) -> None:
        result = semantic_chunk(
            "One sentence.",
            chunk_size=80,
            similarity_threshold=0.63,
            batch_size=4,
            model_name="test/local-model",
            device="cpu",
            model=FakeEmbeddingModel([[1, 0]]),
        )

        self.assertEqual(result.metadata.strategy_name, "semantic")
        self.assertEqual(result.metadata.model_name, "test/local-model")
        self.assertEqual(result.metadata.similarity_method, "cosine_similarity")
        self.assertEqual(result.metadata.similarity_threshold, 0.63)
        self.assertEqual(result.metadata.maximum_chunk_size, 80)
        self.assertEqual(result.metadata.batch_size, 4)
        self.assertEqual(result.metadata.device, "cpu")
        self.assertEqual(result.metadata.original_segment_count, 1)
        self.assertEqual(result.metadata.embedding_input_count, 1)
        self.assertEqual(result.metadata.output_chunk_count, 1)
        self.assertGreaterEqual(result.metadata.execution_time_seconds, 0)
        self.assertTrue(result.metadata.sentence_segmentation_approach)

    def test_chunks_use_the_semantic_chunk_record_type(self) -> None:
        result = semantic_chunk(
            "One sentence.",
            model=FakeEmbeddingModel([[1, 0]]),
        )

        self.assertIsInstance(result.chunks[0], SemanticChunk)


if __name__ == "__main__":
    unittest.main()
