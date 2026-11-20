import argparse
import json
import math
from pathlib import Path

from sentence_transformers import SentenceTransformer


RUN_ID = "20261115T113239215880Z"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parents[1]

CHUNK_DIR = (
    BASE_DIR / "experiments" / "results"
    / "chunk_statistics" / RUN_ID
)

WINDOW_DIR = (
    BASE_DIR / "experiments" / "results"
    / "window_embeddings" / RUN_ID
)

STRATEGIES = [
    "fixed_size",
    "recursive",
    "semantic",
    "structure",
    "logical_unit",
]

WINDOW_SIZE = 250
OVERLAP = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy",
        choices=STRATEGIES,
        help="Chunking strategy to audit"
    )
    args = parser.parse_args()

    strategy = args.strategy

    CHUNK_FILE = CHUNK_DIR / f"{strategy}_chunks.json"
    WINDOW_FILE = WINDOW_DIR / f"{strategy}_windows.jsonl"
    print("Loading tokenizer...")

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        original_chunks = json.load(file)["chunks"]

    with open(WINDOW_FILE, "r", encoding="utf-8") as file:
        windows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    grouped = {}

    for window in windows:
        parent = window["parent_chunk_index"]
        grouped.setdefault(parent, []).append(window)

    print("\nStarting exact token-ID coverage audit...\n")

    passed = 0
    failed = 0
    maximum_input_length = 0

    for position, chunk in enumerate(original_chunks, start=1):

        original_ids = tokenizer.encode(
            chunk["text"],
            add_special_tokens=False,
            truncation=False
        )

        parent = chunk.get("chunk_index", position)

        chunk_windows = sorted(
            grouped.get(parent, []),
            key=lambda item: item["window_index"]
        )

        checks = []

        if not chunk_windows:
            checks.append(False)

        expected_start = 0

        for index, window in enumerate(chunk_windows):

            start = window["window_start_token"]
            end = window["window_end_token"]

            content_ids = window["window_token_ids"]
            model_ids = window["model_input_token_ids"]

            # Check exact original token coverage.
            checks.append(start == expected_start)
            checks.append(content_ids == original_ids[start:end])
            checks.append(len(content_ids) == end - start)

            # Check window size.
            checks.append(len(content_ids) <= WINDOW_SIZE)

            # Check special tokens and exact model input.
            expected_model_ids = (
                [tokenizer.cls_token_id]
                + content_ids
                + [tokenizer.sep_token_id]
            )

            checks.append(model_ids == expected_model_ids)
            checks.append(
                len(model_ids) == window["model_input_token_count"]
            )
            checks.append(
                len(model_ids) <= model.max_seq_length
            )

            maximum_input_length = max(
                maximum_input_length,
                len(model_ids)
            )

            # Check recorded metadata.
            checks.append(
                window["window_token_count"] == len(content_ids)
            )

            # Check embedding dimensions and finite values.
            embedding = window["embedding"]

            checks.append(
                len(embedding) == model.get_sentence_embedding_dimension()
            )

            checks.append(
                all(math.isfinite(value) for value in embedding)
            )

            # Check next window starts with the required overlap.
            if index < len(chunk_windows) - 1:
                checks.append(
                    chunk_windows[index + 1]["window_start_token"]
                    == end - OVERLAP
                )

            if index < len(chunk_windows) - 1:
                expected_start = end - OVERLAP
            else:
                expected_start = end

        # Check that the final window reaches the end.
        if chunk_windows:
            checks.append(
                chunk_windows[-1]["window_end_token"]
                == len(original_ids)
            )

        if checks and all(checks):
            passed += 1
            print(
                f"PASS Chunk {position}: "
                f"{len(chunk_windows)} windows, "
                f"{len(original_ids)} original tokens"
            )
        else:
            failed += 1
            print(
                f"FAIL Chunk {position}: "
                f"one or more validation checks failed"
            )

    print("\n========== AUDIT SUMMARY ==========")
    print(f"Original chunks: {len(original_chunks)}")
    print(f"Generated windows: {len(windows)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Maximum model input length: {maximum_input_length}")
    print(f"Model sequence limit: {model.max_seq_length}")
    print(
        "Embedding dimensions:",
        model.get_sentence_embedding_dimension()
    )

    print(
        "\nOverall result:",
        "PASSED" if failed == 0 else "FAILED"
    )


if __name__ == "__main__":
    main()
