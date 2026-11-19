
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


RUN_ID = "20261115T113239215880Z"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parents[1]

CHUNK_FILE = (
    BASE_DIR / "experiments" / "results"
    / "chunk_statistics" / RUN_ID
    / "fixed_size_chunks.json"
)

WINDOW_FILE = (
    BASE_DIR / "experiments" / "results"
    / "window_embeddings" / RUN_ID
    / "fixed_size_windows.jsonl"
)


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)
    model.eval()

    tokenizer = model.tokenizer

    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)["chunks"]

    with open(WINDOW_FILE, "r", encoding="utf-8") as file:
        windows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    # Select chunks represented by exactly one complete window.
    window_groups = {}

    for window in windows:
        parent = window["parent_chunk_index"]
        window_groups.setdefault(parent, []).append(window)

    test_texts = []
    stored_embeddings = []
    tested_chunks = []

    for position, chunk in enumerate(chunks, start=1):

        parent = chunk.get("chunk_index", position)

        chunk_windows = window_groups.get(parent, [])

        if len(chunk_windows) != 1:
            continue

        window = chunk_windows[0]

        if (
            window["window_start_token"] != 0
            or window["window_end_token"]
            != window["window_token_count"]
        ):
            continue

        if (
            window["model_input_token_count"]
            > model.max_seq_length
        ):
            continue

        test_texts.append(chunk["text"])
        stored_embeddings.append(window["embedding"])
        tested_chunks.append(parent)

    if not test_texts:
        raise RuntimeError("No eligible chunks found for comparison.")

    print(f"\nEligible chunks: {len(test_texts)}")

    # Generate embeddings using the normal SentenceTransformer API.
    print("\nGenerating reference embeddings using model.encode()...")

    reference_embeddings = model.encode(
        test_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    stored_embeddings = np.asarray(
        stored_embeddings,
        dtype=np.float32
    )

    # Compare both embedding sets.
    differences = np.abs(
        reference_embeddings - stored_embeddings
    )

    max_difference = float(np.max(differences))

    cosine_similarities = []

    for reference, stored in zip(
        reference_embeddings,
        stored_embeddings
    ):

        similarity = np.dot(reference, stored) / (
            np.linalg.norm(reference)
            * np.linalg.norm(stored)
        )

        cosine_similarities.append(float(similarity))

    min_cosine = min(cosine_similarities)
    mean_cosine = float(np.mean(cosine_similarities))

    passed = np.allclose(
        reference_embeddings,
        stored_embeddings,
        rtol=1e-5,
        atol=1e-6
    )

    print("\n========== EQUIVALENCE AUDIT ==========")
    print(f"Chunks compared: {len(test_texts)}")
    print(f"Maximum absolute difference: {max_difference:.10f}")
    print(f"Minimum cosine similarity: {min_cosine:.10f}")
    print(f"Mean cosine similarity: {mean_cosine:.10f}")
    print(f"Embedding dimensions: {stored_embeddings.shape[1]}")
    print(f"Numerical equivalence: {'PASSED' if passed else 'FAILED'}")

    print(
        "\nOverall result:",
        "PASSED" if passed else "FAILED"
    )


if __name__ == "__main__":
    main()
