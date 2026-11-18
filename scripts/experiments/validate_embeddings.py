
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUN_ID = "20261115T113239215880Z"
EXPECTED_DIMENSIONS = 384

BASE_DIR = Path(__file__).resolve().parents[2]

REQUIRED_METADATA = {
    "fixed_size": [],
    "recursive": ["overlap_length"],
    "semantic": ["character_count", "boundary_before"],
    "structure": ["heading", "section_path"],
    "logical_unit": ["logical_unit_id", "source_page_numbers"],
}


def validate(strategy):
    file_path = (
        BASE_DIR / "experiments" / "results"
        / "embeddings" / RUN_ID
        / f"{strategy}_embeddings.json"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data["chunks"]

    if not chunks:
        raise ValueError("Embedding file contains no chunks.")

    embeddings = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32
    )

    indices = [chunk["chunk_index"] for chunk in chunks]

    metadata_checks = {
        key: all(key in chunk for chunk in chunks)
        for key in REQUIRED_METADATA[strategy]
    }

    checks = {
        "Embedding dimensions match metadata": (
            embeddings.ndim == 2
            and embeddings.shape[1] == data["embedding_dimensions"]
        ),
        "Chunk count matches metadata": (
            len(chunks) == data["total_chunks"]
        ),
        "Expected embedding dimensions": (
            embeddings.ndim == 2
            and embeddings.shape[1] == EXPECTED_DIMENSIONS
        ),
        "All values finite": np.isfinite(embeddings).all(),
        "All chunk indices unique": len(indices) == len(set(indices)),
        "Text preserved": all("text" in chunk for chunk in chunks),
        "Strategy-specific metadata preserved": all(
            metadata_checks.values()
        ),
    }

    print(f"\nStrategy: {data['strategy']}")
    print("Total chunks:", len(chunks))
    print("Embedding matrix shape:", embeddings.shape)
    print("Model:", data.get("model_name"))

    print("\nValidation checks:")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if metadata_checks:
        print("\nMetadata checks:")
        for name, passed in metadata_checks.items():
            print(f"{'PASS' if passed else 'FAIL'}: {name}")

    success = all(checks.values())

    print("\nOverall result:", "PASSED" if success else "FAILED")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Validate embeddings for a chunking strategy."
    )
    parser.add_argument(
        "strategy",
        choices=REQUIRED_METADATA.keys()
    )

    args = parser.parse_args()

    try:
        success = validate(args.strategy)
        sys.exit(0 if success else 1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Validation error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
