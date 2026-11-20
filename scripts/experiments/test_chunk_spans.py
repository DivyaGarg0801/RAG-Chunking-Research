import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.extraction_artifact import load_extraction_artifact


RUN_ID = "20261115T113239215880Z"

ARTIFACT_PATH = (
    ROOT / "data" / "processed" / "extracted" / "extracted_pages.json"
)

CHUNK_DIRECTORY = ROOT / "experiments" / "results" / "chunk_statistics" / RUN_ID

FILES = {
    "fixed_size": "fixed_size_chunks.json",
    "recursive": "recursive_chunks.json",
    "semantic": "semantic_chunks.json",
    "structure": "structure_chunks.json",
    "logical_unit": "logical_unit_chunks.json",
}


def main():
    document = load_extraction_artifact(ARTIFACT_PATH)

    for strategy, filename in FILES.items():
        path = CHUNK_DIRECTORY / filename
        data = json.loads(path.read_text(encoding="utf-8"))

        # Support either a list of chunks or a dictionary containing chunks.
        if isinstance(data, dict):
            chunks = data.get("chunks", data)
        else:
            chunks = data

        if not isinstance(chunks, list):
            raise ValueError(f"Unexpected chunk format in {filename}")

        texts = [chunk["text"] for chunk in chunks]

        if strategy == "fixed_size":
            overlap_lengths = [0] + [150] * (len(texts) - 1)
        elif strategy == "recursive":
            overlap_lengths = [
                chunk["overlap_length"] for chunk in chunks
            ]
        else:
            overlap_lengths = None

        spans = document.chunk_spans_for_chunks(
            texts,
            overlap_lengths,
        )

        print(
            f"PASS {strategy}: "
            f"{len(spans)} chunks, "
            f"{len(spans)} source spans"
        )

        if strategy == "logical_unit":
            for chunk, span in zip(chunks, spans):
                assert tuple(chunk["source_page_numbers"]) == (
                    span.source_page_numbers
                )

            print("  Logical page provenance: verified")


if __name__ == "__main__":
    main()
