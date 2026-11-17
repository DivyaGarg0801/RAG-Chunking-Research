"""Create the shared cleaned-text artifact used by chunking scripts."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.extraction_artifact import create_extraction_artifact  # noqa: E402


SOURCE_PDF = PROJECT_ROOT / "data" / "raw" / "computer_networks_lab_manual.pdf"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extracted"
    / "extracted_pages.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and clean the source PDF into a reusable JSON artifact."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and replace an existing extraction artifact.",
    )
    args = parser.parse_args()

    document = create_extraction_artifact(
        SOURCE_PDF,
        ARTIFACT_PATH,
        overwrite=args.force,
    )
    print(f"Source document: {document.source_document}")
    print(f"Pages saved: {len(document.pages)}")
    print(f"Pages with no cleaned text: {list(document.empty_pages)}")
    print(f"Cleaned characters: {len(document.text)}")
    print(f"Artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
