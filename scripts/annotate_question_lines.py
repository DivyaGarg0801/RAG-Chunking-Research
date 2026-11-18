
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = ROOT / "data/processed/evaluation/questions.json"
EXTRACTED_FILE = ROOT / "data/processed/extracted/extracted_pages.json"
OUTPUT_FILE = ROOT / "data/processed/evaluation/questions_with_line_numbers.json"


def normalize(text):
    """Normalize whitespace so PDF line wrapping does not prevent matching."""
    return re.sub(r"\s+", " ", text).strip()


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(EXTRACTED_FILE, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    pages = extracted["pages"]
    page_lookup = {
        page["page_number"]: page["text"].splitlines()
        for page in pages
    }

    matched = 0
    unmatched = []

    for question in dataset["questions"]:
        for evidence in question["evidence"]:
            page_number = evidence["pdf_page"]

            if page_number not in page_lookup:
                unmatched.append((question["question_id"], page_number, "Page not found"))
                continue

            lines = page_lookup[page_number]
            normalized_lines = [normalize(line) for line in lines]

            # Join lines while preserving their positions.
            joined_parts = []
            line_ranges = []
            position = 0

            for line_number, line in enumerate(normalized_lines, start=1):
                if not line:
                    continue

                if joined_parts:
                    position += 1  # space between lines

                start = position
                joined_parts.append(line)
                position += len(line)
                line_ranges.append((start, position, line_number))

            page_text = " ".join(joined_parts)
            target = normalize(evidence["evidence_text"])

            start_index = page_text.find(target)

            if start_index == -1:
                unmatched.append(
                    (question["question_id"], page_number, "Evidence text not matched")
                )
                continue

            end_index = start_index + len(target) - 1

            start_line = None
            end_line = None

            for start, end, line_number in line_ranges:
                if start <= start_index < end:
                    start_line = line_number

                if start <= end_index < end:
                    end_line = line_number

            evidence["line_start"] = start_line
            evidence["line_end"] = end_line
            matched += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Evidence passages matched: {matched}")
    print(f"Evidence passages not matched: {len(unmatched)}")

    for question_id, page, reason in unmatched:
        print(f"{question_id} | Page {page} | {reason}")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
