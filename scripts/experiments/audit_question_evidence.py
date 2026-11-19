
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "evaluation"
    / "questions_with_line_numbers.json"
)

EXTRACTED_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "extracted"
    / "extracted_pages.json"
)


def normalize(text):
    """Normalize whitespace for evidence comparison."""
    return re.sub(r"\s+", " ", text).strip()


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    with open(EXTRACTED_PATH, encoding="utf-8") as f:
        extracted = json.load(f)

    pages = {
        page["page_number"]: page["text"].splitlines()
        for page in extracted["pages"]
    }

    total = 0
    passed = 0
    failed = 0

    for question in dataset["questions"]:
        for evidence in question["evidence"]:
            total += 1

            question_id = question["question_id"]
            page_number = evidence["pdf_page"]
            start = evidence["line_start"]
            end = evidence["line_end"]

            if start is None or end is None:
                print(f"FAIL {question_id}: Missing line numbers")
                failed += 1
                continue

            if page_number not in pages:
                print(f"FAIL {question_id}: Page {page_number} not found")
                failed += 1
                continue

            lines = pages[page_number]

            if start < 1 or end < start or end > len(lines):
                print(f"FAIL {question_id}: Invalid line range {start}-{end}")
                failed += 1
                continue

            selected_text = normalize(
                " ".join(lines[start - 1:end])
            )

            expected_text = normalize(evidence["evidence_text"])

            if expected_text in selected_text:
                print(
                    f"PASS {question_id}: "
                    f"Page {page_number}, lines {start}-{end}"
                )
                passed += 1
            else:
                print(
                    f"FAIL {question_id}: "
                    f"Evidence text does not match page {page_number}, "
                    f"lines {start}-{end}"
                )
                failed += 1

    print("\n========== AUDIT SUMMARY ==========")
    print(f"Total evidence passages: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("Overall result: PASSED")
    else:
        print("Overall result: FAILED")


if __name__ == "__main__":
    main()
