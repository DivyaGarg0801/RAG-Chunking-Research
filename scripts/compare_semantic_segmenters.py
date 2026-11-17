"""Compare existing and paragraph-aware segmentation on the local lab manual."""

import json
import statistics

from src.chunking.semantic import segment_text
from src.chunking.semantic_segmentation import segment_text_paragraph_aware
from src.ingestion.extraction_artifact import (
    DEFAULT_ARTIFACT_PATH,
    load_extraction_artifact,
)


EXTRACTION_ARTIFACT = DEFAULT_ARTIFACT_PATH

EXAMPLE_PHRASES = (
    "Experiment-1",
    "Aim: To gain familiarity",
    "The ping command is usually used as a",
    "Procedure:",
    "UDP Socket Programming:",
    "1) Troubleshooting",
    "sudo apt-get install ifplugd",
    "gcc -o server tcp_server.c",
    "individually addressed and",
    "routed.",
    "Multiple packets sent",
)


def _statistics(segments: list[str]) -> dict[str, int | float]:
    lengths = [len(segment) for segment in segments]
    return {
        "segment_count": len(segments),
        "average_characters": statistics.mean(lengths) if lengths else 0,
        "minimum_characters": min(lengths, default=0),
        "maximum_characters": max(lengths, default=0),
        "at_most_10_characters": sum(length <= 10 for length in lengths),
        "at_most_20_characters": sum(length <= 20 for length in lengths),
    }


def _examples(
    segments: list[str],
    phrases: tuple[str, ...],
) -> dict[str, list[dict[str, int | str]]]:
    found: dict[str, list[dict[str, int | str]]] = {}
    for phrase in phrases:
        matches = [
            {"segment": index, "text": segment}
            for index, segment in enumerate(segments, start=1)
            if phrase.casefold() in segment.casefold()
        ]
        found[phrase] = matches[:3]
    return found


def main() -> None:
    document = load_extraction_artifact(EXTRACTION_ARTIFACT)
    source_text = document.text
    current_segments = segment_text(source_text)
    paragraph_segments = segment_text_paragraph_aware(
        source_text,
        page_break_offsets=document.page_break_offsets,
    )

    if "".join(current_segments) != source_text:
        raise RuntimeError("The existing segmenter did not reconstruct its input.")
    if "".join(paragraph_segments) != source_text:
        raise RuntimeError("The paragraph-aware segmenter did not reconstruct its input.")

    result = {
        "source": {
            "filename": document.source_document,
            "page_count": len(document.pages),
            "cleaned_character_count": len(source_text),
            "page_separator": "\n",
        },
        "segmentation": {
            "rule_based_line_and_sentence_v1": _statistics(current_segments),
            "paragraph_aware_v2": _statistics(paragraph_segments),
        },
        "exact_reconstruction": {
            "rule_based_line_and_sentence_v1": (
                "".join(current_segments) == source_text
            ),
            "paragraph_aware_v2": (
                "".join(paragraph_segments) == source_text
            ),
        },
        "examples": {
            "rule_based_line_and_sentence_v1": _examples(
                current_segments,
                EXAMPLE_PHRASES,
            ),
            "paragraph_aware_v2": _examples(
                paragraph_segments,
                EXAMPLE_PHRASES,
            ),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
