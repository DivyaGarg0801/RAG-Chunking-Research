# Structure-Based Chunking

The structure chunker segments laboratory-manual text at recognized heading
lines. Its public interface is `structure_chunk(text, max_chunk_size=None)` in
`src/chunking/structure.py`; each `StructureChunk` contains the exact source
text for that span and its heading, hierarchy, and occurrence metadata. The
preview uses a 1,500-character maximum.

## Heading and hierarchy rules

Heading recognition is deliberately line-based and conservative. It supports
common section labels, standalone configured topic headings, numbered heading
variants, and experiment-label OCR variants such as `Experimet`, `Experlment`,
and `Experment`, including spacing inside the label.

Experiment classification uses both syntax and active document context:

- The first recognized experiment heading starts the outer experiment.
- A colon, hyphen, en dash, or em dash between the label and number is an
  explicit outer-experiment marker.
- A bare experiment number with no trailing separator is an outer heading.
- A title-bearing heading with ordinary spacing, or one with a trailing
  separator after its number and no explicit outer prefix marker, is nested
  when an outer experiment is active.
- A new outer heading clears the previous nested experiment. Subsections under
  a nested experiment use that nearest experiment as parent; otherwise they
  use the outer experiment.

These are formatting heuristics rather than a full document-layout parser. An
unusually formatted heading may remain ambiguous and should be checked against
the source manual.

## Text preservation and size limits

Chunk text is not normalized by the structure chunker. Joining every returned
`chunk.text` must reproduce the input exactly, including whitespace-only
input, leading and trailing whitespace, line endings, and blank lines. Empty
input returns no chunks.

When a section is too large, splitting first attempts paragraph and line
boundaries, then splits at character boundaries if needed. No overlap is added.
Every non-empty chunk must respect the configured maximum. The preview
validates exact reconstruction, sequential indices, hierarchy relationships,
occurrence metadata, and the maximum size before saving outputs.

Heading occurrence IDs are assigned once per heading appearance and are
reused by all chunks split from that section. Separate appearances receive
different IDs, even when their normalized heading names are the same.

## Corrected laboratory-manual preview

The corrected preview was run on
`data/raw/computer_networks_lab_manual.pdf` and saved separately under
`experiments/results/structure_based_preview_v2_corrected/`. The prior
`experiments/results/structure_preview/` output was left untouched.

| Statistic | Corrected preview |
|---|---:|
| Extracted pages | 87 |
| Empty pages | 0 |
| Cleaned source characters | 78,531 |
| Chunks | 82 |
| Unique detected headings | 37 |
| Heading occurrences | 50 |
| Unheaded chunks | 4 |
| Average chunk size | 957.70 |
| Minimum chunk size | 14 |
| Maximum chunk size | 1,495 |
| Chunk counts by level (0 / 1 / 2 / 3) | 4 / 10 / 67 / 1 |
| Exact reconstruction | Verified |

Representative hierarchy includes outer `Experiment 2` with nested
`Experiment 1` as its child, and the nested experiment's sections beneath it.
The output's `summary.json` and `chunk_metadata.json` contain the full
statistics and representative text examples.

An initial in-session preview at
`experiments/results/structure_based_preview_v2/` was created before manual
inspection exposed a trailing-separator ambiguity. It was not overwritten or
deleted. Use the `v2_corrected` output above as the authoritative result.

## Limitations

- Heading detection uses extracted text lines, not PDF layout, fonts, or
  coordinates. OCR and inconsistent line breaks can still affect recognition.
- The chunker does not run OCR on scanned pages.
- The preview joins cleaned page text with a newline; chunks do not retain
  source page metadata.
- A title-bearing experiment heading with no distinguishing punctuation can be
  ambiguous without richer document context.
