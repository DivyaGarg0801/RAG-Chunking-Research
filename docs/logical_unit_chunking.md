# Logical-Unit-Based Chunking

## Purpose

Logical-unit chunking treats an individual laboratory experiment as the main
unit of meaning. It keeps an experiment's heading, aim, theory, requirements,
procedure, commands, code, output, observations, and result together whenever
the unit fits the configured character limit. It does not use embeddings and
does not make claims about retrieval quality.

The implementation is `src/chunking/logical.py`, with the public function
`logical_unit_chunk(text, max_chunk_size=1500, ...)`.

## How it differs from structure-based chunking

Structure-based chunking emits separate spans for headings and their
subsections. Logical-unit chunking independently scans experiment-heading
appearances and groups the source from one experiment boundary to the next.
Aim, Theory, Procedure, and similar headings do not by themselves end a
logical unit. If the experiment fits, these sections can remain in one chunk.
Only experiments that exceed the maximum are split.

## Experiment identification and hierarchy

The detector recognizes conservative line-level forms such as `Experiment-4`,
`Experiment 4`, `Experiment: 4`, `Experiment 4: TCP Flow Control`,
`Experiment 1. Packet Delay`, and common extraction/OCR variants such as
`Experimet-10` and `Experlment5`. It requires a numeric experiment identifier,
rejects long or sentence-like candidates, and does not infer a boundary from
ordinary text such as “Experiment 1 was run.” The period-before-title form
also rejects common sentence openings and predicates to reduce false matches.

An explicit outer form, for example `Experiment-4`, establishes parent
context. A titled experiment heading within that context, such as
`Experiment 1: Point-to-Point link`, starts a child logical unit. If there is
text between the outer heading and its first titled child, the outer heading
and that intervening text form a separate `experiment_context` unit. The child
starts exactly at its own heading and records the outer heading as
`parent_heading`. This keeps the context from being mislabeled as child
content, while preserving the original order and every character. Subsequent
titled experiment headings start additional child units with the same parent.
A titled heading with no active outer context starts its own unit.

IDs are assigned by appearance order (`logical-unit-0001`, and so on), not by
experiment number. Thus repeated `Experiment 1` headings remain separate
units. Text before the first recognized experiment is a preamble unit.
Unrecognized headings and content between recognized experiment boundaries
are preserved in the current unit instead of being guessed into a new one.
If there are no recognized experiment headings, the document is represented
as one unclassified unit.

## Splitting oversized units

The preview uses a maximum of 1,500 characters per physical chunk. When a
logical unit exceeds that maximum, the splitter chooses the latest fitting
natural boundary from subsection, paragraph, sentence, line, and word
boundaries. When multiple boundary types occur at the same position, it
prefers them in that order. This avoids cutting at an early subsection when a
later meaningful boundary still fits, without trying to equalize chunk sizes.
A word boundary immediately after a recognized section label is not preferred
over the section start. If no natural boundary fits, it uses an exact character
boundary. There is no minimum chunk size and no overlap. Every resulting chunk
retains the same logical-unit ID and unit-level metadata; a split never
truncates or duplicates source text.

Subsection headings use the known label list plus conservative generic
colon-terminated headings. Generic candidates must look like concise title
text; command syntax, code markers, common instruction openings, and clauses
with finite verbs are rejected. This remains a heuristic: a short ambiguous
line may still look like either a title or a label.

The sentence fallback handles decimal points, a short list of common
abbreviations, and closing quotation marks. It is intentionally a small
rule-based boundary detector, not a full language parser.

## Metadata and preservation checks

Each chunk records:

- Sequential chunk index and character count.
- Logical-unit ID, title, type, and experiment number when recognized.
- Parent heading for a nested experiment.
- Subsection heading applicable at the beginning of the chunk, when found.
- Whether its logical unit was split and the reason for the chunk boundary.
- Source document and source page numbers when supplied by the PDF preview.

The chunker checks sequential indices, valid and consistent metadata for all
chunks of a unit, contiguous unit appearances, ordered unique page numbers,
the maximum size, character counts, and exact reconstruction:

```python
"".join(chunk.text for chunk in chunks) == original_text
```

The preview passes offsets for its inserted page separators so chunks can
retain one-based PDF page references. The newline inserted between pages is
attributed to the preceding page; text after it belongs to the next page.
Consequently, a chunk ending exactly at the page break references the
preceding page only. Empty intermediate pages can be represented by their
separator, but an empty final page has no extracted text to attribute. These
are page-span references, not PDF layout regions. The source PDF is read only.

## Real-manual preview

The preview used
`data/raw/computer_networks_lab_manual.pdf`, the existing page-by-page
PyMuPDF extraction and `clean_text()` preprocessing, and a newline between
cleaned pages. No embedding model was loaded. The maximum size was 1,500
characters.

| Measure | Result |
|---|---:|
| Extracted pages | 87 |
| Pages with no extractable text | 0 |
| Cleaned source characters | 78,531 |
| Logical units | 39 (29 experiments, 9 experiment contexts, 1 preamble) |
| Physical chunks | 70 |
| Units split across chunks | 21 |
| Units kept in one chunk | 18 |
| Average chunk size | 1,121.87 characters |
| Minimum / maximum chunk size | 3 / 1,500 characters |
| Exact reconstruction | Verified |

Nine outer experiment headings had titled children. Each outer heading and
its content before the first child became an `experiment_context` unit; the
29 child experiment appearances retain their outer heading as parent context.
Repeated experiment numbers under different outer headings remain separate
units with unique IDs. The detected child experiment number counts were
1: 10, 2: 9, 3: 7, and 4: 3. All chunks remained within the 1,500-character
limit.

These are structural measurements only. They do not show whether this method
improves retrieval. The preview's `summary.json` and `chunk_metadata.json`
contain detailed statistics and per-chunk source text in the ignored
`experiments/results/logical_preview_refined_v2/` directory for this corrected
run; the earlier `logical_preview/` outputs were not regenerated. The smallest
chunk is a page-number-only span at a logical-unit boundary; the 1,500-character
limit is kept without introducing an arbitrary minimum.

## Tests

`tests/test_logical.py` covers complete and incomplete experiments, nested and
repeated-number headings, pre-child context attribution, procedures, commands
and code, preamble and inter-experiment content, missing or inconsistent
headings, oversized units and fallback splits, whitespace, exact reconstruction,
IDs, page provenance at empty and exact boundaries, invalid metadata, and
preview overwrite/partial-write safeguards. The unit tests use synthetic text
and mocked PDF extraction; they do not depend on embedding weights.

Run the full suite from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Reproducing the preview

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.preview_logical
```

The script writes `chunks.txt`, `summary.json`, and `chunk_metadata.json` to
`experiments/results/logical_preview/`. It refuses to run if that output
directory already exists. To use a different fresh destination without
changing the default command:

```powershell
.\.venv\Scripts\python.exe -m scripts.preview_logical --output-directory experiments\results\logical_preview_another_run
```

The final corrected run used
`experiments/results/logical_preview_refined_v2/`. The existing ignore rules
exclude generated experiment outputs from Git.

## Limitations

- Heading recognition is based on extracted text lines, not PDF coordinates,
  font styles, or OCR confidence. Unusual headings may be missed, and a
  heading-like line may occasionally be misclassified.
- A missing or malformed experiment heading cannot reliably identify the
  intended unit; the content stays with its current unit or in the preamble.
- The outer/child distinction is a formatting heuristic. Under the selected
  policy, pre-child material remains in an outer `experiment_context` unit;
  it is not inferred to belong to the first child. Manuals with a different
  hierarchy may need a separately reviewed detector.
- Subsection and sentence splitting use explicit rules. Abbreviations outside
  the small configured list, unusual punctuation, and code without line breaks
  may require word or character fallback.
- A large experiment necessarily becomes multiple physical chunks. Shared IDs
  preserve their relationship, but each chunk alone may not contain the full
  experiment.
- Page references identify extracted page spans, not PDF layout regions.
- Chunk count and length do not establish precision, recall, MRR, nDCG, or
  answer quality. Those require the shared query set and relevance judgements
  planned for later evaluation.
