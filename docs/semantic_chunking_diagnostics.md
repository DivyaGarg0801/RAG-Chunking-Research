# Semantic Chunking: Real-Document Diagnostic Analysis

## Scope and reproducibility

This diagnostic uses `computer_networks_lab_manual.pdf` with the existing
page extraction and cleaner, joined in page order with a newline. The cleaned input
contains 78,531 characters across
87 extracted pages. The original
semantic preview files were read for verification and were not modified.

The script loaded `sentence-transformers/all-MiniLM-L6-v2` from the local cache in offline mode;
it did not request model weights from the network. The model revision resolved
from the loaded configuration was `unavailable`. Runtime
versions: `{"PyMuPDF": "2.0", "numpy": "2.5.4", "python": "3.13.14", "sentence-transformers": "5.2.0", "torch": "2.15.0", "transformers": "5.22.1"}`.

The current segmenter produced 1,833 original sentence-like segments.
After token-limit preparation there were 1,833 embedding
inputs. One embedding pass took 13.8135 seconds. The same
embedding matrix was reused for all five threshold runs; the diagnostic confirms
the saved preview's 0.5 result has identical chunk text and boundary labels.
There are 1,832 adjacent embedding pairs for the similarity distribution.

## Adjacent cosine similarity distribution

| Statistic | Cosine similarity |
|---|---:|
| Minimum | -0.1448 |
| Maximum | 1.0000 |
| Mean | 0.2534 |
| Median | 0.2148 |
| 10th percentile | 0.0269 |
| 25th percentile | 0.0985 |
| 50th percentile | 0.2148 |
| 75th percentile | 0.3695 |
| 90th percentile | 0.5392 |

| Threshold | Adjacent pairs below threshold | Percent below |
|---:|---:|---:|
| 0.3 | 1190 | 64.96% |
| 0.4 | 1439 | 78.55% |
| 0.5 | 1598 | 87.23% |
| 0.6 | 1707 | 93.18% |
| 0.7 | 1772 | 96.72% |

The similarity boundary condition is strict (`similarity < threshold`). Thus the
pair counts describe all adjacent embedding pairs; in the chunker, only pairs
between different original sentence-like segments can create a semantic boundary.
Here, the original segment and embedding input counts are
1,833 and 1,833, respectively.

## Threshold sensitivity

Maximum character size is held at 1,000, with batch size 32
and device `cpu`. The model and cached embeddings are identical across rows.
“Cached chunk time” measures the same chunking/validation path using the saved
embedding matrix; it excludes the one-time 13.8135-second
embedding inference and is not end-to-end run time.

| Threshold | Chunks | Average chars | Min | Max | Semantic boundaries | Size boundaries | Cached chunk time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 1191 | 65.94 | 2 | 656 | 1190 | 0 | 0.2554 |
| 0.4 | 1440 | 54.54 | 2 | 416 | 1439 | 0 | 0.2299 |
| 0.5 | 1599 | 49.11 | 2 | 332 | 1598 | 0 | 0.2401 |
| 0.6 | 1708 | 45.98 | 2 | 220 | 1707 | 0 | 0.2114 |
| 0.7 | 1773 | 44.29 | 2 | 188 | 1772 | 0 | 0.2823 |

At 0.3, 1190 boundaries are semantic; at the current
0.5 threshold there are 1598; at 0.7 there are
1772. A higher threshold is expected to classify more
adjacent pairs as below-threshold; this sensitivity table is a structural
diagnostic only, not a basis for selecting a preferred threshold or a retrieval
quality result.

## Qualitative fragmentation and causes

Examples below are taken directly from the saved preview output:

- **Experiment heading and aim:** Chunk 123 is `Experiment-1`, chunk 124 contains
  the Aim ending in “available in the Linux”, and chunk 125 is only `OS.`. The
  physical line break after “Linux” becomes separate segment input; the semantic
  boundary decision then makes that continuation a five-character chunk.
- **Procedure explanation split across physical lines:** chunks 134–136 divide
  “The ping command is usually used as a” / “simple way to verify...” / “computer
  or network device.” The extracted text contains line breaks at these positions;
  line-oriented segmentation supplies separate inputs and the semantic scores
  decide to split them.
- **Separated headings and content:** chunks 129 and 130 contain `Procedure:` and
  `Run the following commands...`; the headings are separate physical lines and
  the threshold places boundaries between those inputs.
- **Very short chunks and page numbers:** chunk 1 is `1`, chunk 11 is `2`, and
  chunk 141 is `7`. These are page-number text extracted from the document layout;
  semantic boundaries isolate them from the surrounding text.
- **Sentence endings and explanatory continuation:** chunks 462–464 split the
  DropTail/RED discussion at “In contrast to the more complex” / “algorithms like
  RED and WRED...”. The segmenter treated the source line break as an input
  boundary, and the semantic decision produced an incomplete phrase chunk.

These are interacting causes: PDF extraction exposes layout line breaks, the
rule-based segmenter treats physical lines conservatively as separate inputs,
and similarity decisions determine whether those neighboring inputs remain
together. The text is not lost or changed: the preview reconstructs all
78,531 cleaned source characters.

### Extraction artifacts observed before chunking

The extracted page text itself contains examples such as
`Thisistocertifythatthisisabonafiderecord...` (merged words) and replacement
characters in `�networks�` and redacted/unrecognized glyph sequences. A URL is
also extracted in the form `available athttp://...`. These are already present
before segmentation and are not introduced by semantic boundary decisions.
The targeted scan did not confirm a specific separated-letter figure label in
this manual's extracted text; figure labels such as `Figure.1.` do appear.

## What the distribution indicates

At the current 0.5 threshold, 1598 of
1832 adjacent pairs (87.23%)
are below threshold, matching the 1598 semantic
boundaries in the saved preview. This explains the high boundary count: most
adjacent embeddings are classified as dissimilar under the current threshold.
The average chunk is only 49.11 characters, and the
minimum is 2 characters.

The threshold changes chunk grouping, not the underlying embedding space or
retrieval performance. No threshold is selected here. The five thresholds should
be compared using judged retrieval queries and the same retrieval setup when
that evaluation milestone is ready.

## Possible improvements and trade-offs (not implemented)

1. **More reliable sentence segmentation:** a local sentence-boundary model or
   library may reduce splits caused by PDF line wrapping, but adds dependency,
   runtime and reproducibility considerations and can still mishandle technical
   abbreviations.
2. **Keep headings with following content:** classify short headings and attach
   them to the next content segment. This is structure-aware behavior and may
   improve usability, but changes the method from purely adjacent embedding
   decisions and must be applied consistently and documented.
3. **Avoid tiny standalone chunks:** report or flag short chunks for inspection
   before adding a minimum size. A minimum-size constraint or merge policy can
   suppress useful boundaries, blur genuinely distinct topics, and bias results;
   it needs a documented rationale and validation rather than an arbitrary
   character cutoff.
4. **Merge incomplete fragments:** a rule that avoids boundaries inside an
   incomplete phrase could use punctuation/grammar signals, but simple
   punctuation heuristics are unreliable for headings, lists, and technical
   notation. Compare this as a separately specified variant.

## Recommended next experiment

First, inspect a stratified sample of segments and boundary decisions across the
five thresholds, including headings, numbered procedures, and very short
fragments. Then run the unchanged threshold variants against manually reviewed
queries and relevance judgements once retrieval evaluation is implemented.
Do not choose a threshold solely because it creates fewer chunks.
