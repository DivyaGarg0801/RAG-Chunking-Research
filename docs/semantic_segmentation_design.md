# Semantic Input Segmentation: Proposed Variant

## Status and scope

This design was reviewed and approved, then implemented as the separate,
non-default `paragraph_aware_v2` variant in
`src/chunking/semantic_segmentation.py`. The existing
`rule_based_line_and_sentence_v1`, semantic chunker, 0.5 similarity threshold,
and saved preview outputs remain unchanged. Actual same-document results are
recorded in `docs/semantic_segmentation_comparison.md`.

The comparison used the existing PDF extraction and cleaner, the same 87
cleaned pages, and the same page-order separator. The variant received
page-boundary information so that a page transition was not confused with an
ordinary line wrap.

## Evidence from the current input

The current segmenter produces 1,833 original segments from 78,531 characters.
Their mean length is 42.84 characters; the shortest is 2 characters and the
longest is 117. For a transparent short-fragment diagnostic, 360 segments
(19.6%) have at most 10 characters and 550 (30.0%) have at most 20. These are
descriptive cutoffs, not proposed minimum segment sizes. Concatenating the
segments reconstructs the cleaned input exactly.

Examples from the extracted manual:

- On page 6, the Aim ends one physical line with “the Linux” and continues on
  the next with “OS.” The current segmenter places these in separate segments.
- On page 6, the description of `ping` continues across physical lines:
  “The ping command is usually used as a” / “simple way to verify...” /
  “computer or network device.” Each line is currently segmented separately.
- `Procedure:` and “Run the following commands on your Terminal window.” are
  separate lines. Numbered and lettered procedure items follow.
- Pages 41–42 include a likely cross-page sentence continuation: page 41 ends
  with “...is individually addressed and”, followed on page 42 by an extracted
  page-number line (`4`) and then “routed. Multiple packets...”.
- The manual includes commands both on their own lines and inline in procedure
  prose. Some extracted command/source text is already merged or malformed,
  for example `gcc -o server_ntcp_server_n.c`; segmentation must not attempt to
  repair that extraction artifact.

The actual text therefore contains soft wraps, but also meaningful paragraph,
heading, list, page, figure, and command boundaries. Removing every newline
would erase distinctions that the segmenter should preserve.

## Proposed rules for review

1. **Keep source text immutable.** Treat segmentation as choosing boundaries
   between source spans. Preserve each original space and newline in exactly
   one segment, in order; assert that concatenating all output segments equals
   the original cleaned input.
2. **Use paragraph breaks as hard boundaries.** A blank line separates
   paragraphs. Do not join across it. Keep the blank-line characters attached
   to an adjacent segment rather than deleting or normalizing them.
3. **Treat ordinary physical line endings within a paragraph as soft wraps.**
   When adjacent nonblank lines are prose and neither begins a structural item,
   sentence detection may continue across the line break. Keep the newline in
   the resulting source span. Split at recognized sentence endings across
   lines, preserving the exact intervening whitespace with one neighboring
   segment.
4. **Preserve experiment headings and section labels.** Recognize common
   variants such as `Experiment-3`, `Experiment 3`, `Aim:`, `AIM:-`,
   `Theory:`, `Procedure:`, `Algorithm:`, `Program:`, `Output:`, and `Result:`.
   Also preserve short colon-terminated topic headings such as
   `UDP Socket Programming:`. A standalone experiment or topic heading remains
   its own segment and is a hard boundary. A section label with inline text
   stays attached to that text; a label-only line such as `Procedure:` is
   prefixed to the next nonblank instruction paragraph, without crossing a
   blank line or another heading. The next section heading always starts a new
   segment.
5. **Keep list items distinct.** Numbered, lettered, and bullet markers begin
   separate items. A wrapped continuation belongs to its current item until a
   blank line, another item marker, heading, or code block. Do not merge
   neighboring items just because their text is short.
6. **Keep code-like blocks intact.** Use conservative signals such as a
   contiguous indentation pattern or clear source-code syntax to avoid
   sentence-splitting code lines. Preserve indentation and line endings. Do not
   classify every line containing a shell command as code: commands embedded
   in explanatory prose remain part of that prose. Ambiguous lines stay
   unchanged and are reported for manual inspection.
7. **Handle page transitions using page-aware input.** A page break alone is
   not a paragraph boundary. Join across it only when the surrounding text is
   a clear continuation and no extracted page-number line intervenes. Keep the
   page-separator newline and any extracted page-number line as source text; do
   not silently drop either. A page-number line is its own segment and a hard
   boundary: joining around it would either mix the number into prose or omit
   source text. Headings, blank paragraphs, or a completed sentence at the
   boundary also prevent a cross-page join.
8. **Keep sentence rules small and explicit.** Reuse the current decimal and
   closing-quote handling and a modest abbreviation list. Split only where
   punctuation provides a reasonable sentence boundary. Do not add a large
   domain-specific dictionary or claim perfect linguistic sentence detection.
9. **Do not impose a minimum segment length.** Report short outputs for
   inspection. A later merge policy would be a separate variant requiring its
   own rationale and evaluation.

## Trade-offs and limitations

- Treating nonblank line breaks as soft wraps should join broken prose, but can
  join separate paragraphs when the PDF extractor omitted their blank line.
  Blank-line and structural-marker rules reduce but cannot eliminate that risk.
- Heading recognition must allow capitalization and punctuation variants.
  Overly broad patterns could mistake ordinary prose, numbered objectives, or
  figure captions for headings; uncertain lines should remain unclassified.
- Keeping a standalone experiment heading separate preserves the structure but
  can still leave a short heading segment. Attaching a section label to its
  first content paragraph avoids an isolated label, but changes the unit
  supplied to the embedding model and must be applied consistently.
- Procedure markers can be inconsistent, and the manual contains both
  explanatory prose and commands in procedure sections. A rule-based detector
  can misclassify either; exact reconstruction does not prove correct
  structural classification.
- PDF extraction has already merged some words and command text. This
  segmentation variant will not correct extraction errors.
- Page-aware boundaries require preserving the individual cleaned page texts
  or their offsets during the diagnostic. The current combined preview does
  not retain per-chunk page provenance. An extracted page number in the middle
  of a continuing sentence remains a deliberate boundary in this variant.

## Comparison performed

The variant was compared with `rule_based_line_and_sentence_v1` on the same
extracted and cleaned manual. The comparison reports segment count,
mean/minimum/maximum length, descriptive 10- and 20-character counts, exact
reconstruction, and representative wrapped prose, heading, procedure, command,
and page-transition examples. Results and limitations are in
`docs/semantic_segmentation_comparison.md`.

The initial segmentation-only comparison did not run embeddings. A subsequent
controlled semantic chunking run is now complete using both segment lists
through the same `semantic_chunk()` implementation. The validated
caller-supplied-segment API preserves existing default behavior; results are
recorded in `docs/semantic_segmentation_comparison.md`. The threshold remains
0.5, and segment or chunk counts are not evidence of retrieval quality.
