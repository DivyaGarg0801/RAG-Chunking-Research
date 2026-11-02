# Project Status

## Current milestone

Structure-based chunking is implemented and validated on the
`feature/structure-based-chunking` branch. The milestone remains local and
uncommitted, as required.

## Completed

- Repaired `requirements.txt` by removing the unresolved merge markers and
  keeping the exact direct dependency pins required for the project baseline:
  `PyMuPDF==2.0`, `requests==2.32.5`, and `sentence-transformers==5.2.0`.
- Verified the active interpreter in the existing project `.venv` before any
  installation step; it is Python 3.12.3, not a recreated environment.
- Installed the pinned dependencies with
  `\.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt`
  and confirmed it completed successfully.
- Verified the runtime imports and versions using the project `.venv`:
  `PyMuPDF 2.0`, `requests 2.32.5`, and `sentence_transformers 5.2.0`.
- Ran the complete project test suite with
  `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and it
  completed successfully with exit code 0.
- Added the structure-aware chunking implementation in `src/chunking/structure.py`.
- Added synthetic tests in `tests/test_structure.py` covering heading detection,
  exact text preservation, multi-level structure, oversized sections, and
  unheaded fallback text.
- Added `scripts/preview_structure.py` to run the structure-based preview on the
  lab manual and save chunk metadata and summary outputs.
- Kept the repo on the local feature branch `feature/structure-based-chunking`,
  left all changes uncommitted, and did not stage, push, merge, or create a PR.
- Ran the structure preview and recorded actual manual statistics for the local
  laboratory manual.

## Structure-based chunking notes

- The implementation uses conservative heading recognition instead of treating
  every shared word such as "Aim" or "Result" as a heading.
- A heading must match a strict line-level pattern involving numbering,
  punctuation, line length, and context before it is treated as a section.
- Every generated chunk is meant to reconstruct the original source exactly by
  joining chunk texts without dropping whitespace or blank lines.
- The chunker preserves section hierarchy using metadata such as heading,
  heading level, and parent heading, but it does not merge whole experiments:
  that behaviour is reserved for the later logical-unit-based milestone.
- The real preview run on the 87-page manual produced 80 chunks from 78,531
  cleaned characters with a 1500-character maximum chunk size. The average chunk
  length was 981.64 characters, with a minimum of 14 and a maximum of 1495.
  Exact reconstruction was confirmed during the preview run.

- Confirmed that Python 3.13.14 is available.
- Created the project `.venv` virtual environment.
- Initialized Git in the project root.
- Created and checked the project-root `.gitignore`.
- Created the initial project directories and starter documentation/configuration files.
- Installed PyMuPDF 2.0 in the project `.venv` and verified its import and location.
- Implemented page-by-page PDF text extraction with source filenames, one-based page numbers, and explicit reporting of text-free pages.
- Added unit tests that generate small synthetic PDFs and check extraction metadata and error handling.
- Implemented conservative text cleaning that trims outer blank lines, reduces repeated internal blank lines, and preserves nonblank content lines.
- Added synthetic-string tests for whitespace, structure, code indentation, punctuation, and mathematical symbols.
- Implemented configurable character-based fixed-size chunking with validated size and overlap parameters.
- Added synthetic-string tests for chunk boundaries, overlap, parameter validation, content preservation, and complete source coverage.
- Implemented recursive chunking with paragraph, line, sentence, word, and character-level fallback separators.
- Added boundary-aware overlap: the earliest valid natural boundary within the requested range is selected, yielding the largest natural-boundary overlap no greater than the requested maximum. Each chunk reports its actual overlap character count.
- Added recursive chunker tests and a structural comparison against fixed-size chunking using the same synthetic input and configuration.
- Added tests for every adjacent overlap suffix, maximum-sized natural overlap, sentence punctuation edge cases, whitespace preservation, single-character input, and zero-overlap multi-chunk output.
- Investigated whether overlap could include characters absent from the previous chunk; no mismatch was found in 50,000 randomized synthetic cases, and a multi-chunk test now checks the invariant for every adjacent pair.
- Added and ran `scripts/preview_fixed_size.py` on `computer_networks_lab_manual.pdf` with a 1,000-character chunk size and 150-character overlap.
- Saved the generated chunks, summary statistics, and per-chunk metadata under `experiments/results/fixed_size_preview/`.
- Implemented local sentence-embedding-based semantic chunking with rule-based sentence-like segmentation, cosine similarity between adjacent segment embeddings, a configurable threshold, maximum character size, and CPU default.
- Added model-token-limit protection: sentence-like source spans are counted with the loaded model tokenizer without truncation and split at whitespace where possible, or at character boundaries otherwise, before encoding. These token limits are separate from the final character-based chunk-size limit; all split spans retain their exact source text.
- Recorded each token-safe embedding input's original sentence-like segment ID. Similarity boundaries are suppressed between artificial pieces of the same original segment, while the configured maximum character chunk size can still split them with size boundaries.
- Replaced the binary-search prefix assumption with a token-count estimate followed by direct, truncation-disabled verification of each candidate. The verified fitting prefix does not depend on token counts being monotonic as characters are added.
- Distinguished `original_segment_count` from `embedding_input_count` in semantic metadata and the preview summary. The latter includes additional pieces introduced by token-limit splitting.
- Improved segmentation so ordinary indented prose can be sentence-split while its indentation remains on the first sentence. Numbered procedure lines and clearly code-like indented lines remain intact before token-safe splitting.
- Added an optional model revision to the Sentence Transformers loader and semantic preview settings. The preview records the requested revision and the Transformers configuration commit hash when exposed by the loaded model; unresolved revisions are recorded as `null`.
- Added semantic chunk tests using deterministic fake embeddings; tests do not initialize Sentence Transformers or download model weights.
- Added a semantic preview script and synthetic integration tests. The script processes pages in order, avoids model loading for empty or whitespace-only text, preserves empty-page reporting, verifies reconstruction and size limits, records runtime versions and chunk statistics, refuses to overwrite output files, and removes files created by an incomplete output write.
- Ran the semantic preview on the 87-page Computer Networks laboratory manual. It produced 1,599 chunks at threshold 0.5 from 78,531 cleaned characters; the existing preview outputs were preserved during later diagnostics.
- Added `scripts/analyze_semantic_thresholds.py` and `docs/semantic_chunking_diagnostics.md`. The diagnostic ran offline, generated embeddings once, reused that matrix for thresholds 0.3–0.7, and verified the threshold-0.5 result against the saved preview.
- The diagnostic measured 1,832 adjacent similarities: mean 0.2534, median 0.2148, and 87.23% below the current 0.5 threshold. No threshold change was made.
- Investigated the current semantic input segmentation on the same 78,531-character, 87-page manual. The existing segmenter produces 1,833 segments (mean 42.84 characters; minimum 2; maximum 117); 360 segments are at most 10 characters and 550 are at most 20. Exact reconstruction is preserved.
- Confirmed wrapped prose on page 6, including the Experiment 1 Aim and ping explanation, as well as headings, list markers, blank paragraphs, page numbers, and inline commands that should not be indiscriminately joined. A possible page 41–42 continuation has an extracted page-number line between its text, so the variant preserves that number as a hard boundary.
- Implemented the separate, non-default `paragraph_aware_v2` in `src/chunking/semantic_segmentation.py`, with page-aware line grouping, heading/section handling, list-item grouping, conservative code preservation, and exact reconstruction checks. The existing segmenter and semantic chunker remain unchanged.
- Compared both segmenters on the same cleaned document using `scripts/compare_semantic_segmenters.py`. The new variant produced 996 segments (mean 78.85 characters; minimum 2; maximum 643), with 107 at most 10 characters and 151 at most 20. Both methods reconstructed all 78,531 characters exactly.
- Extended `semantic_chunk()` with an optional, validated caller-supplied segment list and segmenter label. Existing calls keep the original segmenter and behavior; custom segments must be nonempty strings that reconstruct the full source exactly.
- Ran two semantic pipelines offline, using the same cached model object and identical semantic settings. Embeddings were generated independently for the 1,833 original segments and 996 paragraph-aware segments.
- At threshold 0.5, the original segmentation produced 1,599 chunks; paragraph-aware produced 713. Both reconstructed the source exactly and stayed within 1,000 characters. Short-chunk counts (<=10 / <=20 / <=50 characters) were 336 / 508 / 960 for the original and 101 / 140 / 299 for paragraph-aware.
- The original preview's chunk text and boundary labels matched the rerun exactly. SHA-256 verification before and after the experiment confirmed its three files were unchanged. New chunk outputs were saved separately under `experiments/results/semantic_segmentation_comparison/`.
- Added `scripts/compare_semantic_chunking_segmenters.py` and updated `docs/semantic_segmentation_comparison.md` with the semantic results, timing definitions, examples, model revision status, and limitations. The original 0.5 threshold remains unchanged; no retrieval evaluation was performed.
- Added five deterministic tests for caller-supplied semantic segments, exact reconstruction validation, invalid segment values, and unchanged default metadata. Together with 17 paragraph-aware tests, these cover wrapped prose, headings, section labels, list/code blocks, punctuation, page transitions, invalid offsets, whitespace, and reconstruction.
- Added a narrow `.gitignore` rule for the new generated semantic comparison directory.
- Added a narrow ignore rule for generated semantic preview outputs.
- Added `sentence-transformers==5.2.0` and `requests==2.32.5` to the project requirements after checking Python 3.13 compatibility and verifying the runtime imports.

## Files created or modified for this milestone

- `src/ingestion/pdf_loader.py` - page-level PDF text extraction.
- `tests/test_pdf_loader.py` - tests for multiple pages, text-free pages, missing paths, and invalid PDF content.
- `requirements.txt` - records the verified direct dependency `PyMuPDF==2.0`.
- `src/preprocessing/text_cleaner.py` - conservative text-cleaning function.
- `tests/test_text_cleaner.py` - tests for text-cleaner behavior and preservation rules.
- `src/chunking/fixed_size.py` - configurable fixed-size character chunker.
- `tests/test_fixed_size.py` - fixed-size chunker tests.
- `src/chunking/recursive.py` - recursive separator-based chunker with overlap metadata.
- `tests/test_recursive.py` - recursive chunking tests and synthetic structural comparison.
- `src/chunking/semantic.py` - local sentence-embedding-based semantic chunker and metadata records.
- `tests/test_semantic.py` - semantic segmentation, token-limit splitting, fake-model, validation, size, preservation, and metadata tests.
- `scripts/preview_semantic.py` - reproducible local semantic preview pipeline; run on the manual.
- `scripts/compare_semantic_chunking_segmenters.py` - offline controlled semantic chunking comparison of both segmenters.
- `scripts/analyze_semantic_thresholds.py` - offline threshold-sensitivity diagnostic using one embedding pass.
- `docs/semantic_chunking_diagnostics.md` - real-document similarity, threshold, fragmentation, and next-step analysis.
- `src/chunking/semantic_segmentation.py` - separate page-aware paragraph segmentation variant.
- `tests/test_semantic_segmentation.py` - deterministic tests for the alternative segmenter.
- `scripts/compare_semantic_segmenters.py` - same-document segmentation comparison, with no model loading.
- `experiments/results/semantic_segmentation_comparison/` - ignored comparison JSON and per-variant chunk previews.
- `docs/semantic_segmentation_design.md` - reviewed rules and trade-offs for the alternative variant.
- `docs/semantic_segmentation_comparison.md` - measured real-document comparison and limitations.
- `tests/test_semantic_preview.py` - synthetic integration tests for output generation, revision metadata, whitespace-only input, overwrite/missing-source safeguards, and failed-write cleanup.
- `.gitignore` - excludes `experiments/results/semantic_preview/`.
- `requirements.txt` - pins Sentence Transformers and the Requests runtime dependency.
- `README.md` - reflects the implemented chunkers and outstanding research components.
- `scripts/preview_fixed_size.py` - runs extraction, per-page cleaning, and fixed-size chunking for the selected manual.
- `experiments/results/fixed_size_preview/chunks.txt` - ordered chunk text with headings and character counts.
- `experiments/results/fixed_size_preview/summary.json` - run configuration, page information, chunk statistics, and derived overlap statistics.
- `experiments/results/fixed_size_preview/chunk_metadata.json` - ordered chunk text and character counts.
- `docs/PROJECT_STATUS.md` - this progress record.

## Implementation and testing

- PDF ingestion, conservative text preprocessing, fixed-size, recursive, and semantic chunking are implemented.
- The complete suite was run with `.venv\Scripts\python.exe -m unittest discover -s tests -q`.
- All 114 tests passed: 40 existing PDF/preprocessing/fixed/recursive tests, 51 semantic chunker tests, six semantic preview integration tests, and 17 paragraph-aware segmentation tests.
- Sentence Transformers was installed in `.venv` at version 5.2.0 with CPU PyTorch 2.15.0, Transformers 5.22.1, NumPy 2.5.4, and Requests 2.32.5. Runtime imports were verified; the diagnostic loaded model weights from the existing local cache in offline mode, with no network download.
- An initial runtime import exposed that Sentence Transformers 5.2.0 did not declare `requests` in its installed dependency metadata although its import path uses it. The dependency is now explicitly pinned in `requirements.txt`.
- Semantic preview tests used a synthetic page and deterministic fake embeddings. They verified chunk text reconstruction, output statistics, reporting flags, missing-source handling, and refusal to overwrite existing output.
- `scripts/preview_semantic.py` was run on `computer_networks_lab_manual.pdf`; its saved summary reports 87 pages, 78,531 cleaned characters, and 1,599 chunks at threshold 0.5.
- The preview's `MODEL_REVISION` defaults to `None`, so it does not pin a repository revision. The offline diagnostic could not obtain a resolved commit hash from the loaded model configuration; the report records this as unavailable.
- The threshold diagnostic found that 1,598 of 1,832 adjacent pairs (87.23%) fall below 0.5. The tested threshold results and representative fragmentation causes are documented in `docs/semantic_chunking_diagnostics.md`; no default threshold change was made.
- The diagnostic confirmed that the cleaned text and segment counts match the saved preview and that the recalculated 0.5 chunk text and boundary labels are identical. Existing preview outputs were read only and were not modified.
- Semantic chunking uses a rule-based sentence-like segmenter, not a language-aware parser. It recognizes a small set of common abbreviations, decimals, closing punctuation, numbered procedure lines, and code-like indentation, but other abbreviations, unusual punctuation, and ambiguous code/prose formatting may still produce imperfect boundaries.
- Before encoding, each sentence-like span is token-counted using the loaded model tokenizer with truncation disabled. Oversized spans are sliced at whitespace where possible and otherwise at character boundaries; those exact slices are then passed to the encoder. The sequence-token ceiling prevents model truncation, while `maximum_chunk_size` independently limits final chunks by characters.
- Artificial token-safe pieces keep their original segment identity. The chunker does not use similarities between pieces of the same source segment to add semantic boundaries, because those boundaries were introduced only to respect the model input limit. Size-driven splits still apply independently.
- Token counts can change non-monotonically when a tokenizer re-tokenizes a longer character prefix. Prefix selection therefore uses a proportional estimate and verifies each candidate directly. It returns a fitting prefix, not necessarily the longest possible fitting prefix; in unusual tokenization cases this can create smaller embedding inputs than strictly necessary.
- Semantic similarity boundaries are heuristic and depend on the embedding model and threshold. No retrieval or answer-quality evaluation has been implemented.
- The synthetic structural comparison used the same 298-character input, 100-character maximum chunk size, and zero overlap for both methods. Fixed-size produced three chunks with two internal chunk ends outside paragraph or sentence boundaries; recursive produced four chunks with zero such internal ends.
- This comparison measures structural boundary alignment only and is not a retrieval-quality result.
- The preview script was run with `.venv\Scripts\python.exe -m scripts.preview_fixed_size`.
- Preview output verification confirmed that all 93 chunk records agree with their text, lengths, ordering, and summary statistics.
- The 87 extracted pages produced no pages with empty extracted text.
- The combined cleaned text was 78,531 characters; the 93 chunks averaged 992.81 characters, with a minimum of 331 and maximum of 1,000 characters.
- The chunks contain 92,331 characters in total. The derived repeated-character count was 13,800, calculated as total chunk characters minus combined cleaned text characters.
- The manual was processed for this exploratory preview; the generated statistics are not retrieval or quality evaluation results.
- No evaluation datasets or retrieval experiment results have been added.

## Qualitative analysis of the fixed-size baseline

The saved preview provides examples of structure being divided at character-based chunk boundaries:

- Chunk 42 ends with `Experiment-5 / Aim: To learn about Wi-fi`, while Chunk 43 begins with the continuation `ace 2 as well...`. The experiment heading and Aim are separated from the continuation of the title.
- Chunks 27 and 28 divide the DropTail and RED discussion at an awkward point. The boundary falls within the transition from the DropTail explanation to the RED explanation; the configured overlap repeats some text between the chunks.
- Chunks 67 and 68 split the Experiment 8 TCP flow-control procedure and question content, with procedure instructions continuing in the next chunk.

Fixed-size chunking uses character positions and does not understand document structure. Therefore it can split words, sentences, headings, and procedures. These observations describe the chunk boundaries only; retrieval evaluation has not been implemented, so they do not establish that fixed-size chunking has worse retrieval performance.

The preview also contains extracted-text formatting artifacts, including separated letters in figure labels and merged words. Those are extraction or source-text formatting issues, not errors introduced by the chunk-boundary algorithm. The examples above are qualitative observations, not measured retrieval results.

## Decisions

- Build the project incrementally, following the milestone order in `.github/copilot-instructions.md`.
- Use the already-installed PyMuPDF version, pinned as a direct dependency for reproducibility.
- Use Python's built-in `unittest` for these tests because `pytest` is not installed and no extra package was requested.
- Preserve every nonblank line exactly; normalize CRLF/CR line endings to LF, remove outer blank lines, and keep at most one blank line between content.
- Treat chunk size and overlap as character counts; advance each chunk start by `chunk_size - overlap` and stop when the final source character is covered.
- Recursive chunking tries paragraph breaks, line breaks, sentence endings, and word whitespace in order, then uses character splitting when no larger separator can keep a piece within the size limit. Adjacent pieces are greedily merged while they fit.
- Recursive overlap is a requested maximum. It chooses the earliest valid paragraph, sentence, or word boundary within the requested range, producing the largest natural-boundary overlap that does not exceed the maximum. If no such boundary exists, it uses exact character overlap. `RecursiveChunk.overlap_length` records the actual repeated character count.
- The overlap search begins no earlier than the previous chunk's start; tests verify that every recorded overlap exactly matches the previous chunk's suffix.
- Recursive boundary recognition uses simple regular expressions. The edge-case tests confirm text preservation for abbreviations, decimals, and closing quotation marks, but do not make sentence detection linguistically complete. For example, `Dr.` may be treated as a sentence ending; decimal punctuation is not split unless followed by whitespace, and punctuation before a closing quote is not recognized as a sentence boundary by the current sentence pattern.
- Semantic sentence segmentation is likewise heuristic and preserves input text exactly; it does not guarantee linguistically correct sentences. Size-driven splits prefer whitespace and fall back to character boundaries when needed.
- Semantic chunking embeds each sentence-like segment in batches, compares adjacent normalized vectors using cosine similarity, and places a semantic boundary when similarity is below the configured threshold. It adds no overlap; concatenating chunk text reconstructs the source exactly.
- The semantic preview configuration uses `sentence-transformers/all-MiniLM-L6-v2`, a 0.5 cosine-similarity threshold, a 1,000-character maximum, batch size 32, and CPU. These are exploratory settings, not research conclusions.
- Tests use fake embeddings and do not establish that the real model performs better or worse than another chunking strategy.
- Clean each extracted page separately and join page texts in order with one newline before chunking. Chunks may cross page boundaries; page-level provenance for individual chunks is not available.
- The preview script refuses to overwrite existing output files.
- Keep source documents and results out of the repository unless their inclusion is deliberately reviewed.

## Next recommended step

Review the diagnostic report, then plan the next experiment around manually inspected boundary samples and retrieval evaluation. Do not select a threshold solely because it creates fewer chunks. Structure-based and logical-unit-based chunking remain pending.
