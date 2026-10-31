# Project Status

## Current milestone

Milestone 6: Recursive chunking.

## Completed

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
- `scripts/preview_fixed_size.py` - runs extraction, per-page cleaning, and fixed-size chunking for the selected manual.
- `experiments/results/fixed_size_preview/chunks.txt` - ordered chunk text with headings and character counts.
- `experiments/results/fixed_size_preview/summary.json` - run configuration, page information, chunk statistics, and derived overlap statistics.
- `experiments/results/fixed_size_preview/chunk_metadata.json` - ordered chunk text and character counts.
- `docs/PROJECT_STATUS.md` - this progress record.

## Implementation and testing

- PDF ingestion, conservative text preprocessing, fixed-size chunking, and recursive chunking are implemented.
- Tests were run with `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- All 40 tests passed: four PDF-loader tests, eight text-cleaner tests, 13 fixed-size chunker tests, and 15 recursive chunker tests.
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
- Clean each extracted page separately and join page texts in order with one newline before chunking. Chunks may cross page boundaries; page-level provenance for individual chunks is not available.
- The preview script refuses to overwrite existing output files.
- Keep source documents and results out of the repository unless their inclusion is deliberately reviewed.

## Next recommended step

Review the recursive chunker and its synthetic structural comparison before proceeding to structure-based chunking.
