# Semantic Segmentation and Chunking: Real-Document Comparison

## Objective and controlled setup

This experiment compares two input segmentation variants within the existing
semantic chunking strategy. It does not add another chunking strategy.

Both pipelines used the same `computer_networks_lab_manual.pdf`, page-by-page
PDF extraction, cleaner, cleaned page order, and page separator. The resulting
input was 78,531 characters across 87 pages. Both used the same loaded
`sentence-transformers/all-MiniLM-L6-v2` model object, cosine similarity,
threshold 0.5, maximum chunk size 1,000 characters, batch size 32, and CPU.
Model loading ran in Hugging Face offline mode. The embeddings were generated
separately for the two different segment lists, as required; no embedding
matrix was reused across them.

The original pipeline used `rule_based_line_and_sentence_v1`. The comparison
pipeline passed `paragraph_aware_v2` segments into the same semantic chunker.
The semantic chunker validates that supplied segments are nonempty strings
whose concatenation exactly equals the source. It then uses the existing
token-limit preparation, batched embedding, cosine calculation, threshold
logic, and final character-size enforcement.

The saved original semantic preview was used as a baseline check. The original
variant reproduced its chunk text and boundary labels exactly. SHA-256 checks
before and after the run confirmed that the original preview files were
unchanged. New chunk outputs were written separately under the ignored
`experiments/results/semantic_segmentation_comparison/` directory.

## Quantitative comparison

| Measure | `rule_based_line_and_sentence_v1` | `paragraph_aware_v2` |
|---|---:|---:|
| Input segment count | 1,833 | 996 |
| Embedding input count | 1,833 | 996 |
| Final chunk count | 1,599 | 713 |
| Average chunk length | 49.11 chars | 110.14 chars |
| Minimum chunk length | 2 chars | 2 chars |
| Maximum chunk length | 332 chars | 857 chars |
| Chunks <= 10 characters | 336 | 101 |
| Chunks <= 20 characters | 508 | 140 |
| Chunks <= 50 characters | 960 | 299 |
| Semantic boundaries | 1,598 | 711 |
| Size boundaries | 0 | 1 |
| Exact source reconstruction | Yes | Yes |
| All chunks <= 1,000 characters | Yes | Yes |

Both methods produced the same total chunk-text length as the input:
78,531 characters. Neither segmenter required token-safe subdivision on this
run; embedding-input counts equal their respective original segment counts.
The paragraph-aware output's largest chunk was 857 characters, below the
configured 1,000-character limit. The one size boundary reflects a split at
the configured character limit, even though neither resulting piece exceeds
the limit.

### Timing

| Timing measure | `rule_based_line_and_sentence_v1` | `paragraph_aware_v2` |
|---|---:|---:|
| Segmentation | 0.0293 s | 0.0340 s |
| Embedding inference (`model.encode`) | 8.7122 s | 8.3905 s |
| Non-embedding semantic processing | 0.2902 s | 0.0945 s |
| Total per-variant processing | 9.0317 s | 8.5191 s |

“Non-embedding semantic processing” is the semantic call duration minus the
time spent inside `model.encode`; it includes token-safe preparation,
normalization, similarity calculation, and chunk assembly. Per-variant total
is segmentation plus the complete semantic chunk call. It excludes shared
source extraction/cleaning (0.3768 s), one-time local model loading
(10.0309 s), and one fixed-text model warm-up (0.0604 s). These shared setup
times are recorded separately in `comparison.json`. Timings are a single CPU
run and should not be interpreted as a general performance benchmark.

## Qualitative inspection

### Experiment heading and Aim

The experiment heading remains a short segment in both outputs. In the
original output, chunk 124 contains the Experiment 1 Aim only through
`...available in the Linux`, and the continuation `OS.` is separated.
Paragraph-aware chunk 51 includes the Aim through `Linux` and the wrapped
`OS.` continuation (101 characters). This is more complete as an Aim unit.

### Procedure and ping explanation

Original chunk 129 is only `Procedure:` (12 characters). Paragraph-aware
chunk 54 contains `Procedure:` and “Run the following commands on your
Terminal window.” (66 characters).

The original ping explanation has a chunk ending with
`The ping command is usually used as a` (39 characters), followed by its
continuation in another chunk. Paragraph-aware chunk 56 keeps the
`a) ping <address>` item and its wrapped explanatory paragraph together
(467 characters). This preserves the command context and explanation in one
chunk, without modifying their source text.

### UDP Socket Programming and page transition

Original chunk 906 contains the `UDP Socket Programming:` topic label and only
the beginning of its explanation (129 characters). Paragraph-aware chunk 424
includes the label and more of the associated explanation (240 characters).
The text continues across the next source page, where an extracted page-number
line `4` appears. Both output sets preserve that number as a separate chunk:
original chunks 908–910 and paragraph-aware chunks 424–426 place the text,
number, and `routed.` continuation in sequence. The new segmenter does not
silently remove the page number or merge it into the prose.

### Experiment 8 TCP flow control

In the original output, chunk 1152 is only
`Congestion Control. ... TCP Flow Control:` (41 characters). In the
paragraph-aware output, chunk 530 contains the Experiment 8 Aim and the
`TCP Flow Control:` heading (125 characters). This keeps the section label
associated with its Aim, though the heading and following content are still
subject to the same semantic similarity decisions as all other segments.

### Numbered steps and commands

The numbered `1) Troubleshooting network hosts` step remains intact in both
outputs. The `a) ping <address>` command and explanation are separate from
other list items in the paragraph-aware segmentation.

The standalone command `sudo apt-get install ifplugd` appears as a short
29-character chunk in the original output. In the paragraph-aware output it is
within a 230-character chunk that also contains the preceding
`ifplugstatus` description and installation instruction. The inline compile
command `gcc -o server tcp_server.c` remains in its numbered instruction in
both outputs (208 characters).

These are observed structural differences, not retrieval or answer-quality
results. The paragraph-aware variant generally retains more surrounding prose
and heading context, while it can also produce larger chunks and may combine
text that the original segmentation kept apart.

## Reproducibility and recorded configuration

The comparison record and per-variant chunk previews are in:

- `experiments/results/semantic_segmentation_comparison/comparison.json`
- `experiments/results/semantic_segmentation_comparison/rule_based_line_and_sentence_v1/`
- `experiments/results/semantic_segmentation_comparison/paragraph_aware_v2/`

Recorded runtime versions were Python 3.13.14, PyMuPDF 2.0,
sentence-transformers 5.2.0, Transformers 5.22.1, PyTorch 2.15.0,
NumPy 2.5.4, and Requests 2.32.5. The source PDF SHA-256 and original preview
file hashes are recorded in the comparison JSON.

Requested and resolved model revisions were both unavailable (`null`). The
model identifier was used, but the upstream revision is not pinned; exact
reproducibility across future model-cache changes is therefore not guaranteed.
The script requires local cached weights and explicitly sets Hugging Face
offline mode, so it fails rather than downloading a model if the cache is
missing.

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_semantic_chunking_segmenters
```

The script refuses to overwrite output files. To write a later run to a
different directory, pass a fresh path:

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_semantic_chunking_segmenters --output-directory experiments\results\semantic_segmentation_comparison_rerun
```

## Limitations and interpretation

- The experiment compares chunk structure only. It does not measure retrieval,
  precision, recall, MRR, nDCG, answer correctness, or user relevance.
- Fewer chunks and longer average chunk lengths are not evidence of better
  retrieval. The paragraph-aware variant should proceed to retrieval
  evaluation only as a candidate, using the same query set, relevance labels,
  embedding model, retrieval settings, and metrics as other methods.
- Its rules are heuristic. If the PDF extractor omits a blank paragraph break,
  adjacent prose may be joined. Page-wrap inference is conservative; the
  observed page-number line intentionally prevents joining across it.
- Heading and list markers vary across manuals. A short colon-terminated line
  may be treated as a heading even when it is ordinary prose.
- The real manual did not provide a clear multi-line source-code block for
  inspection. Code-like preservation is covered by synthetic tests but has not
  been confirmed against a substantial real code block in this PDF.
- Source extraction artifacts, such as merged words or malformed command
  text, are not corrected by segmentation.
- Results come from one local CPU run with an unpinned model revision. Timing
  variation, hardware effects, and possible model-cache changes limit exact
  comparison beyond this run.
