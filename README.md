# Document Chunking Strategies for RAG

This project will experimentally compare document chunking strategies for retrieving information from Computer Science laboratory manuals. The goal is to evaluate the methods fairly and report measured results, including negative or inconclusive findings.

## Research question

Does preserving a domain-specific logical unit - an entire experiment in a CSE laboratory manual - improve retrieval performance compared with general-purpose chunking strategies?

## Planned chunking strategies

1. Fixed-size chunking
2. Recursive chunking
3. Semantic chunking
4. Structure-based chunking
5. Logical-unit-based chunking, which aims to preserve each experiment as a unit when it fits the configured size limit

## Research pipeline

The planned pipeline is to extract text and page information from source manuals, apply documented preprocessing, and create chunks with each strategy. The same embedding model, retrieval configuration, queries, relevance judgements, and evaluation metrics will be used across strategies wherever technically applicable. Raw results and experiment configurations will be saved separately from aggregated results so that comparisons can be reviewed and reproduced.

## Current implementation status

The project currently includes page-level PDF text extraction, conservative text
preprocessing, fixed-size and recursive chunking, a local sentence-embedding
semantic chunker, and a structure-aware chunking pipeline.

The fixed-size preview was run on the local laboratory manual. A semantic
preview script is available, and the structure-based preview is now implemented
for real-manual runs without changing the semantic or logical-unit milestone
work. Retrieval, evaluation queries, relevance judgements, and retrieval
results remain to be implemented. No retrieval-quality conclusions have been
made.

## Structure-based chunking

The structure-based chunker uses a conservative line-by-line heading detector
that recognises common lab-manual headings such as Experiment, Aim, Theory,
Procedure, Program, Output, and Result while avoiding ordinary paragraph text.
It preserves exact source text by concatenating generated chunks without
silently discarding or modifying characters. If a section exceeds the configured
maximum size, it is split at paragraph or line boundaries before falling back to
character-level splitting.

The sequence remains intentionally separate from logical-unit-based chunking:
structure-based chunking groups headings and subsections, while logical-unit
chunking will preserve an entire experiment as one unit in a later milestone.

### Real preview result

The structure preview was executed on the local laboratory manual with a maximum
chunk size of 1500 characters. It processed 87 pages and produced 80 chunks from
78,531 cleaned characters. The average chunk length was 981.64 characters, the
minimum length was 14, and the maximum was 1495. The chunker reconstructed the
source text exactly, and the preview outputs were saved under
`experiments/results/structure_preview/`.
