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

Implementation, evaluation queries, and experimental results have not been created yet.
