# RAG Evaluation Checklist

Use this checklist before presenting a RAG demo, portfolio project, or customer
prototype.

## Retrieval quality

- [ ] Are evaluation queries representative of real users?
- [ ] Does each query have at least one relevant source ID?
- [ ] Are precision@k, recall@k, and MRR tracked at a fixed `k`?
- [ ] Are dense and lexical retrieval compared before fusion?

## Context quality

- [ ] Are chunks small enough to avoid topic drift?
- [ ] Is overlap large enough to preserve sentence continuity?
- [ ] Are headings, titles, dates, and source IDs preserved in metadata?
- [ ] Does the context builder enforce a hard character/token budget?

## Answer quality

- [ ] Does the prompt require `[S1]`-style citations?
- [ ] Are unsupported claims detected before publishing an answer?
- [ ] Does the assistant abstain when context is missing?
- [ ] Are unknown citation labels rejected?

## Operations

- [ ] Can the RAG-specific tests run without downloading a model?
- [ ] Are public demos free from secrets and private data?
- [ ] Are benchmark results reproducible from committed fixtures?
