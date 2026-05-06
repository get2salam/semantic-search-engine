# RAG Playbook

This playbook shows how to turn the semantic search engine into a small,
repeatable retrieval-augmented generation pipeline without adding a hosted LLM
or private data dependency.

## 1. Prepare documents

- Keep one stable `source_id` per original document.
- Attach metadata early: collection, topic, date, author, and permissions.
- Use `chunk_text()` for plain text and `chunk_markdown_sections()` when
  headings matter.

## 2. Retrieve broadly

- Generate deterministic query variants with `generate_query_variants()`.
- Run dense search for semantic matches.
- Optionally run lexical search for exact terminology.
- Fuse ranked lists with Reciprocal Rank Fusion when scores are not comparable.

## 3. Build grounded context

- Convert hits into `RetrievedPassage` objects.
- Use `build_context_window()` to enforce prompt budgets.
- Preserve `[S1]` labels so every factual claim can cite a source.

## 4. Prompt safely

- Use `build_rag_prompt()` for a consistent instruction block.
- Require citations after factual claims.
- Abstain when retrieval does not supply enough context.

## 5. Evaluate continuously

- Track `precision_at_k`, `recall_at_k`, and MRR for retrieval quality.
- Run citation guardrails against generated answers.
- Keep fixtures small and deterministic so CI catches regressions quickly.
