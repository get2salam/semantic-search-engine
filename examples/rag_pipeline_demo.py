"""Tiny dependency-free RAG pipeline demo.

Run with:
    python examples/rag_pipeline_demo.py
"""

from rag_attribution import assign_citation_labels, format_source_bibliography
from rag_chunking import chunk_text
from rag_context import RetrievedPassage, build_context_window
from rag_prompting import build_rag_prompt
from rag_query import build_retrieval_plan

DOCUMENTS = {
    "retrieval.md": "RAG retrieves relevant context before generation to improve grounding.",
    "citations.md": "Grounded AI answers should cite the passages used for factual claims.",
}


def main() -> None:
    chunks = []
    for source, text in DOCUMENTS.items():
        chunks.extend(chunk_text(text, max_words=18, overlap_words=0, source_id=source))

    plan = build_retrieval_plan("How should RAG answers stay grounded for AI users?")
    passages = [
        RetrievedPassage(chunk.text, chunk.chunk_id, score=1.0, metadata={"title": chunk.chunk_id})
        for chunk in chunks
    ]
    context = build_context_window(passages, max_chars=1200, include_scores=True)
    prompt = build_rag_prompt(plan.original_query, context)
    bibliography = format_source_bibliography(
        assign_citation_labels([chunk.chunk_id for chunk in chunks])
    )

    print(prompt)
    print("\nSources:\n" + bibliography)


if __name__ == "__main__":
    main()
