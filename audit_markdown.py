"""
Audit Report — Markdown Renderer
================================
Render a :class:`audit_report.RagReadinessReport` as a Markdown document
suitable for PR comments, dashboards or static-site snippets.

The layout mirrors the existing quality_gate Markdown so a CI commenter
can render either artefact with the same formatter.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from audit_report import RagReadinessReport

__all__ = [
    "render_markdown",
]


_STATUS_BADGE = {
    "ready": "✅ READY",
    "needs_attention": "⚠️ NEEDS ATTENTION",
}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _length_table(report: RagReadinessReport) -> list[str]:
    ls = report.length
    return [
        "### Length distribution",
        "",
        "| Metric        |   Chars |  Words |",
        "| ---           |     ---:|    ---:|",
        f"| Mean          | {ls.char_mean:7.1f} | {ls.word_mean:6.1f} |",
        f"| Median        | {ls.char_median:7.0f} | {ls.word_median:6.0f} |",
        f"| p90           | {ls.char_p90:7.0f} | {ls.word_p90:6.0f} |",
        f"| p99           | {ls.char_p99:7.0f} | {ls.word_p99:6.0f} |",
        "",
        f"_Empty: **{ls.empty_count}** · "
        f"very short (< {ls.very_short_threshold} chars): **{ls.very_short_count}** · "
        f"very long (> {ls.very_long_threshold} chars): **{ls.very_long_count}**_",
        "",
    ]


def _vocab_table(report: RagReadinessReport) -> list[str]:
    vs = report.vocabulary
    lines = [
        "### Vocabulary",
        "",
        "| Metric            |          Value |",
        "| ---               |            ---:|",
        f"| Total tokens      | {vs.total_tokens:>14,} |",
        f"| Unique tokens     | {vs.unique_tokens:>14,} |",
        f"| Type/token ratio  | {vs.type_token_ratio:>14.4f} |",
        f"| Hapax ratio       | {vs.hapax_ratio:>14.4f} |",
        "",
    ]
    if vs.top_tokens:
        top = ", ".join(f"`{tok}` ({count})" for tok, count in vs.top_tokens[:10])
        lines.extend([f"_Top tokens (≤10): {top}_", ""])
    return lines


def _exact_dup_table(report: RagReadinessReport) -> list[str]:
    ed = report.exact_duplicates
    return [
        "### Exact duplicates",
        "",
        f"- Documents: **{ed.n_documents}**",
        f"- Unique:    **{ed.n_unique}**",
        f"- Duplicate documents: **{ed.n_duplicate_documents}** ({_pct(ed.duplication_ratio)})",
        f"- Duplicate groups: **{ed.n_groups}**",
        "",
    ]


def _near_dup_table(report: RagReadinessReport) -> list[str]:
    nd = report.near_duplicates
    if nd is None:
        return []
    return [
        f"### Near duplicates (≥ {nd.threshold:.2f} cosine)",
        "",
        f"- Documents: **{nd.n_documents}**",
        f"- Duplicate documents: **{nd.n_duplicate_documents}** ({_pct(nd.duplication_ratio)})",
        f"- Duplicate clusters: **{nd.n_groups}**",
        "",
    ]


def _embedding_table(report: RagReadinessReport) -> list[str]:
    es = report.embedding
    if es is None:
        return []
    return [
        "### Embedding-space health",
        "",
        "| Metric                     |    Value |",
        "| ---                        |      ---:|",
        f"| n × dim                    | {es.n} × {es.dim} |",
        f"| Centroid norm              | {es.centroid_norm:>8.4f} |",
        f"| Mean pairwise similarity   | {es.mean_pairwise_similarity:>8.4f} |",
        f"| Median pairwise similarity | {es.median_pairwise_similarity:>8.4f} |",
        f"| Hubness skewness           | {es.hubness_skewness:>8.4f} |",
        f"| Effective rank             | {es.effective_rank:>8.2f} |",
        "",
    ]


def _coverage_table(report: RagReadinessReport) -> list[str]:
    cov = report.coverage
    if cov is None:
        return []
    lines = [
        "### Query coverage",
        "",
        "| Bucket     |  Count |   Share |",
        "| ---        |    ---:|     ---:|",
        f"| Confident  | {cov.n_confident:>6} | {_pct(cov.n_confident / cov.n_queries):>7} |",
        f"| Ambiguous  | {cov.n_ambiguous:>6} | {_pct(cov.n_ambiguous / cov.n_queries):>7} |",
        f"| Uncovered  | {cov.n_uncovered:>6} | {_pct(cov.n_uncovered / cov.n_queries):>7} |",
        "",
        f"_Coverage rate: **{_pct(cov.coverage_rate)}** · "
        f"Confidence rate: **{_pct(cov.confidence_rate)}**_",
        "",
    ]
    if cov.uncovered_examples:
        lines.append("**Worst uncovered queries:**")
        lines.append("")
        for v in cov.uncovered_examples[:5]:
            lines.append(f"- `{v.query}` — top-1 {v.top1:.3f}")
        lines.append("")
    if cov.ambiguous_examples:
        lines.append("**Most ambiguous queries:**")
        lines.append("")
        for v in cov.ambiguous_examples[:5]:
            lines.append(f"- `{v.query}` — clarity {v.clarity:.3f}, top-1 {v.top1:.3f}")
        lines.append("")
    return lines


def _notes_block(report: RagReadinessReport) -> list[str]:
    if not report.notes:
        return ["_No action items — corpus looks healthy._", ""]
    return ["### Action items", "", *(f"- {n}" for n in report.notes), ""]


def render_markdown(report: RagReadinessReport, *, title: str = "RAG Readiness Audit") -> str:
    """Render an audit report as a Markdown document.

    Args:
        report: The aggregated audit report to render.
        title: Heading for the document. Defaults to the audit's marketing
            name; override for project-specific dashboards.

    Returns:
        Markdown string with status badge, per-signal tables and the
        auto-derived action items.
    """
    badge = _STATUS_BADGE.get(report.headline_status(), report.headline_status().upper())
    lines: list[str] = [
        f"## {title} — **{badge}**",
        f"_generated_at `{report.generated_at}` · documents **{report.n_documents}**_",
        "",
    ]
    lines.extend(_length_table(report))
    lines.extend(_vocab_table(report))
    lines.extend(_exact_dup_table(report))
    lines.extend(_near_dup_table(report))
    lines.extend(_embedding_table(report))
    lines.extend(_coverage_table(report))
    lines.extend(_notes_block(report))
    return "\n".join(lines).rstrip() + "\n"
