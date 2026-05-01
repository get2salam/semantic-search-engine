# RAG Readiness Audit — Example

This example demonstrates the `audit` subcommand on a deliberately *messy*
corpus, showing the kinds of issues the auditor surfaces before you spend
time and GPU on a doomed retrieval pipeline.

## What's in the dataset

`dataset/corpus.txt`

* 20 well-formed paragraphs about ML, retrieval and infrastructure.
* 2 identical `TODO:` placeholders left over from a prior import.
* 3 identical `n/a` rows (unhelpful but common in real exports).
* 1 lorem-ipsum paragraph nobody removed.
* 3 identical one-word `short` rows.

`dataset/queries.txt`

* 5 in-domain queries the corpus should answer well.
* 2 deliberately out-of-domain queries (GDPR, US tax filing) that the
  audit should flag as *uncovered*.

## Run the fast (stdlib) audit

No encoder is loaded — useful in CI or as a pre-flight check on a new
corpus dump.

```bash
python cli.py audit \
    --corpus examples/audit/dataset/corpus.txt \
    --no-embedding-stats
```

You should see the headline status come back as `NEEDS_ATTENTION` along
with action items pointing at the duplicate `TODO:` and `n/a` rows and
the `short` one-word docs.

## Run the full audit (loads the encoder)

This adds embedding-space health, near-duplicate clustering and a query
coverage probe.

```bash
python cli.py audit \
    --corpus examples/audit/dataset/corpus.txt \
    --queries examples/audit/dataset/queries.txt \
    --markdown audit_report.md \
    --output audit_report.json
```

* `audit_report.md` is PR-ready — paste it into a pull request to share
  the corpus health snapshot with reviewers.
* `audit_report.json` is the machine-readable form, suitable for storing
  as a CI artefact and diffing across runs.

## Suggested workflow

1. Run the audit on every new corpus snapshot before re-indexing.
2. Treat exit code `1` (needs attention) as a soft gate in CI — block
   merges until either the corpus is cleaned or the action items are
   explicitly acknowledged in the PR.
3. Persist `audit_report.json` as a build artefact so corpus health can
   be tracked over time alongside retrieval quality.
