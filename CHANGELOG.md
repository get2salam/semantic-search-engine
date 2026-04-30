# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Retrieval Quality Gate** (`quality_gate.py`). A model-agnostic CI
  guardrail that compares a current `EvalReport` against a committed
  baseline JSON and flags regressions on MRR, MAP, NDCG@k, P@k and
  R@k using combined absolute / relative / floor thresholds. Surfaces
  per-query regressions and emits a PR-comment-friendly Markdown
  summary plus a structured JSON result.
- `search-cli quality-gate` subcommand with `--strict` preset,
  `--config`, `--update-baseline` bootstrap, and dual `--markdown` /
  `--output` writers. Honours the canonical CI exit-code contract:
  `0` pass, `1` regression, `2` usage / I/O error.
- `examples/quality_gate/` ships a small public-safe TSV dataset, a
  precomputed baseline (`all-MiniLM-L6-v2`), a sample threshold
  config, and a `regenerate_baseline.py` helper.
- 35 new unit tests covering threshold semantics, configuration,
  per-query detection, JSON round-trips, rendering and the CLI
  subcommand end-to-end.

## [1.2.0] - 2026-04-18

### Added

- **Structured JSON logging** (`logging_config.py`). One JSON object per
  line with a stable schema (`ts`, `level`, `logger`, `msg`,
  `request_id`, plus any `extra=` fields). Toggle via
  `SSE_LOG_JSON=false` for human-readable output.
- **Request-ID propagation.** The API accepts an upstream
  `X-Request-ID` header (or mints a UUID), stamps every response with
  it, and propagates the ID through `contextvars` so async log records
  stay correlated.
- **Prometheus `/metrics` endpoint.** Zero-dependency metrics registry
  exposing `sse_requests_total`, `sse_request_latency_seconds`
  (histogram), `sse_searches_total`, `sse_documents_indexed`, and
  `sse_rate_limited_total` in the standard text exposition format.
- **Token-bucket rate limiter** (`rate_limit.py`). Per client IP, opt-in
  via `SSE_RATE_LIMIT_ENABLED`. Rejections return HTTP 429 with a
  `Retry-After` header. `/health`, `/metrics`, and `/docs` are exempt.
- **Security headers middleware.** `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`.
  Enabled by default; toggle with `SSE_SECURITY_HEADERS_ENABLED`.
- **Command-line interface** (`cli.py`). Subcommands `index`, `search`,
  `evaluate`, `info`, `version`. Supports stdin batching, JSON output,
  and runs without downloading the model in unit tests via a stub.
- **BEIR / TREC TSV dataset loader** (`eval_data.py`).
  `load_beir_like()` ingests `corpus.tsv`, `queries.tsv`, `qrels.tsv`
  (TREC 4-column format) and produces `EvalQuery` objects ready for
  `RetrievalEvaluator`.

### Changed

- `app_version` bumped to `1.2.0`.
- `.env.example` now documents the rate-limit and security-header flags.
- README gains an "Observability & Hardening" section and a "Command-Line
  Interface" section.

### Tests

- 61 new tests added this release (rate limiter, JSON logging /
  contextvars, metrics registry, CLI, TSV loader, API security headers,
  API request-ID behaviour, `/metrics` integration).

## [1.1.0]

### Added

- Maximal Marginal Relevance (MMR) diversification on `.search()`.
- Search benchmarking suite (latency / throughput / quality).
- Embedding-space anomaly detection and drift monitoring.
- Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.
- Cross-encoder two-stage retrieval pipeline.
- Pure-NumPy K-means document clustering with cluster-aware search.
- NLP query preprocessing (normalizer, stop-word filter, expander).
- Lightweight MLOps experiment tracker with model versioning.
- Model fine-tuning pipeline and retrieval evaluation metrics.
- Production REST API with FastAPI, config management, and API tests.

## [1.0.0]

Initial public release.
