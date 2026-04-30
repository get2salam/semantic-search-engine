# 🔍 Semantic Search Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Sentence Transformers](https://img.shields.io/badge/🤗-Sentence%20Transformers-yellow)](https://www.sbert.net/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/get2salam/semantic-search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/get2salam/semantic-search-engine/actions)

A lightweight, production-ready semantic search engine powered by state-of-the-art sentence embeddings. Find similar documents based on **meaning**, not just keywords.

Ships with a **REST API** (FastAPI), **Docker** support, and **CI/CD** pipeline — ready for production deployment.

## ⚡ Features

- 🔎 **Fast & Efficient** — FAISS-powered vector similarity search
- 🧭 **MMR Diversification** — Optional Maximal Marginal Relevance re-ranking to cut near-duplicate results
- 🤖 **State-of-the-Art Embeddings** — Uses `all-MiniLM-L6-v2` (384-dim, blazing fast)
- 🌐 **REST API** — Production-grade FastAPI with OpenAPI docs, validation, CORS
- 🎯 **Fine-Tuning Pipeline** — Domain-adaptive training with contrastive/triplet loss and k-fold CV
- 📊 **Retrieval Evaluation** — MRR, MAP, NDCG@k, Precision@k, Recall@k with multi-model benchmarking
- 🛡️ **Quality Gate** — CI regression guardrail that compares evals against a committed baseline (PR-comment Markdown output)
- 📂 **BEIR/TREC Loader** — Drop-in loader for corpus/queries/qrels TSV datasets
- 🐳 **Docker Ready** — Multi-stage build, non-root user, health checks
- 🔄 **CI/CD** — GitHub Actions: lint → test (matrix) → Docker build & verify
- 🧪 **Experiment Tracking** — Lightweight MLOps: log runs, compare models, track lineage (no external deps)
- 📈 **Observability** — JSON logs, X-Request-ID propagation, Prometheus `/metrics`
- 🛡️ **Production Hardening** — Token-bucket rate limiter + security headers (opt-in)
- 🧰 **CLI** — Scriptable `search-cli index | search | evaluate | info`
- 💾 **Persistent Storage** — Save and load indices to disk
- ⚙️ **12-Factor Config** — Environment-based configuration via pydantic-settings

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/get2salam/semantic-search-engine.git
cd semantic-search-engine
pip install -r requirements.txt
```

### Option 1: REST API

```bash
# Start the API server
make serve
# or
uvicorn api:app --host 0.0.0.0 --port 8000

# Interactive docs at http://localhost:8000/docs
```

### Option 2: Python Library

```python
from semantic_search import SemanticSearchEngine

engine = SemanticSearchEngine()

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language for data science",
    "Neural networks are inspired by biological neurons",
]
engine.add_documents(documents)

results = engine.search("AI and deep neural nets", top_k=3)
for doc, score in results:
    print(f"[{score:.3f}] {doc}")
```

### Option 3: Docker

```bash
# Build and run with docker compose
make serve-docker
# or
docker compose up -d

# Standalone
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

## 🌐 API Reference

### Endpoints

| Method   | Path              | Description                        |
|----------|-------------------|------------------------------------|
| `GET`    | `/health`         | Health check for load balancers    |
| `GET`    | `/stats`          | Index statistics and model info    |
| `GET`    | `/metrics`        | Prometheus scrape endpoint         |
| `POST`   | `/documents`      | Add documents to the index         |
| `GET`    | `/documents/count`| Document count                     |
| `DELETE` | `/documents`      | Clear the entire index             |
| `POST`   | `/search`         | Semantic search (JSON body)        |
| `GET`    | `/search?q=...`   | Semantic search (query params)     |
| `POST`   | `/search/batch`   | Batch search (multiple queries)    |

### Add Documents

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Machine learning is great", "Python is versatile"]}'
```

### Search

```bash
# POST
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 3}'

# GET (convenience)
curl "http://localhost:8000/search?q=artificial+intelligence&top_k=3"
```

**Response:**

```json
{
  "query": "artificial intelligence",
  "results": [
    {"document": "Machine learning is great", "score": 0.7842, "rank": 1},
    {"document": "Python is versatile", "score": 0.3210, "rank": 2}
  ],
  "total_documents": 2,
  "elapsed_ms": 4.72
}
```

### Batch Search

```bash
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["AI models", "web development"], "top_k": 3}'
```

## 🧭 Diversified Results (MMR)

Dense retrievers cluster near-duplicates in the top-k — five phrasings of the
same idea instead of five distinct ideas. The `mmr_lambda` parameter applies
[Maximal Marginal Relevance](https://dl.acm.org/doi/10.1145/290941.291025)
(Carbonell & Goldstein, 1998) to re-rank a larger candidate pool with an
explicit diversity term:

```python
# Pure relevance (default)
engine.search("climate change", top_k=5)

# Balanced relevance + diversity
engine.search("climate change", top_k=5, mmr_lambda=0.5)

# Max diversity — useful for "show me a spread of topics"
engine.search("climate change", top_k=5, mmr_lambda=0.1, mmr_candidate_k=50)
```

- `mmr_lambda=1.0` → identical to the default (pure relevance)
- `mmr_lambda=0.0` → ignores the query after the first pick (maximum diversity)
- `mmr_lambda=0.5` → balanced (recommended starting point)
- `mmr_candidate_k` controls the candidate pool size; larger pools give MMR
  more room to diversify at the cost of extra compute (defaults to
  `max(4·top_k, 25)`).

Returned similarity scores remain on the cosine scale — MMR only changes
*which* docs are returned, not how they are scored.

MMR is also exposed through the REST API:

```bash
# POST
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change", "top_k": 5, "mmr_lambda": 0.5}'

# GET
curl "http://localhost:8000/search?q=climate+change&top_k=5&mmr_lambda=0.5"
```

## 🧰 Command-Line Interface

A scriptable CLI is included for ad-hoc use and shell pipelines:

```bash
# Build a persistent index from a file (one doc per line, or JSONL)
python cli.py index --input docs.txt --output ./my-index

# Run a query
python cli.py search --index ./my-index --query "machine learning" --top-k 5

# Pipe queries (one per line)
echo -e "AI\nweb dev" | python cli.py search --index ./my-index --stdin --json | jq .

# Inspect a saved index
python cli.py info --index ./my-index

# Run a full BEIR/TREC-style evaluation
python cli.py evaluate --dataset ./datasets/nfcorpus --k 1,3,5,10
```

Subcommands: `index`, `search`, `evaluate`, `quality-gate`, `info`, `version`.

## 📂 Evaluation Datasets (BEIR / TREC format)

The built-in loader reads the three conventional TSV files:

```
<dataset>/
├── corpus.tsv     # doc_id \t [title \t] text
├── queries.tsv    # query_id \t text
└── qrels.tsv      # query_id \t iteration \t doc_id \t relevance
```

```python
from eval_data import load_beir_like
from evaluation import RetrievalEvaluator

ds = load_beir_like("datasets/nfcorpus")
print(f"{len(ds.corpus)} docs, {len(ds.queries)} queries")

# ds.queries is a list of EvalQuery ready to pass to the evaluator
evaluator = RetrievalEvaluator()
evaluator.add_queries(ds.queries)
report = evaluator.evaluate(search_fn=my_search, k_values=[1, 5, 10])
```

Works unmodified with most BEIR datasets, CISI, or any dataset that
follows the TREC qrels convention. Corpus rows with an optional
`title` column are joined with a space; `qrels` rows with grade `0`
are dropped by default; references to missing docs are silently
skipped (matching `trec-eval`).

## 🛡️ Retrieval Quality Gate

A CI guardrail that compares a current evaluation report against a
**committed baseline** and fails the build when retrieval quality
regresses beyond configurable thresholds. Because the gate operates
on serialized `EvalReport` JSON it stays cheap, deterministic and
fully decoupled from the embedding model — the same logic guards
dense, sparse, hybrid and reranked pipelines without modification.

How regressions are detected (per metric):

- **Absolute drop** — flagged when `current < baseline − abs_drop`
  (catches small but material drops on already-high metrics).
- **Relative drop** — flagged when `current < baseline · (1 − rel_drop)`
  (catches proportionally large regressions on low-magnitude metrics).
- **Hard floor** — optional `min_value` so a metric never silently
  drifts below an SLA.
- **Per-query diagnostics** — the gate also surfaces *which* queries
  regressed (by reciprocal-rank, AP and NDCG@k) so the PR comment
  pinpoints what to investigate.

Workflow:

```bash
# 1. Run evaluation and write a current report
python cli.py evaluate \
  --dataset examples/quality_gate/dataset \
  --k 1,3,5,10 \
  --output report.json --quiet

# 2. Compare it against the committed baseline
python cli.py quality-gate \
  --baseline examples/quality_gate/baseline.json \
  --report   report.json \
  --config   examples/quality_gate/config.json \
  --markdown gate.md

# Exit codes: 0 = pass, 1 = regression, 2 = usage / I/O error
```

Output (excerpt of the Markdown summary the gate writes for PR comments):

```markdown
## Retrieval quality gate — **PASS**
_model `all-MiniLM-L6-v2` · queries 8 → 8 · 2026-04-30T12:00:00Z_

| Metric    | Baseline | Current |       Δ |    Δ% | Status |
| ---       |     ---: |    ---: |    ---: |  ---: |  :---: |
| `mrr`     |   0.9583 |  0.9583 | +0.0000 | +0.0% |     OK |
| `ndcg@5`  |   0.9667 |  0.9667 | +0.0000 | +0.0% |     OK |
| `recall@10` | 1.0000 |  1.0000 | +0.0000 | +0.0% |     OK |
```

Bootstrap a baseline the first time you wire the gate up:

```bash
python cli.py quality-gate \
  --baseline examples/quality_gate/baseline.json \
  --report   report.json \
  --update-baseline
```

Re-run [`examples/quality_gate/regenerate_baseline.py`](examples/quality_gate/regenerate_baseline.py)
when you intentionally change the model, the dataset, or the
retrieval configuration. Tweak per-metric thresholds in
[`examples/quality_gate/config.json`](examples/quality_gate/config.json)
or use the built-in `--strict` preset.

## 🧪 A/B Retrieval Comparison

Use `ab-compare` when you have per-query metric vectors for two retrieval
configurations and want uncertainty-aware evidence before changing the
ranking stack. The command reports bootstrap confidence intervals, paired
bootstrap p-values, sign-test counts, and a compact winner label.

Input files are intentionally simple JSON objects mapping metric names to
aligned per-query values:

```json
{
  "mrr": [1.0, 0.5, 0.0],
  "ndcg@10": [0.92, 0.61, 0.18]
}
```

Run the comparison and write both machine-readable JSON and Markdown for a
pull request comment:

```bash
python cli.py ab-compare \
  --a baseline-metrics.json \
  --b candidate-metrics.json \
  --name-a dense-only \
  --name-b reranked \
  --output ab-report.json \
  --markdown ab-report.md
```

The terminal summary stays terse for CI logs:

```text
A/B Comparison: dense-only vs reranked (50 queries)
- mrr: Δ=+0.0240 (+3.19%), p=0.0180, winner=reranked
- ndcg@10: Δ=+0.0175 (+2.08%), p=0.0435, winner=reranked
```

## 🎚️ Score Calibration Diagnostics

When search scores are used as user-facing confidence signals or threshold
cutoffs, calibration matters. The `calibrate` command converts labelled
retrieval examples into reliability bins, Brier score, expected calibration
error (ECE), and max calibration error.

Accepted input shapes:

```json
{"scores": [0.12, 0.88, 0.73], "labels": [0, 1, 1]}
```

or row-oriented JSON:

```json
[
  {"score": 0.12, "label": 0},
  {"score": 0.88, "label": 1}
]
```

Run:

```bash
python cli.py calibrate \
  --input labelled-scores.json \
  --bins 10 \
  --output calibration-report.json
```

Use high ECE or large per-bin gaps as a signal to retune thresholds, add a
calibration layer, or split analysis by query type before shipping score
interpretation changes.

## 📈 Observability & Hardening

- **Structured logs** — Every line is a single JSON object with
  `ts`, `level`, `logger`, `msg`, `request_id`, plus any `extra=`
  fields. Switch to human-readable logs with `SSE_LOG_JSON=false`.
- **Request correlation** — The API accepts an upstream
  `X-Request-ID` header (or mints a UUID) and echoes it on every
  response, including rate-limit rejections. The ID is propagated
  into every log record emitted during the request via
  `contextvars`, so async code paths stay correlated.
- **Prometheus metrics** — `GET /metrics` returns scrape-ready text:
  `sse_requests_total`, `sse_request_latency_seconds` (histogram),
  `sse_searches_total`, `sse_documents_indexed`,
  `sse_rate_limited_total`. No `prometheus_client` dependency.
- **Rate limiting** — Opt-in token-bucket limiter per client IP.
  `/health`, `/metrics`, and `/docs` are exempt. Rejections return
  HTTP 429 with a `Retry-After` header.
- **Security headers** — On by default: `X-Content-Type-Options:
  nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy:
  strict-origin-when-cross-origin`, `Permissions-Policy:
  geolocation=(), microphone=(), camera=()`.

Toggle via env vars:

```bash
SSE_LOG_JSON=true
SSE_RATE_LIMIT_ENABLED=true
SSE_RATE_LIMIT_PER_MINUTE=120
SSE_SECURITY_HEADERS_ENABLED=true
```

## ⚙️ Configuration

All settings are loaded from environment variables (prefix `SSE_`) or a `.env` file.
See [`.env.example`](.env.example) for the full list.

| Variable                         | Default              | Description                      |
|----------------------------------|----------------------|----------------------------------|
| `SSE_MODEL_NAME`                 | `all-MiniLM-L6-v2`  | Sentence-transformer model       |
| `SSE_USE_FAISS`                  | `true`               | Enable FAISS backend             |
| `SSE_PORT`                       | `8000`               | API server port                  |
| `SSE_WORKERS`                    | `1`                  | Uvicorn worker count             |
| `SSE_LOG_LEVEL`                  | `INFO`               | Logging level                    |
| `SSE_LOG_JSON`                   | `true`               | Emit JSON logs (one per line)    |
| `SSE_CORS_ORIGINS`               | `["*"]`              | Allowed CORS origins             |
| `SSE_RATE_LIMIT_ENABLED`         | `false`              | Enable in-process rate limiter   |
| `SSE_RATE_LIMIT_PER_MINUTE`      | `60`                 | Max requests/min per client IP   |
| `SSE_SECURITY_HEADERS_ENABLED`   | `true`               | Attach common security headers   |
| `SSE_MAX_TOP_K`                  | `50`                 | Maximum results per query        |
| `SSE_MAX_BATCH_SIZE`             | `100`                | Maximum queries per batch        |
| `SSE_INDEX_PATH`                 | —                    | Load index on startup            |
| `SSE_AUTO_SAVE_PATH`             | —                    | Auto-save after modifications    |

## 🎯 Fine-Tuning

Train domain-specific embeddings using contrastive learning:

```python
from training import FineTuner, TrainingConfig, TrainingPair

config = TrainingConfig(
    base_model="all-MiniLM-L6-v2",
    output_dir="models/fine-tuned",
    epochs=5,
    loss_type="cosine",      # or "contrastive", "triplet"
    cv_folds=3,              # k-fold cross-validation
)

tuner = FineTuner(config)
tuner.add_pairs([
    TrainingPair(query="breach of contract", positive="contractual obligation violated"),
    TrainingPair(query="negligence claim", positive="failure to exercise reasonable care"),
])
# Or load from JSONL
tuner.load_pairs_jsonl("data/training_pairs.jsonl")

result = tuner.train()
print(f"Best score: {result.best_score}")
```

**CLI:**

```bash
python training.py --data pairs.jsonl --model all-MiniLM-L6-v2 --epochs 5 --cv-folds 3
```

## 📊 Evaluation & Benchmarking

Evaluate retrieval quality with standard IR metrics:

```python
from evaluation import RetrievalEvaluator, EvalQuery, ModelBenchmark

evaluator = RetrievalEvaluator()
evaluator.add_queries([
    EvalQuery(query="machine learning", relevant_docs=["doc_1", "doc_3"]),
    EvalQuery(query="deep learning", relevant_docs=["doc_2"],
              relevance_grades={"doc_2": 3, "doc_5": 1}),  # graded relevance
])

report = evaluator.evaluate(search_fn=my_search, k_values=[1, 3, 5, 10])
report.print_summary()
```

**Multi-model benchmark:**

```python
benchmark = ModelBenchmark(
    models=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
    queries=eval_queries,
    corpus=documents,
    corpus_ids=doc_ids,
)
result = benchmark.run()
result.print_comparison()
```

**Output:**

```
  Model Benchmark Comparison
  ================================================================
  Model                          MRR      MAP    NDCG@5     R@10
  all-mpnet-base-v2            0.9200   0.8850   0.9100   0.9500 🏆
  multi-qa-MiniLM-L6-cos-v1   0.8800   0.8400   0.8700   0.9200
  all-MiniLM-L6-v2            0.8500   0.8100   0.8400   0.8900
```

## 🧪 Experiment Tracking

Track training runs, compare models, and manage model versions — all locally with zero external dependencies:

```python
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("experiments/")

# Log a training run
with tracker.start_run("fine-tune-v1", tags=["baseline"]) as run:
    run.log_params({"model": "all-MiniLM-L6-v2", "epochs": 5, "lr": 2e-5})
    for epoch in range(5):
        run.log_metric("train_loss", losses[epoch], step=epoch)
    run.log_metrics({"mrr": 0.87, "map": 0.82, "ndcg@5": 0.79})
    run.set_model_version("1.0.0")
    run.log_artifact("models/fine-tuned/config.json")

# Compare experiments
comparison = tracker.compare(["fine-tune-v1", "fine-tune-v2"])
comparison.print_table()

# Find the best model
best = tracker.best_run(metric="mrr", higher_is_better=True)
print(f"Best: {best.name} (MRR={best.final_metrics['mrr']})")

# Model lineage
lineage = tracker.model_lineage("fine-tune-v3")
for run in lineage:
    print(f"  {run.model_version} <- ", end="")
```

**CLI:**

```bash
# List all runs
python experiment_tracker.py list --status completed

# Compare runs
python experiment_tracker.py compare fine-tune-v1 fine-tune-v2

# Find best run
python experiment_tracker.py best mrr

# Export summary
python experiment_tracker.py export --output results.json
```

**Features:**
- **Run lifecycle** — automatic timing, status tracking, failure handling
- **Metric history** — step-by-step training curves with timestamps
- **Artifact logging** — copy model files, configs, checkpoints
- **Model versioning** — version strings + SHA-256 integrity hashes
- **Lineage tracking** — parent-child chains for iterative experiments
- **Comparison tables** — side-by-side metric diffs against baseline
- **Persistence** — JSON files, git-friendly, no server required

## 📁 Project Structure

```
semantic-search-engine/
├── api.py                    # FastAPI REST application
├── semantic_search.py        # Core search engine class
├── cli.py                    # Command-line interface (index/search/evaluate/info)
├── mmr.py                    # Maximal Marginal Relevance diversifier
├── training.py               # Fine-tuning pipeline (contrastive/triplet/CV)
├── evaluation.py             # Retrieval metrics & multi-model benchmarking
├── eval_data.py              # BEIR/TREC TSV dataset loader
├── experiment_tracker.py     # MLOps experiment tracking & model registry
├── metrics.py                # Zero-dependency Prometheus metrics registry
├── rate_limit.py             # Token-bucket rate limiter (per client IP)
├── logging_config.py         # JSON logging + request-ID contextvars
├── config.py                 # Pydantic-settings configuration
├── demo.py                   # Interactive CLI demo
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata & tool config
├── Makefile                  # Dev shortcuts
├── Dockerfile                # Multi-stage production build
├── docker-compose.yml        # Container orchestration
├── .env.example              # Configuration template
├── .github/
│   └── workflows/
│       └── ci.yml            # CI pipeline (lint → test → docker)
├── tests/
│   ├── test_search.py        # Core engine unit tests
│   ├── test_api.py           # API integration tests
│   ├── test_training.py      # Training & evaluation tests
│   └── test_experiment_tracker.py  # Experiment tracking tests (46 tests)
├── LICENSE
└── README.md
```

## 🧪 Running Tests

```bash
# All tests
make test

# API tests only
make test-api

# With verbose output
pytest tests/ -v --tb=short
```

## 📈 Benchmarks

| Dataset Size  | Index Time | Query Time | Memory |
|---------------|------------|------------|--------|
| 1,000 docs    | 2.1s       | 5ms        | 45MB   |
| 10,000 docs   | 18.5s      | 8ms        | 120MB  |
| 100,000 docs  | 3.2min     | 15ms       | 850MB  |

*Tested on Intel i7-10700K, 32GB RAM*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run `make lint` and `make test` before committing
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) — Amazing embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Hugging Face](https://huggingface.co/) — Model hosting and community

---

Made with ❤️ by [get2salam](https://github.com/get2salam)
