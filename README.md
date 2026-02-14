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
- 🤖 **State-of-the-Art Embeddings** — Uses `all-MiniLM-L6-v2` (384-dim, blazing fast)
- 🌐 **REST API** — Production-grade FastAPI with OpenAPI docs, validation, CORS
- 🎯 **Fine-Tuning Pipeline** — Domain-adaptive training with contrastive/triplet loss and k-fold CV
- 📊 **Retrieval Evaluation** — MRR, MAP, NDCG@k, Precision@k, Recall@k with multi-model benchmarking
- 🐳 **Docker Ready** — Multi-stage build, non-root user, health checks
- 🔄 **CI/CD** — GitHub Actions: lint → test (matrix) → Docker build & verify
- 📈 **Observability** — Request timing headers, structured logging, health endpoint
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

## ⚙️ Configuration

All settings are loaded from environment variables (prefix `SSE_`) or a `.env` file.
See [`.env.example`](.env.example) for the full list.

| Variable                  | Default              | Description                      |
|---------------------------|----------------------|----------------------------------|
| `SSE_MODEL_NAME`          | `all-MiniLM-L6-v2`  | Sentence-transformer model       |
| `SSE_USE_FAISS`           | `true`               | Enable FAISS backend             |
| `SSE_PORT`                | `8000`               | API server port                  |
| `SSE_WORKERS`             | `1`                  | Uvicorn worker count             |
| `SSE_LOG_LEVEL`           | `INFO`               | Logging level                    |
| `SSE_CORS_ORIGINS`        | `["*"]`              | Allowed CORS origins             |
| `SSE_MAX_TOP_K`           | `50`                 | Maximum results per query        |
| `SSE_MAX_BATCH_SIZE`      | `100`                | Maximum queries per batch        |
| `SSE_INDEX_PATH`          | —                    | Load index on startup            |
| `SSE_AUTO_SAVE_PATH`      | —                    | Auto-save after modifications    |

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

## 📁 Project Structure

```
semantic-search-engine/
├── api.py                    # FastAPI REST application
├── semantic_search.py        # Core search engine class
├── training.py               # Fine-tuning pipeline (contrastive/triplet/CV)
├── evaluation.py             # Retrieval metrics & multi-model benchmarking
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
│   └── test_training.py      # Training & evaluation tests
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
