# 🔍 Semantic Search Engine

[![CI](https://github.com/get2salam/semantic-search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/get2salam/semantic-search-engine/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Sentence Transformers](https://img.shields.io/badge/🤗-Sentence%20Transformers-yellow)](https://www.sbert.net/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

A lightweight, production-ready semantic search engine powered by state-of-the-art sentence embeddings. Find similar documents based on **meaning**, not just keywords.

---

## ✨ Features

- 🚀 **Fast & Efficient** — FAISS-powered vector similarity search
- 🧠 **State-of-the-Art Embeddings** — Uses `all-MiniLM-L6-v2` (384-dim, blazing fast)
- 📊 **Multiple Use Cases** — Document search, Q&A, recommendation systems
- 🔧 **Easy to Extend** — Clean, modular architecture
- 💾 **Persistent Storage** — Save and load indices to disk
- 🐳 **Docker Support** — Build and run in a container with one command
- 🐍 **Pure Python** — No external services required

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      User / Client                       │
└────────────────────────┬─────────────────────────────────┘
                         │  query text
                         ▼
┌──────────────────────────────────────────────────────────┐
│               SemanticSearchEngine                       │
│                                                          │
│  ┌────────────────┐   ┌─────────────────────────────┐   │
│  │ Sentence-       │   │  Vector Index                │   │
│  │ Transformers    │──▶│  (FAISS / NumPy fallback)    │   │
│  │ Encoder         │   │                              │   │
│  │ (all-MiniLM-    │   │  • add_documents()           │   │
│  │  L6-v2)         │   │  • search(query, top_k)      │   │
│  └────────────────┘   │  • save() / load()            │   │
│                        └─────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Persistence Layer                               │     │
│  │  documents.json  │  embeddings.npy  │  config.json│    │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

**Data flow:** Documents → Encoder → Embedding vectors → FAISS index → Ranked results

---

## 🎯 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/get2salam/semantic-search-engine.git
cd semantic-search-engine

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from semantic_search import SemanticSearchEngine

# Initialize the engine
engine = SemanticSearchEngine()

# Add your documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language for data science",
    "Neural networks are inspired by biological neurons",
    "Deep learning requires large amounts of training data",
    "Natural language processing deals with text understanding",
]
engine.add_documents(documents)

# Search by meaning!
results = engine.search("AI and deep neural nets", top_k=3)
for doc, score in results:
    print(f"[{score:.3f}] {doc}")
```

**Output:**
```
[0.847] Neural networks are inspired by biological neurons
[0.823] Machine learning is a subset of artificial intelligence
[0.756] Deep learning requires large amounts of training data
```

### Run the Interactive Demo

```bash
python demo.py
```

---

## 🐳 Docker

Build and run without installing anything locally:

```bash
# Using Docker Compose (recommended)
docker compose up --build

# Or build manually
docker build -t semantic-search-engine .
docker run -it semantic-search-engine
```

---

## 🛠️ Advanced Usage

### Persistent Index

```python
# Save your index to disk
engine.save("my_search_index")

# Load it later — no need to re-encode
engine = SemanticSearchEngine.load("my_search_index")
```

### Custom Embedding Models

```python
# Use any model from the sentence-transformers hub
engine = SemanticSearchEngine(model_name="all-mpnet-base-v2")
```

### Similarity Threshold

```python
# Only return results above a minimum similarity score
results = engine.search("quantum computing", top_k=5, threshold=0.6)
```

### Batch Search

```python
queries = ["AI research", "web development", "health tips"]
all_results = engine.search_batch(queries, top_k=3)
```

### Large Dataset Ingestion

```python
# Control batch size to manage memory
engine.add_documents(large_document_list, batch_size=1000)
```

---

## 📖 API Reference

### `SemanticSearchEngine`

| Method | Description |
|---|---|
| `__init__(model_name, use_faiss, normalize_embeddings)` | Create an engine. Defaults: `all-MiniLM-L6-v2`, FAISS on, normalized. |
| `add_documents(docs, batch_size, show_progress)` | Encode and index a list of text documents. |
| `search(query, top_k, threshold)` | Return the `top_k` most similar documents with scores. |
| `search_batch(queries, top_k)` | Run multiple searches in one call. |
| `save(path)` | Persist documents, embeddings, and config to a directory. |
| `load(path)` *(classmethod)* | Restore an engine from a saved directory. |
| `clear()` | Remove all documents and embeddings. |

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | Any [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) model name |
| `use_faiss` | `bool` | `True` | Use FAISS for fast search; falls back to NumPy if unavailable |
| `normalize_embeddings` | `bool` | `True` | L2-normalize vectors (enables cosine similarity via dot product) |

---

## 📁 Project Structure

```
semantic-search-engine/
├── semantic_search.py       # Core search engine class
├── demo.py                  # Interactive demo script
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata & tool config
├── Makefile                 # Common dev commands
├── Dockerfile               # Container build
├── docker-compose.yml       # Compose service definition
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions — lint, test, Docker build
├── tests/
│   └── test_search.py       # Unit tests (pytest)
└── README.md
```

---

## 🧪 Testing & Development

```bash
# Run tests
make test          # or: pytest tests/ -v

# Lint
make lint          # or: ruff check .

# Auto-format
make format        # or: ruff format .

# See all available commands
make help
```

---

## 📊 Benchmarks

| Dataset Size | Index Time | Query Time | Memory |
|---|---|---|---|
| 1,000 docs | ~2 s | ~5 ms | ~45 MB |
| 10,000 docs | ~18 s | ~8 ms | ~120 MB |
| 100,000 docs | ~3 min | ~15 ms | ~850 MB |

*Measured on Intel i7-10700K, 32 GB RAM, FAISS-cpu*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure `ruff check .` and `pytest` pass before submitting.

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) — State-of-the-art embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search by Meta AI
- [Hugging Face](https://huggingface.co/) — Model hosting and community

---

Made with ❤️ by [get2salam](https://github.com/get2salam)
