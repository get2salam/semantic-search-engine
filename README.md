# 🔍 Semantic Search Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Sentence Transformers](https://img.shields.io/badge/🤗-Sentence%20Transformers-yellow)](https://www.sbert.net/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight, production-ready semantic search engine powered by state-of-the-art sentence embeddings. Find similar documents based on **meaning**, not just keywords.

![Demo](https://img.shields.io/badge/demo-interactive-brightgreen)

## ✨ Features

- 🚀 **Fast & Efficient** - FAISS-powered vector similarity search
- 🧠 **State-of-the-Art Embeddings** - Uses `all-MiniLM-L6-v2` (384-dim, blazing fast)
- 📊 **Multiple Use Cases** - Document search, Q&A, recommendation systems
- 🔧 **Easy to Extend** - Clean, modular architecture
- 💾 **Persistent Storage** - Save and load indices to disk
- 🐍 **Pure Python** - No external services required

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
    "Natural language processing deals with text understanding"
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

## 🛠️ Advanced Usage

### Persistent Index

```python
# Save your index
engine.save("my_search_index")

# Load it later
engine = SemanticSearchEngine.load("my_search_index")
```

### Custom Embedding Models

```python
# Use any sentence-transformers model
engine = SemanticSearchEngine(model_name="all-mpnet-base-v2")
```

### Batch Processing

```python
# Add documents in batches for large datasets
engine.add_documents(large_document_list, batch_size=1000)
```

## 📁 Project Structure

```
semantic-search-engine/
├── semantic_search.py    # Core search engine class
├── demo.py               # Interactive demo script
├── requirements.txt      # Python dependencies
├── tests/
│   └── test_search.py    # Unit tests
└── README.md
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📊 Benchmarks

| Dataset Size | Index Time | Query Time | Memory |
|-------------|------------|------------|--------|
| 1,000 docs  | 2.1s       | 5ms        | 45MB   |
| 10,000 docs | 18.5s      | 8ms        | 120MB  |
| 100,000 docs| 3.2min     | 15ms       | 850MB  |

*Tested on Intel i7-10700K, 32GB RAM*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) - Amazing embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Hugging Face](https://huggingface.co/) - Model hosting and community

---

Made with ❤️ by [get2salam](https://github.com/get2salam)
