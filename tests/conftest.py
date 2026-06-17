"""Test-only offline fakes for heavyweight embedding dependencies."""

from __future__ import annotations

import re
import sys
import types
import zlib

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SYNONYMS = {
    "ai": ("artificial", "intelligence", "machine", "learning"),
    "ml": ("machine", "learning", "artificial", "intelligence"),
    "programming": ("python", "javascript", "language"),
    "neural": ("deep", "network", "networks", "learning"),
    "networks": ("neural", "deep"),
    "web": ("javascript", "browser", "browsers", "frontend"),
}


class DeterministicSentenceTransformer:
    """Small deterministic encoder that keeps tests offline and semantic-ish."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._dim = 64

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **_: object,
    ):
        single = isinstance(sentences, str)
        texts = [sentences] if single else sentences
        vectors = np.vstack([self._encode_one(text) for text in texts]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        if single:
            return vectors[0] if convert_to_numpy else vectors[0].tolist()
        return vectors if convert_to_numpy else vectors.tolist()

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        tokens = list(_TOKEN_RE.findall(text.lower()))
        expanded = tokens + [alias for token in tokens for alias in _SYNONYMS.get(token, ())]
        for token in expanded:
            vector[zlib.crc32(token.encode("utf-8")) % self._dim] += 1.0
        return vector


fake_module = types.ModuleType("sentence_transformers")
fake_module.__dict__["SentenceTransformer"] = DeterministicSentenceTransformer
sys.modules.setdefault("sentence_transformers", fake_module)
