"""Embedding backends and vector maths.

Three embedders ship with the package:

* :class:`HashEmbedder` - deterministic, offline, dependency-free. Great for
  tests, demos and lexical similarity experiments.
* :class:`TfidfEmbedder` - classic TF-IDF vectors fitted on a corpus.
* :class:`OpenAIEmbedder` - ``/v1/embeddings`` for real semantic search.

All of them implement ``embed(texts) -> list[list[float]]`` and
``aembed(texts)``, so they can be swapped freely (including in
:class:`pyagents.memory.VectorMemory`).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigurationError, ProviderError
from .llm.transport import HttpTransport, get_default_transport
from .utils import maybe_await

__all__ = [
    "Embedder",
    "HashEmbedder",
    "OpenAIEmbedder",
    "TfidfEmbedder",
    "cosine_similarity",
    "dot_product",
    "normalize",
    "top_k_indices",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(a * b for a, b in zip(left, right))


def normalize(vector: Sequence[float]) -> List[float]:
    """Return the L2-normalised copy of ``vector``."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return list(vector)
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0.0 when either is empty).

    >>> round(cosine_similarity([1, 0], [1, 0]), 3)
    1.0
    """
    if not left or not right:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product(left, right) / (left_norm * right_norm)


def top_k_indices(
    query: Sequence[float], vectors: Sequence[Sequence[float]], k: int = 3
) -> List[Tuple[int, float]]:
    """Return ``(index, score)`` for the ``k`` most similar vectors."""
    scored = [(index, cosine_similarity(query, vector)) for index, vector in enumerate(vectors)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


class Embedder(ABC):
    """Abstract embedding backend."""

    name = "embedder"
    #: Dimensionality of produced vectors, when fixed.
    dimensions: Optional[int] = None

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts."""

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        """Async variant of :meth:`embed` (defaults to a worker thread)."""
        return await asyncio.to_thread(self.embed, list(texts))

    def embed_one(self, text: str) -> List[float]:
        """Embed a single string."""
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    """Hashing bag-of-words embedder.

    Deterministic across processes (``hash()`` is salted, :mod:`hashlib` is
    not), so it is safe for tests and caching. Similarity reflects shared
    vocabulary - enough for many retrieval and routing experiments without a
    network call.

    Args:
        dimensions: Size of the embedding space.
        ngrams: Also index character 3-grams, which improves robustness to
            typos and morphology.
    """

    name = "hash"

    def __init__(self, dimensions: int = 256, *, ngrams: bool = True) -> None:
        if dimensions <= 0:
            raise ConfigurationError("dimensions must be positive", dimensions=dimensions)
        self.dimensions = dimensions
        self.ngrams = ngrams

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        dimensions = int(self.dimensions or 0)
        return int.from_bytes(digest, "big") % dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts using hashed token counts."""
        vectors: List[List[float]] = []
        for text in texts:
            counts: Counter = Counter()
            tokens = _tokenize(text)
            counts.update(tokens)
            if self.ngrams:
                for token in tokens:
                    padded = f"#{token}#"
                    counts.update(padded[i : i + 3] for i in range(max(1, len(padded) - 2)))
            vector = [0.0] * int(self.dimensions or 0)
            for token, count in counts.items():
                vector[self._bucket(token)] += float(count)
            vectors.append(normalize(vector))
        return vectors


class TfidfEmbedder(Embedder):
    """TF-IDF vectors fitted on a corpus.

    Call :meth:`fit` (or :meth:`fit_transform`) with your document set before
    embedding queries. Unseen terms are ignored at query time.

    >>> embedder = TfidfEmbedder()
    >>> _ = embedder.fit(["the cat sat", "the dog barked", "a cat and a dog"])
    >>> len(embedder.embed(["cat"])[0])
    7
    """

    name = "tfidf"

    def __init__(self, *, lowercase: bool = True) -> None:
        self.lowercase = lowercase
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []
        self.dimensions: Optional[int] = None

    def _terms(self, text: str) -> List[str]:
        return _tokenize(text) if self.lowercase else _TOKEN_RE.findall(text)

    def fit(self, corpus: Sequence[str]) -> TfidfEmbedder:
        """Learn the vocabulary and IDF weights from ``corpus``."""
        documents = [self._terms(document) for document in corpus]
        vocabulary: Dict[str, int] = {}
        for document in documents:
            for term in document:
                vocabulary.setdefault(term, len(vocabulary))
        total = max(1, len(documents))
        idf = [0.0] * len(vocabulary)
        for term, index in vocabulary.items():
            document_frequency = sum(1 for document in documents if term in set(document))
            idf[index] = math.log((1 + total) / (1 + document_frequency)) + 1.0
        self.vocabulary = vocabulary
        self.idf = idf
        self.dimensions = len(vocabulary)
        return self

    def fit_transform(self, corpus: Sequence[str]) -> List[List[float]]:
        """Fit on ``corpus`` and return the document vectors."""
        self.fit(corpus)
        return self.embed(corpus)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts using the fitted vocabulary."""
        if not self.vocabulary:
            raise ConfigurationError("TfidfEmbedder must be fitted before embedding")
        vectors: List[List[float]] = []
        for text in texts:
            terms = self._terms(text)
            counts = Counter(terms)
            vector = [0.0] * len(self.vocabulary)
            for term, count in counts.items():
                index = self.vocabulary.get(term)
                if index is None:
                    continue
                tf = 1.0 + math.log(count)
                vector[index] = tf * self.idf[index]
            vectors.append(normalize(vector))
        return vectors


class OpenAIEmbedder(Embedder):
    """Embeddings from OpenAI's ``/v1/embeddings`` endpoint.

    Works with any OpenAI-compatible embeddings server via ``base_url``.

    >>> embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")  # doctest: +SKIP
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        transport: Optional[HttpTransport] = None,
        dimensions: Optional[int] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        import os

        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAIEmbedder needs an API key (api_key=... or $OPENAI_API_KEY)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dimensions = dimensions
        self.headers = dict(headers or {})
        self._transport = transport
        self._owns_transport = transport is None

    @property
    def transport(self) -> HttpTransport:
        """Transport used for HTTP calls (lazily resolved)."""
        if self._transport is None:
            self._transport = get_default_transport()
        return self._transport

    def _payload(self, texts: Sequence[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        data = self.transport.post_json(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", **self.headers},
            payload=self._payload(texts),
            timeout=self.timeout,
            provider="openai",
        )
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [list(item.get("embedding") or []) for item in items]
        if len(vectors) != len(texts):
            raise ProviderError(
                f"Expected {len(texts)} embeddings, received {len(vectors)}", provider="openai"
            )
        if self.dimensions is None and vectors:
            self.dimensions = len(vectors[0])
        return vectors

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        """Async variant of :meth:`embed`."""
        data = await self.transport.apost_json(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", **self.headers},
            payload=self._payload(texts),
            timeout=self.timeout,
            provider="openai",
        )
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        return [list(item.get("embedding") or []) for item in items]

    def close(self) -> None:
        """Close the owned transport."""
        if self._owns_transport and self._transport is not None:
            self._transport.close()


def embeddings_from_config(config: Optional[Mapping[str, Any]]) -> Optional[Embedder]:
    """Build an embedder from a small config mapping.

    >>> embeddings_from_config({"type": "hash", "dimensions": 64}).dimensions
    64
    """
    if not config:
        return None
    kind = str(config.get("type", "hash")).lower()
    options = {key: value for key, value in config.items() if key != "type"}
    if kind in {"hash", "hashing"}:
        return HashEmbedder(**options)
    if kind in {"tfidf", "tf-idf"}:
        embedder = TfidfEmbedder()
        corpus = options.pop("corpus", None)
        if corpus:
            embedder.fit(corpus)
        return embedder
    if kind in {"openai", "openai-compatible"}:
        return OpenAIEmbedder(**options)
    raise ConfigurationError(f"Unknown embedder type '{kind}'", known=["hash", "tfidf", "openai"])


async def embed_many(embedder: Embedder, texts: Iterable[str]) -> List[List[float]]:
    """Embed texts with either sync or async embedders, uniformly."""
    return list(await maybe_await(embedder.aembed(list(texts))))
