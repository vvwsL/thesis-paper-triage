"""Эмбеддинги.

Dense:
  - есть ключ Jina -> jina-embeddings-v3 через REST (
    task=retrieval.passage для документов, retrieval.query для запроса);
  - иначе (dry-run / нет ключа) -> детерминированный hashed bag-of-words вектор.
    Это позволяет триажу по косинусной близости работать БЕЗ вызовов API
    (релевантная статья всё равно получит более высокий score, чем нерелевантная).

Sparse (BM25): только при HYBRID=true, через fastembed Qdrant/bm25 (локально, CPU).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Tuple

import requests

from .config import settings

_WORD = re.compile(r"[a-zA-Zа-яА-Я0-9]+")


# ---------- dense ----------

def _hashed_embedding(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    for tok in _WORD.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


_JINA_BATCH = 64   # макс. чанков в одном запросе к Jina (защита от лимита размера)


def _jina_embed(texts: List[str], task: str) -> List[List[float]]:
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {settings.jina_api_key}"}
    out: List[List[float]] = []
    for i in range(0, len(texts), _JINA_BATCH):
        batch = texts[i:i + _JINA_BATCH]
        payload = {"model": settings.embed_model, "task": task,
                   "dimensions": settings.embed_dim, "input": batch}
        r = requests.post(settings.embed_url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        out.extend([float(x) for x in item["embedding"]] for item in data)
    return out


def embed_dense(texts: List[str], *, is_query: bool) -> List[List[float]]:
    if settings.has_embeddings:
        task = "retrieval.query" if is_query else "retrieval.passage"
        return _jina_embed(texts, task)
    return [_hashed_embedding(t, settings.embed_dim) for t in texts]


def embed_dense_one(text: str, *, is_query: bool) -> List[float]:
    return embed_dense([text], is_query=is_query)[0]


# ---------- sparse (BM25 via fastembed) ----------

_sparse_model = None


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def embed_sparse(texts: List[str]) -> List[Tuple[List[int], List[float]]]:
    """Возвращает [(indices, values), ...]. Только при HYBRID=true."""
    model = _get_sparse_model()
    out = []
    for emb in model.embed(texts):
        out.append((emb.indices.tolist(), emb.values.tolist()))
    return out


def embed_sparse_one(text: str) -> Tuple[List[int], List[float]]:
    return embed_sparse([text])[0]
