"""Qdrant on-disk: хранение точек (chunk = точка) и поиск.

Каждая точка: id(uuid) + векторы {dense [+ sparse]} + payload {paper_name, paper_path, page, text}.
Поиск: dense по умолчанию; при HYBRID=true — dense+sparse со слиянием RRF (как в оригинале).

Индексация ИНКРЕМЕНТАЛЬНАЯ: манифест (manifest.json) хранит per-file хэши, поэтому при запуске
индексируются только новые/изменённые статьи, уже готовые не трогаются. Если меняется конфиг
векторного пространства (hybrid/размерность/режим эмбеддингов/чанкинг) — пересобирается всё.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings
from .chunking import Chunk
from . import embeddings as emb


@dataclass
class Hit:
    text: str
    score: float
    paper_name: str
    paper_path: str
    page: int


@dataclass
class FileStatus:
    name: str
    action: str          # indexed | new | changed | unchanged | removed | pending
    chunks: int = 0


class Store:
    def __init__(self) -> None:
        self.client = QdrantClient(path=settings.qdrant_path)
        self.collection = settings.qdrant_collection

    # ---------- collection ----------

    def _config_key(self) -> str:
        """Идентичность векторного пространства: при её смене нужен полный пересбор."""
        return (f"hybrid={settings.hybrid}|dim={settings.embed_dim}"
                f"|embed={'jina' if settings.has_embeddings else 'hashed'}"
                f"|chunk={settings.chunk_max_chars}/{settings.chunk_overlap_chars}"
                f"|maxchunks={settings.max_chunks_per_paper}")

    def _create(self) -> None:
        vectors = {"dense": qm.VectorParams(size=settings.embed_dim, distance=qm.Distance.COSINE)}
        sparse = {"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)} if settings.hybrid else None
        self.client.create_collection(self.collection, vectors_config=vectors,
                                      sparse_vectors_config=sparse)

    def _recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._create()

    def _ensure(self) -> None:
        if not self.client.collection_exists(self.collection):
            self._create()

    # ---------- manifest ----------

    def _manifest_path(self) -> Path:
        return Path(settings.qdrant_path) / "manifest.json"

    def load_manifest(self) -> dict:
        p = self._manifest_path()
        if not p.exists():
            return {"config": "", "files": {}}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"config": "", "files": {}}

    def _save_manifest(self, files: Dict[str, dict]) -> None:
        self._manifest_path().write_text(
            json.dumps({"config": self._config_key(), "files": files}, ensure_ascii=False, indent=2))

    # ---------- diff (что будет проиндексировано) ----------

    def diff(self, valid) -> Dict[str, object]:
        """Без записи: какие файлы новые/изменённые/без изменений и нужен ли полный пересбор."""
        cfg = self._config_key()
        manifest = self.load_manifest()
        rebuild_all = (manifest.get("config") != cfg
                       or not self.client.collection_exists(self.collection))
        prev = {} if rebuild_all else manifest.get("files", {})
        statuses: List[FileStatus] = []
        for p in valid:
            if rebuild_all or p.name not in prev:
                statuses.append(FileStatus(p.name, "new"))
            elif prev[p.name].get("hash") != p.content_hash:
                statuses.append(FileStatus(p.name, "changed", prev[p.name].get("chunks", 0)))
            else:
                statuses.append(FileStatus(p.name, "unchanged", prev[p.name].get("chunks", 0)))
        removed = [n for n in prev if n not in {p.name for p in valid}]
        return {"rebuild_all": rebuild_all, "statuses": statuses, "removed": removed}

    # ---------- index ----------

    def _points_for(self, chunks: List[Chunk]) -> List[qm.PointStruct]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        dense = emb.embed_dense(texts, is_query=False)
        sparse = emb.embed_sparse(texts) if settings.hybrid else None
        points = []
        for i, c in enumerate(chunks):
            vector = {"dense": dense[i]}
            if sparse is not None:
                idx, val = sparse[i]
                vector["sparse"] = qm.SparseVector(indices=idx, values=val)
            points.append(qm.PointStruct(
                id=str(uuid.uuid4()), vector=vector,
                payload={"paper_name": c.paper_name, "paper_path": c.paper_path,
                         "page": c.page, "text": c.text}))
        return points

    def _delete_paper(self, name: str) -> None:
        self.client.delete(self.collection, points_selector=qm.Filter(
            must=[qm.FieldCondition(key="paper_name", match=qm.MatchValue(value=name))]))

    def sync(self, valid, chunks_by_name: Dict[str, List[Chunk]],
             force: bool = False) -> List[FileStatus]:
        """Инкрементальная синхронизация индекса с папкой. Возвращает per-file отчёт."""
        cfg = self._config_key()
        manifest = self.load_manifest()
        rebuild_all = (force or manifest.get("config") != cfg
                       or not self.client.collection_exists(self.collection))
        prev = {} if rebuild_all else dict(manifest.get("files", {}))

        if rebuild_all:
            self._recreate()
            prev = {}
        else:
            self._ensure()

        report: List[FileStatus] = []
        files = dict(prev)
        cur_names = {p.name for p in valid}

        # удалённые из папки -> чистим точки
        for name in list(prev):
            if name not in cur_names:
                self._delete_paper(name)
                files.pop(name, None)
                report.append(FileStatus(name, "removed"))

        for p in valid:
            ch = chunks_by_name.get(p.name, [])
            if not rebuild_all and p.name in prev and prev[p.name].get("hash") == p.content_hash:
                report.append(FileStatus(p.name, "unchanged", prev[p.name].get("chunks", len(ch))))
                continue
            action = "indexed" if rebuild_all else ("changed" if p.name in prev else "new")
            if action == "changed":
                self._delete_paper(p.name)
            points = self._points_for(ch)
            if points:
                self.client.upsert(self.collection, points=points)
            files[p.name] = {"hash": p.content_hash, "chunks": len(ch)}
            report.append(FileStatus(p.name, action, len(ch)))

        self._save_manifest(files)
        return report

    # ---------- search ----------

    def search(self, query: str, limit: Optional[int] = None,
               paper_name: Optional[str] = None) -> List[Hit]:
        k = limit or settings.search_limit
        dense_q = emb.embed_dense_one(query, is_query=True)
        flt = None
        if paper_name:
            flt = qm.Filter(must=[qm.FieldCondition(
                key="paper_name", match=qm.MatchValue(value=paper_name))])

        if settings.hybrid:
            idx, val = emb.embed_sparse_one(query)
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    qm.Prefetch(query=dense_q, using="dense", limit=k * 2),
                    qm.Prefetch(query=qm.SparseVector(indices=idx, values=val),
                                using="sparse", limit=k * 2),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=k, with_payload=True, query_filter=flt,
            ).points
        else:
            res = self.client.query_points(
                self.collection, query=dense_q, using="dense",
                limit=k, with_payload=True, query_filter=flt,
            ).points

        return [Hit(text=p.payload["text"], score=p.score, paper_name=p.payload["paper_name"],
                    paper_path=p.payload["paper_path"], page=p.payload["page"]) for p in res]
