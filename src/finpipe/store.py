"""Persistent local vector store (ChromaDB, cosine HNSW)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finpipe.chunker import Chunk

UPSERT_BATCH = 512


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    similarity: float
    score: float = 0.0


class VectorStore:
    def __init__(self, path: Path, collection: str):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def has_doc(self, doc_id: str) -> bool:
        return bool(self.col.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def add_chunks(self, doc_id: str, chunks: list[Chunk], embeddings: np.ndarray, base_meta: dict,
                   extra: list[dict] | None = None) -> None:
        """`extra` holds per-chunk metadata (e.g. the account or tax year a chunk's page belongs to)."""
        ids = [f"{doc_id}:{c.index}" for c in chunks]
        extra = extra or [{} for _ in chunks]
        metas = [
            base_meta | extra[i] | {
                "doc_id": doc_id,
                "chunk_index": c.index,
                "heading": " > ".join(c.heading_path),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "has_table": c.has_table,
            }
            for i, c in enumerate(chunks)
        ]
        docs = [c.text for c in chunks]
        for i in range(0, len(ids), UPSERT_BATCH):
            s = slice(i, i + UPSERT_BATCH)
            self.col.upsert(ids=ids[s], documents=docs[s], metadatas=metas[s], embeddings=embeddings[s].tolist())

    def query(self, embedding: np.ndarray, n: int, where: dict | None = None) -> list[Hit]:
        res = self.col.query(
            query_embeddings=[embedding.tolist()], n_results=n, where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(id=i, text=d, metadata=m, similarity=1.0 - dist)
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def retag(self, doc_id: str, meta: dict) -> None:
        res = self.col.get(where={"doc_id": doc_id}, include=["metadatas"])
        if res["ids"]:
            self.col.update(ids=res["ids"], metadatas=[m | meta for m in res["metadatas"]])

    def documents(self) -> list[dict]:
        """One row per indexed document: id, source, category, doc_type, chunk and page counts."""
        docs: dict[str, dict] = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(m["doc_id"], {
                "doc_id": m["doc_id"], "source": m.get("source", "?"), "category": m.get("category", "other"),
                "doc_type": m.get("doc_type", "other"), "chunks": 0, "pages": 0,
            })
            d["chunks"] += 1
            d["pages"] = max(d["pages"], m.get("page_end", 0))
        return sorted(docs.values(), key=lambda d: d["source"].lower())

    def count(self) -> int:
        return self.col.count()

    def sources(self) -> dict[str, int]:
        metas = self.col.get(include=["metadatas"])["metadatas"]
        out: dict[str, int] = {}
        for m in metas:
            out[m.get("source", "?")] = out.get(m.get("source", "?"), 0) + 1
        return out
