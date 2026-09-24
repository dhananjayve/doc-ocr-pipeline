"""Hybrid retrieval: dense vector candidates re-ranked with BM25 over the candidate set.

Financial questions often hinge on exact terms ("diluted EPS", "Note 14", "FY2024") that dense
embeddings blur; BM25 restores them without a separate keyword index.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from finpipe.config import Settings
from finpipe.embed import Embedder
from finpipe.store import Hit, VectorStore

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what "
    "which with how much many did does do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    q_terms = set(tokenize(query))
    tokenized = [tokenize(d) for d in docs]
    if not q_terms or not docs:
        return [0.0] * len(docs)
    avgdl = sum(len(t) for t in tokenized) / len(tokenized) or 1.0
    df = Counter(term for toks in tokenized for term in set(toks) & q_terms)
    n = len(docs)
    scores = []
    for toks in tokenized:
        tf = Counter(toks)
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(toks) / avgdl))
        scores.append(s)
    return scores


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder):
        self.s = settings
        self.store = store
        self.embedder = embedder
        self._reranker = None
        if settings.reranker_model:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.reranker_model)

    def retrieve(self, query: str, k: int | None = None, where: dict | None = None) -> list[Hit]:
        k = k or self.s.top_k
        total = self.store.count()
        if total == 0:
            return []
        n = min(total, k * self.s.candidate_multiplier)
        hits = self.store.query(self.embedder.embed_query(query), n=n, where=where)
        if not hits:
            return []

        kw = bm25_scores(query, [h.text for h in hits])
        kw_max = max(kw) or 1.0
        a = self.s.hybrid_alpha
        for h, s in zip(hits, kw):
            h.score = a * h.similarity + (1 - a) * (s / kw_max)
        hits.sort(key=lambda h: h.score, reverse=True)

        if self._reranker is not None:
            pool = hits[: max(k * 2, k)]
            ce = self._reranker.predict([(query, h.text) for h in pool])
            for h, s in zip(pool, ce):
                h.score = float(s)
            hits = sorted(pool, key=lambda h: h.score, reverse=True)
        return hits[:k]
