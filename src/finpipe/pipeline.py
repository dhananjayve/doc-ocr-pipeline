"""Ingestion orchestrator.

Throughput design:
  1. Page triage (pypdfium2, milliseconds per page) runs in the main process.
  2. Every document becomes page shards grouped by kind; all shards from all documents go into one
     process pool, so a 300-page annual report and 50 bank statements parallelise the same way.
  3. As soon as all shards of a document finish, the main process chunks, embeds and upserts it
     while the pool keeps parsing the rest (parse and embed overlap).
  4. Parsed markdown is cached by content hash + parser config; unchanged files are never re-parsed,
     and files already in the vector store are skipped entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from finpipe.chunker import MarkdownChunker
from finpipe.config import Settings
from finpipe.detect import PageKind, Shard, classify_pdf, plan_shards
from finpipe.parser import ShardResult, assemble_markdown, init_worker, parse_shard

log = logging.getLogger(__name__)


@dataclass
class DocReport:
    path: str
    doc_id: str
    status: str  # "ingested" | "skipped" | "cached" | "failed"
    pages: int = 0
    kinds: dict[str, int] = field(default_factory=dict)
    chunks: int = 0
    seconds: float = 0.0
    error: str | None = None


def file_hash(path: Path) -> str:
    st = path.stat()
    return _hash(str(path.resolve()), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=4096)
def _hash(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:24]


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}


def image_to_pdf(src: Path, out_dir: Path) -> Path:
    """Wrap a scan/photo (multi-page TIFF included) in an image-only PDF, which routes it to full OCR.

    The output is named by the image's content hash, so re-uploading the same image keeps the same doc_id.
    """
    from PIL import Image, ImageOps, ImageSequence

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{file_hash(src)}.pdf"
    if out.exists():
        return out
    with Image.open(src) as im:
        dpi = im.info.get("dpi", (200, 200))[0] or 200
        frames = [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(im)]
    tmp = out.with_suffix(".tmp")
    frames[0].save(tmp, "PDF", resolution=float(dpi), save_all=True, append_images=frames[1:])
    tmp.replace(out)
    return out


def _runs(pages: list[int]) -> list[tuple[int, int]]:
    """[3, 4, 5, 9] -> [(3, 5), (9, 9)]"""
    runs: list[tuple[int, int]] = []
    for p in sorted(pages):
        if runs and p == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], p)
        else:
            runs.append((p, p))
    return runs


def discover(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED_SUFFIXES))
        elif p.suffix.lower() in SUPPORTED_SUFFIXES and p.exists():
            found.append(p)
        else:
            log.warning("skipping %s (unsupported type or missing)", p)
    return list(dict.fromkeys(found))


class ParseCache:
    def __init__(self, root: Path, fingerprint: str):
        self.root = root
        self.fingerprint = fingerprint
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}_{self.fingerprint}.json"

    def get(self, doc_id: str) -> dict | None:
        p = self._path(doc_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, doc_id: str, payload: dict) -> None:
        tmp = self._path(doc_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(doc_id))


class Pipeline:
    def __init__(self, settings: Settings, embedder=None, store=None, persistent_pool: bool = False):
        self.s = settings
        self.persistent_pool = persistent_pool  # servers keep workers (and their loaded models) alive
        self._pool: ProcessPoolExecutor | None = None
        self.cache = ParseCache(settings.cache_dir, settings.parser_fingerprint())
        self.chunker = MarkdownChunker(settings.chunk_max_tokens, settings.chunk_min_tokens, settings.table_max_tokens)
        self._embedder = embedder
        self._store = store
        self._init_lock = threading.Lock()  # server threads may ask for these at the same time

    # lazily built so `finpipe parse` never loads the embedding model or the DB
    @property
    def embedder(self):
        with self._init_lock:
            if self._embedder is None:
                from finpipe.embed import SentenceTransformerEmbedder

                self._embedder = SentenceTransformerEmbedder(self.s)
        return self._embedder

    @property
    def facts(self):
        from finpipe.facts import FactsStore

        return FactsStore(self.s.data_dir / "facts")

    @property
    def store(self):
        with self._init_lock:
            if self._store is None:
                from finpipe.store import VectorStore

                self._store = VectorStore(self.s.chroma_dir, self.s.collection)
        return self._store

    def _executor(self, n_jobs: int) -> ProcessPoolExecutor:
        if self.persistent_pool:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.s.resolved_workers(), initializer=init_worker, initargs=(self.s,)
                )
            return self._pool
        workers = min(self.s.resolved_workers(), n_jobs)
        return ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(self.s,))

    def warmup(self) -> None:
        """Start the persistent pool and load Docling models in every worker (non-blocking)."""
        from finpipe.parser import warmup

        pool = self._executor(1)
        for _ in range(self.s.resolved_workers()):
            pool.submit(warmup)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    # -------------------------------------------------------------------------
    def parse_many(self, paths: list[Path], on_parsed) -> list[DocReport]:
        """Parse PDFs in parallel; `on_parsed(report, markdown)` is called as each document completes."""
        reports: list[DocReport] = []
        pending: dict[str, dict] = {}
        jobs: list[tuple[str, str, Shard]] = []
        seen: dict[str, str] = {}

        for path in paths:
            t0 = time.perf_counter()
            doc_id = file_hash(path)
            report = DocReport(str(path), doc_id, "ingested")
            reports.append(report)
            if doc_id in seen:
                report.status, report.error = "skipped", f"duplicate of {seen[doc_id]}"
                continue
            seen[doc_id] = path.name
            if (cached := self.cache.get(doc_id)) is not None:
                report.status, report.pages, report.kinds = "cached", cached["pages"], cached["kinds"]
                report.seconds = time.perf_counter() - t0
                on_parsed(report, cached["markdown"])
                continue
            try:
                kinds = classify_pdf(path, self.s)
            except Exception as exc:
                report.status, report.error = "failed", f"unreadable PDF: {exc}"
                continue
            if not kinds:
                report.status, report.error = "failed", "PDF has no pages"
                continue
            report.pages = len(kinds)
            report.kinds = {k.value: kinds.count(k) for k in PageKind if kinds.count(k)}
            shards = plan_shards(kinds, self.s.max_pages_per_shard, self.s.resolved_workers())
            pending[doc_id] = {"report": report, "left": len(shards), "pages": [], "errors": [], "t0": t0}
            jobs.extend((str(path), doc_id, sh) for sh in shards)
            log.info("%s: %d pages %s -> %d shard(s)", path.name, len(kinds), report.kinds, len(shards))

        if not jobs:
            return reports

        # Scanned shards are the slowest; start them first so they don't form the tail.
        cost = {PageKind.SCANNED: 8, PageKind.HYBRID: 4, PageKind.DIGITAL: 1}
        jobs.sort(key=lambda j: cost[j[2].kind] * j[2].pages, reverse=True)
        log.info("parsing %d shard(s) on up to %d worker(s) x %d thread(s)",
                 len(jobs), self.s.resolved_workers(), self.s.resolved_threads())

        pool = self._executor(len(jobs))
        paths_by_doc = {doc_id: path for path, doc_id, _ in jobs}
        try:
            running: set[Future] = {pool.submit(parse_shard, *job) for job in jobs}
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    res: ShardResult = fut.result()
                    state = pending[res.doc_id]
                    state["left"] -= 1
                    if res.error:
                        state["errors"].append(f"pages {res.shard.start}-{res.shard.end}: {res.error}")
                    state["pages"].extend(res.pages)
                    # Pages the fast route couldn't read confidently go back to Docling.
                    for first, last in _runs(res.fallback):
                        shard = Shard(res.shard.kind, first, last, engine="docling")
                        running.add(pool.submit(parse_shard, paths_by_doc[res.doc_id], res.doc_id, shard))
                        state["left"] += 1
                        state["report"].kinds["docling-fallback"] = (
                            state["report"].kinds.get("docling-fallback", 0) + last - first + 1
                        )
                    if state["left"] == 0:
                        self._finish_parse(res.doc_id, state, on_parsed)
        finally:
            if not self.persistent_pool:
                pool.shutdown()
        return reports

    def _finish_parse(self, doc_id: str, state: dict, on_parsed) -> None:
        report: DocReport = state["report"]
        report.seconds = time.perf_counter() - state["t0"]
        if not state["pages"]:
            report.status, report.error = "failed", "; ".join(state["errors"]) or "no content extracted"
            return
        if state["errors"]:
            report.error = "partial: " + "; ".join(state["errors"])
        markdown = assemble_markdown(state["pages"])
        if not state["errors"]:  # never cache a partial parse
            self.cache.put(doc_id, {"markdown": markdown, "pages": report.pages, "kinds": report.kinds})
        on_parsed(report, markdown)

    # -------------------------------------------------------------------------
    def ingest(
        self, paths: list[str | Path], force: bool = False, category: str | None = None,
        doc_type: str | None = None, source_names: dict[str, str] | None = None,
    ) -> list[DocReport]:
        """Parse and index files. `doc_type` (see doctypes.py) tags every chunk; `source_names` maps a
        file path to the display name stored as `source` (e.g. the original name of an upload)."""
        from finpipe import doctypes

        dtype = doctypes.get(doc_type or category)
        base_meta = {"doc_type": dtype.key, "category": category or dtype.category}
        names = {str(Path(k)): v for k, v in (source_names or {}).items()}
        originals: dict[str, str] = {}
        files = []
        for path in discover(paths):
            name = names.get(str(path), path.name)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    path = image_to_pdf(path, self.s.data_dir / "converted")
                except Exception as exc:
                    log.error("cannot read image %s: %s", path, exc)
                    continue
            originals[str(path)] = name
            files.append(path)

        todo, skipped = [], []
        for path in files:
            doc_id = file_hash(path)
            if self.store.has_doc(doc_id):
                if not force:
                    # Already indexed: just apply the (possibly new) category/type tags.
                    self.store.retag(doc_id, base_meta | {"source": originals[str(path)]})
                    skipped.append(DocReport(originals[str(path)], doc_id, "skipped"))
                    continue
                self.store.delete_doc(doc_id)
            todo.append(path)

        def index(report: DocReport, markdown: str) -> None:
            t0 = time.perf_counter()
            chunks = self.chunker.chunk(markdown)
            if not chunks:
                report.status, report.error = "failed", "no chunks produced"
                return
            from finpipe.facts import extract_facts, section_for_page

            source = originals.get(report.path, Path(report.path).name)
            try:
                facts = extract_facts(markdown)
            except Exception:  # facts are a bonus; never fail an ingest over them
                log.exception("facts extraction failed for %s", source)
                facts = None
            extra = []
            for c in chunks:
                sec = section_for_page(facts, c.page_start) or {}
                tags = {}
                if sec.get("account_number"):
                    tags["account"] = f"{sec.get('bank') or 'Bank'} {sec['account_number']}"
                if sec.get("assessment_year"):
                    tags["year"] = sec["assessment_year"]
                    tags["return"] = f"{sec.get('form', 'ITR')} AY {sec['assessment_year']}"
                extra.append(tags)
            try:
                vectors = self.embedder.embed_documents([
                    (f"[{t.get('account') or t.get('return')}] " if t else "") + c.embed_text()
                    for c, t in zip(chunks, extra)
                ])
                self.store.add_chunks(report.doc_id, chunks, vectors, base_meta | {"source": source}, extra)
                self.facts.put(report.doc_id, source, facts)
            except Exception as exc:
                log.exception("indexing %s failed", report.path)
                report.status, report.error = "failed", f"indexing failed: {exc}"
                return
            report.chunks = len(chunks)
            report.seconds += time.perf_counter() - t0
            log.info("indexed %s: %d chunks", Path(report.path).name, len(chunks))

        reports = self.parse_many(todo, index)
        for r in reports:
            r.path = originals.get(r.path, r.path)
        return skipped + reports

    def parse_to_markdown(self, path: str | Path) -> tuple[DocReport, str]:
        path = Path(path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            path = image_to_pdf(path, self.s.data_dir / "converted")
        out: dict[str, str] = {}
        reports = self.parse_many([path], lambda r, md: out.__setitem__(r.doc_id, md))
        report = reports[0]
        return report, out.get(report.doc_id, "")
