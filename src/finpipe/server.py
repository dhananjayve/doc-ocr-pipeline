"""Web API + single-page UI.

    finpipe serve            -> http://127.0.0.1:8765

Uploads are queued and ingested one batch at a time (each batch is already parallel inside the
pipeline). Parser workers exit after each batch: digital pages don't need Docling at all, and freeing
the workers' memory (~2 GB each with Docling loaded) leaves room for the language model.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from finpipe import doctypes
from finpipe.config import Settings
from finpipe.pipeline import SUPPORTED_SUFFIXES, Pipeline

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_MB = 200


class AskRequest(BaseModel):
    question: str
    category: str | None = None
    doc_type: str | None = None
    doc_id: str | None = None
    year: str | None = None
    k: int | None = None
    use_model: bool = False  # skip the instant facts answer and ask the model


class State:
    def __init__(self, settings: Settings):
        self.s = settings
        self.pipeline = Pipeline(settings)
        self.queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self.jobs: dict[str, dict] = {}
        self._rag = None
        self._lock = threading.Lock()

    @property
    def rag(self):
        with self._lock:
            if self._rag is None:
                from finpipe.rag import RAG
                from finpipe.retrieval import Retriever
                from finpipe.slm import make_slm

                fin = self.s.finance_settings()
                retriever = Retriever(self.s, self.pipeline.store, self.pipeline.embedder)
                self._rag = RAG(
                    retriever, make_slm(self.s), self.s.context_max_tokens,
                    finance_slm=(lambda: make_slm(fin)) if fin else None,
                    finance_context_tokens=self.s.finance_context_max_tokens,
                    facts_lookup=self.pipeline.facts.get,
                    facts_all=self.pipeline.facts.all,
                )
            return self._rag

    def warmup(self) -> None:
        """Load the embedding model, the index and the language model in the background at startup,
        so the first question doesn't wait ~30-60 s for them."""
        t0 = time.perf_counter()
        try:
            rag = self.rag
            rag.retriever.embedder.embed_query("warm up")
            rag.retriever.store.count()
            log.info("retrieval ready in %.1fs", time.perf_counter() - t0)
            rag.slm.warmup()
            log.info("language model %s loaded in %.1fs", self.s.slm_model, time.perf_counter() - t0)
        except Exception as exc:  # the first question will simply load them itself
            log.warning("warm-up failed: %s", exc)

    def run_job(self, job_id: str, paths: list[Path], names: dict[str, str], doc_type: str, force: bool):
        job = self.jobs[job_id]
        job["status"], job["started"] = "processing", time.time()
        try:
            reports = self.pipeline.ingest(paths, force=force, doc_type=doc_type, source_names=names)
            job["reports"] = [asdict(r) for r in reports]
            job["status"] = "failed" if all(r.status == "failed" for r in reports) else "done"
        except Exception as exc:
            log.exception("job %s failed", job_id)
            job["status"], job["error"] = "failed", str(exc)
        job["finished"] = time.time()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state: State | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal state
        state = State(settings)
        threading.Thread(target=state.warmup, name="warmup", daemon=True).start()
        yield
        state.queue.shutdown(wait=False, cancel_futures=True)
        state.pipeline.close()

    app = FastAPI(title="finpipe", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/meta")
    def meta():
        fin = settings.finance_settings()
        return {
            "categories": doctypes.as_options(),
            "accept": sorted(SUPPORTED_SUFFIXES),
            "default_model": settings.slm_model,
            "finance_model": (fin.slm_adapter or fin.slm_model) if fin else None,
        }

    @app.get("/api/documents")
    def documents():
        from finpipe.facts import render, summary

        docs = state.pipeline.store.documents()
        for d in docs:
            entry = state.pipeline.facts.get(d["doc_id"]) or {}
            facts = entry.get("facts")
            d["summary"] = summary(facts)
            d["facts_text"] = render(facts, d["source"]) if facts else ""
            d["years"] = sorted({r["assessment_year"] for r in (facts or {}).get("returns", [])
                                 if r.get("assessment_year")})
        return docs

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str):
        state.pipeline.store.delete_doc(doc_id)
        state.pipeline.facts.delete(doc_id)
        return {"deleted": doc_id}

    @app.post("/api/documents", status_code=202)
    async def upload(
        files: list[UploadFile] = File(...),
        doc_type: str = Form("other"),
        force: bool = Form(False),
    ):
        if doc_type not in doctypes.DOC_TYPES:
            raise HTTPException(400, f"unknown doc_type {doc_type!r}")
        upload_dir = settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths, names = [], {}
        for f in files:
            name = Path(f.filename or "upload").name
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(400, f"{name}: unsupported file type (allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))})")
            dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            size = 0
            with open(dest, "wb") as out:
                while chunk := await f.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        out.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, f"{name}: larger than {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
            paths.append(dest)
            names[str(dest)] = re.sub(r"[^\w.\- ()]", "_", name)

        job_id = uuid.uuid4().hex[:12]
        state.jobs[job_id] = {
            "id": job_id, "status": "queued", "files": list(names.values()), "doc_type": doc_type,
            "category": doctypes.get(doc_type).category, "created": time.time(), "reports": [],
        }
        state.queue.submit(state.run_job, job_id, paths, names, doc_type, force)
        return state.jobs[job_id]

    @app.get("/api/jobs")
    def jobs():
        return sorted(state.jobs.values(), key=lambda j: j["created"], reverse=True)[:50]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in state.jobs:
            raise HTTPException(404, "no such job")
        return state.jobs[job_id]

    @app.post("/api/ask")
    def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(400, "empty question")
        from finpipe.rag import Answer, Prepared, facts_quick, source_label

        filters = {"doc_type": req.doc_type, "doc_id": req.doc_id, "year": req.year}
        if not req.use_model:  # instant answer from facts: don't wait for models that are still loading
            text = facts_quick(req.question, state.pipeline.facts.get, state.pipeline.facts.all,
                               req.category, req.doc_id, req.year)
            if text:
                body = json.dumps({"type": "answer", "text": text, "route": "facts",
                                   "model": "document facts · no model", "unverified": [], "sources": []})
                return StreamingResponse(iter([body + "\n"]), media_type="application/x-ndjson")
        rag = state.rag

        def events():
            try:
                stream = rag.ask_stream(req.question, k=req.k, category=req.category, use_model=req.use_model,
                                        **filters)
                for piece in stream:
                    if isinstance(piece, Prepared):
                        yield json.dumps({"type": "status", "sources": piece.sources,
                                          "context_tokens": piece.context_tokens, "route": piece.route}) + "\n"
                    elif isinstance(piece, Answer):
                        yield json.dumps({
                            "type": "answer",
                            "text": piece.text if not piece.sources else None,
                            "route": piece.route,
                            "model": "document facts · no model" if piece.route == "facts" else
                                     (settings.finance_slm_adapter or settings.finance_slm_model)
                                     if piece.route == "finance" else settings.slm_model,
                            "unverified": piece.unverified,
                            "sources": [
                                {"label": source_label(h), "score": round(h.score, 3), "text": h.text,
                                 "doc_type": h.metadata.get("doc_type")}
                                for h in piece.sources
                            ],
                        }) + "\n"
                    else:
                        yield json.dumps({"type": "token", "text": piece}) + "\n"
            except Exception as exc:
                log.exception("ask failed")
                yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
