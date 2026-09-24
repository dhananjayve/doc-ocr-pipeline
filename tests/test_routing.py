from pathlib import Path

import pypdfium2 as pdfium
from fastapi.testclient import TestClient
from PIL import Image

from finpipe import doctypes
from finpipe.chunker import MarkdownChunker
from finpipe.detect import PageKind, classify_pdf
from finpipe.pipeline import image_to_pdf
from finpipe.rag import RAG, build_where, system_prompt
from finpipe.retrieval import Retriever
from finpipe.server import create_app
from finpipe.store import VectorStore


class FakeModel:
    def __init__(self, name):
        self.name, self.calls = name, []

    def stream(self, messages):
        self.calls.append(messages)
        yield f"answer from {self.name}"

    def chat(self, messages):
        return "".join(self.stream(messages))


def _index(store, embedder, doc_id, md, doc_type):
    t = doctypes.get(doc_type)
    chunks = MarkdownChunker(max_tokens=200, min_tokens=5).chunk(md)
    store.add_chunks(doc_id, chunks, embedder.embed_documents([c.embed_text() for c in chunks]),
                     {"source": f"{doc_id}.pdf", "doc_type": t.key, "category": t.category})


def test_doctype_registry():
    cats = {g["category"]: [t["key"] for t in g["types"]] for g in doctypes.as_options()}
    assert {"bank_statement", "ais", "form16"} <= set(cats["finance"])
    assert {"sale_deed", "land_record", "lease"} <= set(cats["property"])
    assert cats["other"] == ["other"]
    assert doctypes.get("nonsense").key == "other"


def test_build_where():
    assert build_where(category=None, doc_id="") is None
    assert build_where(category="finance") == {"category": "finance"}
    assert build_where(category="finance", doc_id="x") == {"$and": [{"category": "finance"}, {"doc_id": "x"}]}


def test_finance_questions_route_to_finance_model(settings, embedder):
    store = VectorStore(settings.chroma_dir, "route")
    _index(store, embedder, "hdfc", "# HDFC Bank statement\n\nClosing balance 45,210.50 savings account", "bank_statement")
    _index(store, embedder, "lease", "# Lease deed\n\nMonthly rent 32,000 lock-in eleven months tenant", "lease")
    default, fin = FakeModel("default"), FakeModel("fingpt")
    loads = []
    rag = RAG(Retriever(settings, store, embedder), default,
              finance_slm=lambda: loads.append(1) or fin, finance_context_tokens=500)

    a = rag.ask("monthly rent lock-in tenant lease")
    assert a.route == "default" and a.text == "answer from default" and not loads  # finance model not loaded

    a = rag.ask("closing balance savings account bank statement")
    assert a.route == "finance" and a.text == "answer from fingpt"
    assert "several banks" in fin.calls[0][0]["content"]  # bank statement hint injected

    a = rag.ask("closing balance", category="property")
    assert a.route == "default"
    assert all(h.metadata["category"] == "property" for h in a.sources)
    rag.ask("rent", category="finance")
    assert len(loads) == 1  # factory called once, model reused


def test_system_prompt_without_hints_is_base():
    assert "Document notes" not in system_prompt([])


def test_image_to_pdf_routes_to_ocr(tmp_path, settings):
    img = tmp_path / "scan.jpg"
    Image.new("RGB", (1200, 1600), "white").save(img, dpi=(300, 300))
    pdf = image_to_pdf(img, tmp_path / "conv")
    assert pdf == image_to_pdf(img, tmp_path / "conv")  # stable name -> stable doc_id
    assert len(pdfium.PdfDocument(str(pdf))) == 1
    assert classify_pdf(pdf, settings) == [PageKind.SCANNED]


def test_api_meta_and_upload_validation(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/").status_code == 200
        meta = client.get("/api/meta").json()
        assert [c["category"] for c in meta["categories"]] == ["finance", "property", "other"]
        assert ".pdf" in meta["accept"] and ".jpg" in meta["accept"]
        assert meta["finance_model"] is None

        r = client.post("/api/documents", data={"doc_type": "form16"},
                        files={"files": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 400 and "unsupported" in r.json()["detail"]
        r = client.post("/api/documents", data={"doc_type": "bogus"},
                        files={"files": ("a.pdf", b"%PDF", "application/pdf")})
        assert r.status_code == 400
        assert client.get("/api/documents").json() == []


def test_finance_settings(settings):
    assert settings.finance_settings() is None
    s = settings.model_copy(update={"finance_slm_backend": "hf", "finance_slm_model": "meta-llama/Llama-2-7b-chat-hf",
                                    "finance_slm_adapter": "FinGPT/fingpt-mt_llama2-7b_lora"})
    fin = s.finance_settings()
    assert fin.slm_backend == "hf" and fin.slm_adapter.startswith("FinGPT/")
    assert fin.context_max_tokens == s.finance_context_max_tokens
    assert s.slm_model == "qwen3:8b"  # default route unchanged
