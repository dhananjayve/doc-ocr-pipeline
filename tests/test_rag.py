import json

import httpx

from finpipe.chunker import MarkdownChunker
from finpipe.grounding import normalize, unverified_numbers
from finpipe.rag import RAG
from finpipe.retrieval import Retriever, bm25_scores
from finpipe.slm import SLMClient, ThinkFilter
from finpipe.store import VectorStore


def test_think_filter_across_chunk_boundaries():
    f = ThinkFilter()
    stream = ["<thi", "nk>secret reasoning</th", "ink>Revenue was ", "$5,000.08", " <", "b>"]
    out = "".join(f.feed(s) for s in stream) + f.flush()
    assert out == "Revenue was $5,000.08 <b>"


def test_normalize_financial_numbers():
    assert normalize("$(1,234.50)") == -1234.5
    assert normalize("12.5%") == 12.5
    assert normalize("-3,000") == -3000


def test_unverified_numbers():
    ctx = "| Net income | (1,200) | 3,450.75 |\nFiscal 2024"
    ans = "Net income was $(1,200) in fiscal 2024 [1], versus 3,450.75 [2]; margin 17.3%."
    assert unverified_numbers(ans, ctx) == ["17.3%"]


def test_bm25_prefers_exact_terms():
    docs = ["diluted EPS was 4.12", "revenue grew strongly", "basic earnings per share"]
    scores = bm25_scores("diluted EPS", docs)
    assert scores[0] == max(scores) and scores[1] == 0


def _ollama_transport(captured):
    def handler(request: httpx.Request):
        captured.append(json.loads(request.content))
        lines = [
            {"message": {"content": "<think>hmm</think>"}, "done": False},
            {"message": {"content": "Total revenue was 9,876.5 [1]."}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))
    return httpx.MockTransport(handler)


def test_end_to_end_rag(settings, embedder):
    md = (
        "<!-- page: 3 -->\n\n# Income Statement\n\n| Item | FY2024 |\n|---|---:|\n| Total revenue | 9,876.5 |\n"
        "| Net income | (321.0) |\n\n<!-- page: 4 -->\n\n# Risk Factors\n\nCompetition and interest rates."
    )
    store = VectorStore(settings.chroma_dir, "test")
    chunks = MarkdownChunker(max_tokens=60, min_tokens=5).chunk(md)
    store.add_chunks("doc1", chunks, embedder.embed_documents([c.embed_text() for c in chunks]), {"source": "10k.pdf"})
    assert store.has_doc("doc1") and store.count() == len(chunks)

    captured = []
    rag = RAG(Retriever(settings, store, embedder), SLMClient(settings, transport=_ollama_transport(captured)))
    answer = rag.ask("What was total revenue in the income statement?", k=2)

    assert answer.text == "Total revenue was 9,876.5 [1]."
    assert answer.unverified == []
    assert answer.sources[0].metadata["page_start"] == 3
    body = captured[0]
    assert body["model"] == settings.slm_model and body["options"]["num_ctx"] == settings.slm_num_ctx
    assert "9,876.5" in body["messages"][1]["content"]

    store.delete_doc("doc1")
    assert store.count() == 0
