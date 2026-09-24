"""Question answering over ingested documents with a small language model.

Routing: when a finance model (e.g. FinGPT) is configured, questions scoped to the finance category,
or unscoped questions whose best source is a finance document, go to it; everything else goes to the
default SLM. Type-specific instructions (bank statement, Form 16, lease, ...) are added to the prompt
for the document types present in the retrieved sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from finpipe import doctypes
from finpipe.chunker import est_tokens
from finpipe.grounding import unverified_numbers
from finpipe.retrieval import Retriever
from finpipe.slm import ChatModel
from finpipe.store import Hit

SYSTEM_PROMPT = """You are a meticulous financial and legal document analyst. Answer strictly from the numbered SOURCES.

Rules:
- Quote figures exactly as written in the sources, including units, currency and scale (e.g. "in lakhs").
  Never add a currency symbol the source doesn't show (Indian documents are in ₹/INR, never "$").
- When sources come from different accounts, banks, years or documents, say which one each figure belongs to.
- Parentheses in financial tables mean negative values: (1,234) = -1,234.
- Read tables row by row; confirm the column (period/year/account) before reporting a figure.
- If you compute something (growth, ratio, sum), show the formula and the source figures.
- Cite sources inline like [1] or [2][3]; cite the DOCUMENT FACTS block as [F]. Prefer [F] for totals,
  counts, closing balances and per-year figures, and passages for individual transactions or lines.
- If the sources do not contain the answer, say so. Never guess or use outside knowledge.
- Be concise: lead with the direct answer. Put lists of accounts, years or transactions in one compact
  markdown table instead of repeated bullet labels, shorten long transaction descriptions, and don't
  restate these rules or the sources."""


@dataclass
class Prepared:
    """First event of a streamed answer: what the model is about to read."""
    sources: int
    context_tokens: int
    route: str


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    unverified: list[str] = field(default_factory=list)
    route: str = "default"


def build_where(**filters) -> dict | None:
    """Chroma metadata filter from keyword filters, ignoring empty values."""
    clauses = [{k: v} for k, v in filters.items() if v not in (None, "", "all")]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def source_label(hit: Hit) -> str:
    m = hit.metadata
    pages = f"p.{m['page_start']}" if m["page_start"] == m["page_end"] else f"pp.{m['page_start']}-{m['page_end']}"
    section = m.get("account") or m.get("return")
    section = f" | {section}" if section else ""
    heading = f" | {m['heading']}" if m.get("heading") else ""
    return f"{m.get('source', '?')} {pages}{section}{heading}"


def build_context(hits: list[Hit], max_tokens: int) -> tuple[str, list[Hit]]:
    parts, used, budget = [], [], max_tokens
    for hit in hits:
        block = f"[{len(used) + 1}] {source_label(hit)}\n{hit.text}"
        cost = est_tokens(block)
        if used and cost > budget:
            break
        parts.append(block)
        used.append(hit)
        budget -= cost
    return "\n\n---\n\n".join(parts), used


def system_prompt(hits: list[Hit]) -> str:
    notes = []
    for key in dict.fromkeys(h.metadata.get("doc_type") for h in hits):
        t = doctypes.get(key)
        if t.hint:
            notes.append(f"- {t.label}: {t.hint}")
    return SYSTEM_PROMPT + ("\n\nDocument notes:\n" + "\n".join(notes) if notes else "")


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        slm: ChatModel,
        context_max_tokens: int = 8000,
        finance_slm: ChatModel | Callable[[], ChatModel] | None = None,
        finance_context_tokens: int = 2500,
        facts_lookup: Callable[[str], dict | None] | None = None,
        facts_all: Callable[[], dict[str, dict]] | None = None,
    ):
        self.facts_lookup = facts_lookup  # doc_id -> {"source", "facts"} (see facts.FactsStore.get)
        self.facts_all = facts_all        # () -> {doc_id: {"source", "facts"}} for instant answers
        self.retriever = retriever
        self.slm = slm
        self.context_max_tokens = context_max_tokens
        self._finance = finance_slm  # a model, or a factory so heavy local models load on first use
        self.finance_context_tokens = finance_context_tokens

    @property
    def has_finance_route(self) -> bool:
        return self._finance is not None

    def _finance_model(self) -> ChatModel:
        if callable(self._finance) and not hasattr(self._finance, "stream"):
            self._finance = self._finance()
        return self._finance  # type: ignore[return-value]

    def _route(self, category: str | None, hits: list[Hit]) -> str:
        if not self.has_finance_route:
            return "default"
        if category == "finance":
            return "finance"
        if not category and hits and hits[0].metadata.get("category") == "finance":
            return "finance"
        return "default"

    def prepare(self, question: str, k: int | None = None, category: str | None = None, **filters):
        hits = self.retriever.retrieve(question, k=k, where=build_where(category=category, **filters))
        route = self._route(category, hits)
        budget = self.finance_context_tokens if route == "finance" else self.context_max_tokens
        facts = self._facts_block(hits, filters.get("doc_id"), filters.get("year"))
        context, used = build_context(hits, budget - est_tokens(facts))
        if facts:
            context = f"[F] DOCUMENT FACTS (read directly from the documents at upload)\n{facts}\n\n---\n\n{context}"
        messages = [
            {"role": "system", "content": system_prompt(used)},
            {"role": "user", "content": f"SOURCES:\n\n{context}\n\nQUESTION: {question}"},
        ]
        return messages, context, used, route

    def _facts_block(self, hits: list[Hit], doc_id: str | None, year: str | None = None) -> str:
        """Facts for the documents in play: the one asked about, else those the passages came from."""
        from finpipe.facts import render

        if not self.facts_lookup:
            return ""
        ids = [doc_id] if doc_id else list(dict.fromkeys(h.metadata.get("doc_id") for h in hits))
        blocks = []
        for i in ids[:3]:
            entry = self.facts_lookup(i) if i else None
            if entry and entry.get("facts"):
                block = render(entry["facts"], entry["source"], year)
                if block:
                    blocks.append(block)
        return "\n".join(blocks)

    def quick(self, question: str, category: str | None = None, doc_id: str | None = None,
              year: str | None = None, **_ignored) -> str | None:
        """Instant answer from stored facts (no model), or None if the question needs the model."""
        return facts_quick(question, self.facts_lookup, self.facts_all, category, doc_id, year)

    def _model(self, route: str) -> ChatModel:
        return self._finance_model() if route == "finance" else self.slm

    def ask(self, question: str, k: int | None = None, category: str | None = None, **filters) -> Answer:
        messages, context, used, route = self.prepare(question, k, category, **filters)
        if not used:
            return Answer("No matching documents have been ingested yet.", [])
        text = self._model(route).chat(messages)
        return Answer(text, used, unverified_numbers(text, context), route)

    def ask_stream(
        self, question: str, k: int | None = None, category: str | None = None, use_model: bool = False,
        **filters,
    ) -> Iterator[str | Prepared | Answer]:
        """Yields a Prepared event, answer text pieces, then a final Answer with sources and grounding.
        Questions the stored facts fully answer return a single Answer (route "facts") instantly,
        unless `use_model` is set."""
        if not use_model and (text := self.quick(question, category, **filters)):
            yield Answer(text, [], [], route="facts")
            return
        messages, context, used, route = self.prepare(question, k, category, **filters)
        if not used:
            yield Answer("No matching documents have been ingested yet.", [])
            return
        yield Prepared(len(used), est_tokens(messages[0]["content"] + messages[1]["content"]), route)
        pieces = []
        for piece in self._model(route).stream(messages):
            pieces.append(piece)
            yield piece
        text = "".join(pieces).strip()
        yield Answer(text, used, unverified_numbers(text, context), route)


def facts_quick(question: str, facts_lookup, facts_all, category: str | None = None,
                doc_id: str | None = None, year: str | None = None) -> str | None:
    """Instant answer from stored facts for the documents in scope; needs no model or index, so the
    server can answer before its models have finished loading."""
    from finpipe.quick import quick_answer

    if category not in (None, "", "finance"):
        return None
    if doc_id:
        entry = facts_lookup(doc_id) if facts_lookup else None
        entries = [entry] if entry else []
    else:
        entries = list(facts_all().values()) if facts_all else []
    return quick_answer(question, entries, year) if entries else None


def build_rag(settings) -> RAG:
    from finpipe.embed import SentenceTransformerEmbedder
    from finpipe.retrieval import Retriever
    from finpipe.slm import make_slm
    from finpipe.store import VectorStore

    from finpipe.facts import FactsStore

    store = VectorStore(settings.chroma_dir, settings.collection)
    retriever = Retriever(settings, store, SentenceTransformerEmbedder(settings))
    fin = settings.finance_settings()
    return RAG(
        retriever, make_slm(settings), settings.context_max_tokens,
        finance_slm=(lambda: make_slm(fin)) if fin else None,
        finance_context_tokens=settings.finance_context_max_tokens,
        facts_lookup=FactsStore(settings.data_dir / "facts").get,
        facts_all=FactsStore(settings.data_dir / "facts").all,
    )
