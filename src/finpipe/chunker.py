"""Markdown-aware chunking for financial documents.

- Splits on headings, never inside a table: a balance sheet or cash-flow statement stays one unit.
- A table that exceeds `table_max_tokens` is split by rows, repeating its header on every piece.
- Every chunk carries its heading breadcrumb and page span, so the SLM sees where numbers came from.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
PAGE_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


TOKEN_PIECE_RE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")
# number runs not attached to letters ("1,25,450.75", "30/01/24", "000123456789"), table pipes, rules
EMBED_STRIP_RE = re.compile(r"(?<![A-Za-z])\d[\d,./:\-]*(?![A-Za-z])|\|+|-{2,}")


def est_tokens(text: str) -> int:
    """Token estimate for Gemma/Qwen/Llama-3 style tokenizers.

    Those tokenizers give every digit its own token, so digit-heavy financial text costs far more than a
    chars/4 rule suggests (a bank-statement prompt measured 8,016 real tokens vs 3,158 by chars/3.5).
    Counting digits and punctuation as one token each and letter runs as ~4 chars/token lands within
    ~5% on statements and ITRs, and slightly over on prose.
    """
    return sum(math.ceil(len(p) / 4) if p[0].isalpha() else 1 for p in TOKEN_PIECE_RE.findall(text))


@dataclass
class Block:
    kind: str  # "heading" | "table" | "text"
    text: str
    page: int
    level: int = 0


@dataclass
class Chunk:
    text: str
    heading_path: list[str]
    page_start: int
    page_end: int
    has_table: bool
    index: int = 0
    metadata: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """Text given to the embedding model (the stored and displayed text is always `text`).

        For table chunks, runs of digits (amounts, balances, dates, reference numbers) and table
        punctuation are dropped: they carry little meaning for vector search, cost most of the tokens,
        and pushed half of all statement chunks past the embedding model's 512-token limit, silently
        truncating them. Payees, narrations, labels and headings are kept; exact numbers are matched by
        the keyword (BM25) stage, which sees the full text.
        """
        crumb = " > ".join(self.heading_path)
        body = self.text
        if self.has_table:
            body = re.sub(r"\s+", " ", EMBED_STRIP_RE.sub(" ", body)).strip()
        return f"[{crumb}]\n{body}" if crumb else body


def parse_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    page = 1
    buf: list[str] = []
    buf_kind = ""

    def flush():
        nonlocal buf, buf_kind
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(buf_kind, text, page))
        buf, buf_kind = [], ""

    for raw in markdown.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if m := PAGE_RE.match(stripped):
            flush()
            page = int(m.group(1))
            continue
        if m := HEADING_RE.match(stripped):
            flush()
            blocks.append(Block("heading", m.group(2).strip(), page, level=len(m.group(1))))
            continue
        kind = "table" if stripped.startswith("|") else "text"
        if not stripped:
            if buf_kind == "text":
                flush()
            continue  # blank lines inside a table (page-break artefacts) don't end it
        if kind != buf_kind:
            flush()
            buf_kind = kind
        buf.append(line)
    flush()
    return blocks


def split_table(table: str, max_tokens: int) -> list[str]:
    """Row-split an oversized table, repeating header + separator rows on each piece."""
    lines = table.splitlines()
    header_end = 0
    for i, line in enumerate(lines[:4]):
        if TABLE_SEP_RE.match(line.strip()):
            header_end = i + 1
            break
    header, rows = lines[:header_end], lines[header_end:]
    budget = max(max_tokens - est_tokens("\n".join(header)), 1)
    pieces, cur, cur_tokens = [], [], 0
    for row in rows:
        t = est_tokens(row) + 1
        if cur and cur_tokens + t > budget:
            pieces.append("\n".join(header + cur))
            cur, cur_tokens = [], 0
        cur.append(row)
        cur_tokens += t
    if cur or not pieces:
        pieces.append("\n".join(header + cur))
    return pieces


class MarkdownChunker:
    def __init__(self, max_tokens: int = 700, min_tokens: int = 150, table_max_tokens: int = 1800):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.table_max_tokens = table_max_tokens

    def chunk(self, markdown: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        path: list[tuple[int, str]] = []  # (level, title)
        cur: list[Block] = []
        cur_path: list[str] = []
        cur_tokens = 0

        def emit(blocks: list[Block], heading_path: list[str]):
            body = [b for b in blocks if b.kind != "heading"]
            if not body:
                return
            text = "\n\n".join(
                ("#" * b.level + " " + b.text) if b.kind == "heading" else b.text for b in blocks
            )
            chunks.append(Chunk(
                text=text,
                heading_path=heading_path,
                page_start=min(b.page for b in blocks),
                page_end=max(b.page for b in blocks),
                has_table=any(b.kind == "table" for b in blocks),
                index=len(chunks),
            ))

        def flush():
            nonlocal cur, cur_tokens
            emit(cur, cur_path)
            cur, cur_tokens = [], 0

        for block in parse_blocks(markdown):
            if block.kind == "heading":
                # Small sections are packed with their successors instead of becoming tiny chunks.
                if cur_tokens >= self.min_tokens:
                    flush()
                while path and path[-1][0] >= block.level:
                    path.pop()
                path.append((block.level, block.text))
                if all(b.kind == "heading" for b in cur):  # no body yet: breadcrumb follows nesting
                    cur_path = [t for _, t in path]
                cur.append(block)
                cur_tokens += est_tokens(block.text)
                continue

            tokens = est_tokens(block.text)
            if block.kind == "table" and tokens > self.max_tokens:
                # A table gets its own chunk(s): keep the preceding heading/lead-in only if it fits.
                pieces = (
                    [block.text] if tokens <= self.table_max_tokens
                    else split_table(block.text, self.table_max_tokens)
                )
                lead = cur if cur_tokens + est_tokens(pieces[0]) <= self.table_max_tokens else []
                if not lead:
                    flush()
                lead_path = cur_path if lead else [t for _, t in path]
                for i, piece in enumerate(pieces):
                    emit((lead if i == 0 else []) + [Block("table", piece, block.page)], lead_path)
                cur, cur_tokens = [], 0
                cur_path = [t for _, t in path]
                continue

            if cur and cur_tokens + tokens > self.max_tokens:
                flush()
                cur_path = [t for _, t in path]
            if block.kind == "text" and tokens > self.max_tokens:
                for piece in self._split_text(block.text):
                    emit([Block("text", piece, block.page)], [t for _, t in path])
                continue
            if not cur:
                cur_path = [t for _, t in path]
            cur.append(block)
            cur_tokens += tokens

        flush()
        return chunks

    def _split_text(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces, cur = [], ""
        for s in sentences:
            if cur and est_tokens(cur + " " + s) > self.max_tokens:
                pieces.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            pieces.append(cur)
        return pieces
