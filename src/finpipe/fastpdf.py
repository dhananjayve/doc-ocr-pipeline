"""Fast extraction route for digital PDF pages (no neural models).

Digital pages already carry every character and its position, and financial PDFs (bank statements,
ITR forms, government statements) draw their tables with ruling lines. So tables are rebuilt from
geometry with pdfplumber (~0.2-1 s/page) instead of Docling's TableFormer (~5-20 s/page on CPU),
with identical figures on ruled tables.

Pages this route cannot read with confidence (tabular-looking text but no ruled table) are reported
back so the pipeline can re-parse just those pages with Docling.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

NUM_RE = re.compile(r"\(?-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\)?|\(?-?\d+\.\d{2}\)?")
TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
LINE_TOL = 2.5  # points: words whose tops differ by less than this are on the same line
# Max gap (points) between letters of one word. pdfplumber's default of 3 glues words together in
# tightly set PDFs (an HDFC statement came out as "FREEPRESSJOURNALMARG"); 1.5 splits them correctly.
X_TOL = 1.5
CID_RE = re.compile(r"\(cid:\d+\)")  # glyphs without a Unicode mapping (usually tabs)


@dataclass
class PageResult:
    markdown: str
    needs_fallback: bool = False
    tables: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
def _clean(cell: str | None) -> str:
    return re.sub(r"\s+", " ", CID_RE.sub(" ", cell or "").replace("|", "/")).strip()


def _lines_in(page, bbox) -> list[tuple[float, str]]:
    """Text lines (top, text) inside a bbox, top to bottom."""
    words = page.crop(bbox, strict=False).extract_words(x_tolerance=X_TOL, keep_blank_chars=False, use_text_flow=False)
    lines: list[list] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0] - w["top"]) < LINE_TOL:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    return [(top, " ".join(x["text"] for x in sorted(ws, key=lambda x: x["x0"]))) for top, ws in lines]


def _split_merged_row(page, cells: list) -> list[list[str]] | None:
    """Split a row that holds several records (a table without horizontal rules between rows).

    The first column with several lines is the anchor (dates in a bank statement): every anchor line
    starts a record, and lines in other columns join the last record starting at or above them, so
    wrapped descriptions stay with their transaction.
    """
    col_lines = [(_lines_in(page, c) if c else []) for c in cells]
    anchor = next((i for i, ls in enumerate(col_lines) if len(ls) >= 2), None)
    if anchor is None:
        return None
    multi = sum(1 for ls in col_lines if len(ls) >= 2)
    if multi < 2:
        return None
    tops = [t for t, _ in col_lines[anchor]]
    records = [[""] * len(cells) for _ in tops]
    for ci, lines in enumerate(col_lines):
        for top, text in lines:
            ri = max((i for i, t in enumerate(tops) if t <= top + LINE_TOL), default=0)
            records[ri][ci] = f"{records[ri][ci]} {text}".strip()
    return records


def _table_rows(page, table) -> list[list[str]]:
    rows: list[list[str]] = []
    texts = table.extract(x_tolerance=X_TOL)
    for row, raw in zip(table.rows, texts):
        heavy = sum(1 for c in raw if c and c.count("\n") >= 2)
        if heavy >= 2:  # several records squeezed into one ruled row
            split = _split_merged_row(page, row.cells)
            if split:
                rows.extend(split)
                continue
        rows.append([_clean(c) for c in raw])
    # drop empty rows and columns
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def _to_markdown(rows: list[list[str]]) -> str:
    header, body = rows[0], rows[1:]
    out = ["| " + " | ".join(h or " " for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(c for c in r) + " |" for r in body]
    return "\n".join(out)


def _heading_level(line: dict, body_size: float) -> int:
    text = line["text"].strip()
    chars = [c for c in line.get("chars", []) if c["text"].strip()]
    if not chars or len(text) > 120 or len(re.sub(r"[^A-Za-z]", "", text)) < 3:
        return 0
    size = statistics.mean(c["size"] for c in chars)
    if size >= body_size * 1.45:
        return 2
    if size >= body_size * 1.15:
        return 3
    bold = sum("bold" in c["fontname"].lower() for c in chars) / len(chars)
    if bold > 0.9 and len(text) <= 80 and not NUM_RE.search(text):
        return 3
    return 0


def _inside(line: dict, boxes: list[tuple]) -> bool:
    cy = (line["top"] + line["bottom"]) / 2
    cx = (line["x0"] + line["x1"]) / 2
    return any(x0 - 1 <= cx <= x1 + 1 and top - 1 <= cy <= bottom + 1 for x0, top, x1, bottom in boxes)


# ---------------------------------------------------------------------------------------------
def extract_page(page) -> PageResult:
    tables = [t for t in page.find_tables(TABLE_SETTINGS)]
    blocks: list[tuple[float, str]] = []
    boxes = []
    n_tables = 0
    for t in tables:
        rows = _table_rows(page, t)
        if len(rows) >= 2 and len(rows[0]) >= 2:
            blocks.append((t.bbox[1], _to_markdown(rows)))
            boxes.append(t.bbox)
            n_tables += 1
        # a 1-row or 1-column "table" is just a bordered box: its text is emitted as normal lines below
    result = _assemble(page, blocks, boxes, n_tables)
    if result.needs_fallback and n_tables == 0:
        # No ruling lines: rebuild the transaction table from header positions and running balances
        # (checked against the statement's own arithmetic) before handing the page to Docling.
        from finpipe.unruled import extract_unruled

        found = extract_unruled(page)
        if found:
            labels, rows, (top, bottom) = found
            region = (0, top, page.width, bottom)
            result = _assemble(page, [(top, _to_markdown([labels] + rows))], [region], 1)
            result.notes.append("unruled table rebuilt from column positions")
    return result


def _assemble(page, blocks: list[tuple[float, str]], boxes: list[tuple], n_tables: int) -> PageResult:
    """Merge table blocks with the page's remaining text lines, in reading order."""
    blocks = list(blocks)
    lines =[ln for ln in page.extract_text_lines(return_chars=True, strip=True, x_tolerance=X_TOL)
             if not _inside(ln, boxes)]
    sizes = [c["size"] for ln in lines for c in ln.get("chars", []) if c["text"].strip()]
    body = statistics.median(sizes) if sizes else 10.0
    numeric_lines = 0
    for ln in lines:
        text = re.sub(r"\s+", " ", CID_RE.sub(" ", ln["text"])).strip()
        if not text:
            continue
        if len(NUM_RE.findall(text)) >= 2:
            numeric_lines += 1
        level = _heading_level(ln, body)
        blocks.append((ln["top"], ("#" * level + " " + text) if level else text))

    blocks.sort(key=lambda b: b[0])
    parts: list[str] = []
    for _, text in blocks:
        is_block = text.startswith("|") or text.startswith("#")
        if parts and not is_block and not parts[-1].startswith(("|", "#")):
            parts[-1] += "\n" + text  # consecutive text lines form one paragraph block
        else:
            parts.append(text)

    # Rows of figures outside any ruled table: an unruled table this route would flatten.
    needs_fallback = numeric_lines >= 6
    return PageResult("\n\n".join(parts), needs_fallback, n_tables,
                      [f"{numeric_lines} numeric lines outside tables"] if needs_fallback else [])


def extract_pages(path: str, first: int, last: int) -> tuple[list[tuple[int, str]], list[int]]:
    """Extract pages first..last (1-based). Returns ([(page_no, markdown)], [pages needing Docling])."""
    import pdfplumber

    done, fallback = [], []
    with pdfplumber.open(path, pages=list(range(first, last + 1))) as pdf:
        for page in pdf.pages:
            res = extract_page(page)
            if res.needs_fallback:
                fallback.append(page.page_number)
            else:
                done.append((page.page_number, res.markdown))
            page.close()
    return done, fallback
