"""Transaction tables without ruling lines (columns aligned by whitespace only).

Many bank statements draw no lines at all. Instead of sending those pages to Docling (~20 s/page on
CPU), columns are rebuilt from the header row's positions and rows from the running balance:

  * the header line (Date / Particulars / Debit / Credit / Balance ...) gives each column's x-range;
    amounts are right-aligned, so they are matched to money columns by their right edge;
  * every transaction prints exactly one balance, so each balance marks one row; wrapped narration
    lines are attached to the nearest row (rows may be top-aligned or vertically centred);
  * the result is checked with the statement's own arithmetic: previous balance - debit + credit must
    equal the printed balance. Pages whose rows don't reconcile are left for Docling.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

# Statement amounts always carry two decimals; requiring them keeps day/year numbers ("25", "2025")
# and reference numbers out of the money columns.
AMOUNT_RE = re.compile(r"^\(?-?\d{1,3}(?:,\d{2,3})*\.\d{2}\)?$|^\(?-?\d+\.\d{2}\)?$")
DATE_RE = re.compile(r"\b\d{1,2}[\s/\-.](?:\d{1,2}|[A-Za-z]{3,9})[\s/\-.,]+\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
SKIP_TOKENS = {"INR", "RS", "RS.", "₹", "-"}
SIGN_TOKENS = {"DR", "CR", "DR.", "CR."}
LINE_TOL = 2.5
X_TOL = 1.5

MONEY = ("debit", "credit", "balance")


def _kind(label: str, position: int) -> str:
    """Column type from its header label ("Withdrawal Amt." -> debit, "Transaction Details" -> text)."""
    n = _norm(label)
    if "balance" in n or n in ("bal", "closingbal"):
        return "balance"
    if "debit" in n or "withdrawal" in n or n == "dr":
        return "debit"
    if "credit" in n or "deposit" in n or n == "cr":
        return "credit"
    if "value" in n:
        return "text"  # value date: kept, but the transaction date identifies the row
    if "date" in n or (position == 0 and n in ("transaction", "txn", "tran", "txndt", "trandate")):
        return "date"
    return "text"


@dataclass
class Col:
    label: str
    kind: str
    x0: float
    x1: float


def _lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) < LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(ln, key=lambda w: w["x0"]) for ln in lines]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z]", "", t.lower())


def _header(lines: list[list[dict]]) -> tuple[int, list[Col]] | None:
    for i, ln in enumerate(lines[:60]):
        if not {"balance", "debit", "credit"} <= {_kind(w["text"], 1) for w in ln}:
            continue
        groups: list[list[dict]] = []
        for w in ln:  # merge words of one header label ("Transaction Details", "Value Date")
            if groups and w["x0"] - groups[-1][-1]["x1"] < 6:
                groups[-1].append(w)
            else:
                groups.append([w])
        cols = []
        for pos, g in enumerate(groups):
            label = " ".join(w["text"] for w in g)
            cols.append(Col(label, _kind(label, pos), g[0]["x0"], g[-1]["x1"]))
        if sum(c.kind in MONEY for c in cols) >= 3:
            return i, cols
    return None


def _assign(word: dict, cols: list[Col]) -> int | None:
    t = word["text"]
    if t.upper() in SKIP_TOKENS:
        return None
    if AMOUNT_RE.match(t) or t.upper() in SIGN_TOKENS:
        money = [i for i, c in enumerate(cols) if c.kind in MONEY]
        best = min(money, key=lambda i: abs(cols[i].x1 - word["x1"]))
        # right-aligned under its header; an amount far from every money column is narration text
        # ("Txn Amt. 2,00,000.00" inside the details)
        if abs(cols[best].x1 - word["x1"]) <= 45 or t.upper() in SIGN_TOKENS:
            return best
    cx = (word["x0"] + word["x1"]) / 2
    centers = [(c.x0 + c.x1) / 2 for c in cols]
    bounds = [(centers[i] + centers[i + 1]) / 2 for i in range(len(cols) - 1)]
    idx = sum(cx > b for b in bounds)
    if cols[idx].kind in MONEY:  # stray text in a money column belongs to the nearest text column
        text_cols = [i for i, c in enumerate(cols) if c.kind not in MONEY]
        idx = min(text_cols, key=lambda i: abs(centers[i] - cx)) if text_cols else idx
    return idx


def signed(cell: str) -> float | None:
    t = cell.replace(" ", "").upper()
    m = re.search(r"\(?-?[\d,]+(?:\.\d+)?\)?", t)
    if not m or not re.search(r"\d", m.group(0)):
        return None
    raw = m.group(0)
    try:
        value = float(raw.strip("()-").replace(",", ""))
    except ValueError:
        return None
    negative = raw.startswith("(") or raw.startswith("-") or t.endswith("DR") or t.endswith("DR.")
    return -value if negative else value


def reconcile_rate(rows: list[dict]) -> float:
    """Share of consecutive rows where prev balance - debit + credit == balance (either row order)."""
    def rate(seq):
        ok = n = 0
        for prev, cur in zip(seq, seq[1:]):
            if prev["bal"] is None or cur["bal"] is None:
                continue
            n += 1
            ok += abs(prev["bal"] - (cur["dr"] or 0) + (cur["cr"] or 0) - cur["bal"]) < 0.015
        return ok / n if n else 0.0
    return max(rate(rows), rate(rows[::-1]))


def extract_unruled(page, min_rows: int = 3, min_reconcile: float = 0.8):
    """Returns (header_labels, rows_as_cells, (top, bottom) of the table) or None if not confident."""
    words = [w for w in page.extract_words(x_tolerance=X_TOL, keep_blank_chars=False) if w["text"].strip()]
    lines = _lines(words)
    found = _header(lines)
    if not found:
        return None
    hi, cols = found
    header_bottom = max(w["bottom"] for w in lines[hi])
    body = [ln for ln in lines[hi + 1:]]
    bal_col = next(i for i, c in enumerate(cols) if c.kind == "balance")

    # anchors: lines that print an amount in the balance column
    anchors = []
    for li, ln in enumerate(body):
        if any(AMOUNT_RE.match(w["text"]) and _assign(w, cols) == bal_col for w in ln):
            anchors.append(li)
    if len(anchors) < min_rows:
        return None
    date_cols = {i for i, c in enumerate(cols) if c.kind == "date"}

    def has_date(line):  # dates may be split into words ("25 Aug 2025")
        return bool(DATE_RE.search(" ".join(w["text"] for w in line if _assign(w, cols) in date_cols)))

    # Top-aligned rows print date and amounts on their first line, with wrapped details below it;
    # otherwise the amount line sits in the middle of the row.
    top_aligned = sum(has_date(body[a]) for a in anchors) >= len(anchors) / 2
    ys = [body[a][0]["top"] for a in anchors]
    gaps = [b - a for a, b in zip(ys, ys[1:])] or [20.0]
    half = statistics.median(gaps) / 2

    records: list[list[list[str]]] = [[[] for _ in cols] for _ in anchors]
    for li, ln in enumerate(body):
        y = ln[0]["top"]
        if top_aligned:
            r = max((k for k, a in enumerate(anchors) if a <= li), default=None)
            if r is None or (r == len(anchors) - 1 and y - ys[r] > 2 * half + 1):
                continue  # above the first row, or footer text below the last row
        else:
            r = min(range(len(anchors)), key=lambda k: abs(ys[k] - y))
            if abs(ys[r] - y) > 2 * half + 1:
                continue
        for w in ln:
            ci = _assign(w, cols)
            if ci is not None:
                records[r][ci].append(w["text"])

    rows = [[" ".join(cell) for cell in rec] for rec in records]
    idx = {k: next((i for i, c in enumerate(cols) if c.kind == k), None) for k in MONEY}
    parsed = [{"bal": signed(r[idx["balance"]]),
               "dr": abs(signed(r[idx["debit"]]) or 0) if idx["debit"] is not None and r[idx["debit"]] else None,
               "cr": abs(signed(r[idx["credit"]]) or 0) if idx["credit"] is not None and r[idx["credit"]] else None}
              for r in rows]
    if reconcile_rate(parsed) < min_reconcile:
        return None
    # The table region starts at the first row: the header line and any text continued from the
    # previous page stay outside it and are kept as ordinary text.
    first_top = min(w["top"] for w in body[anchors[0]])
    if not top_aligned:
        first_top -= half
    bottom = max(w["bottom"] for ln in body[: anchors[-1] + 1] for w in ln)
    return [c.label for c in cols], rows, (first_top - 1, bottom + 2 * half)
