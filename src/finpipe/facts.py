"""Key facts read by rules at ingest time, no language model involved.

Questions about a whole document ("how many accounts?", "total income for AY 2024-25?", "closing
balance of each account?") are what top-k passage retrieval gets wrong: the answer is spread over
pages the retriever may not return. Bank statements and ITRs have fixed layouts, so their key facts
are read directly from the parsed markdown, stored per document, shown in the UI and handed to the
model with every question about that document.

Everything here works on the page-tagged markdown produced by the parser (tables as `| a | b |` rows,
`<!-- page: N -->` markers), so it is independent of which parser route produced it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PAGE_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")
CID_RE = re.compile(r"\(cid:\d+\)")
NUM_CELL_RE = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:Cr|Dr|CR|DR))?$")
TRAILING_NUM_RE = re.compile(r"(\(?-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\)?)\s*$")

# IFSC prefix -> bank (the first four letters of an IFSC identify the bank)
IFSC_BANKS = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank", "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank", "PUNB": "Punjab National Bank", "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank", "UBIN": "Union Bank of India", "IDIB": "Indian Bank", "BKID": "Bank of India",
    "YESB": "Yes Bank", "INDB": "IndusInd Bank", "IDFB": "IDFC First Bank", "FDRL": "Federal Bank",
    "MAHB": "Bank of Maharashtra", "IOBA": "Indian Overseas Bank", "UCBA": "UCO Bank",
    "CBIN": "Central Bank of India", "PSIB": "Punjab & Sind Bank", "RATN": "RBL Bank",
    "AUBL": "AU Small Finance Bank", "HSBC": "HSBC", "SCBL": "Standard Chartered", "CITI": "Citibank",
    "DBSS": "DBS Bank", "KARB": "Karnataka Bank", "SIBL": "South Indian Bank", "KVBL": "Karur Vysya Bank",
    "TMBL": "Tamilnad Mercantile Bank", "CSBK": "CSB Bank", "DLXB": "Dhanlaxmi Bank", "JAKA": "J&K Bank",
}
BANK_NAMES = {  # names as they appear in statement headers (spaces removed, upper-case)
    "STATEBANKOFINDIA": "State Bank of India", "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "AXISBANK": "Axis Bank", "KOTAKMAHINDRA": "Kotak Mahindra Bank", "PUNJABNATIONALBANK": "Punjab National Bank",
    "BANKOFBARODA": "Bank of Baroda", "CANARABANK": "Canara Bank", "UNIONBANKOFINDIA": "Union Bank of India",
    "YESBANK": "Yes Bank", "INDUSINDBANK": "IndusInd Bank", "IDFCFIRST": "IDFC First Bank",
    "FEDERALBANK": "Federal Bank",
}

ACCOUNT_RE = re.compile(
    r"(?i)\b(?:account\s*(?:number|no)\.?|a/c\s*(?:number|no)\.?)\s*[:\-]?\s*([0-9Xx*]{6,20})\b")
IFSC_LABEL_RE = re.compile(r"(?i)IFSC\s*(?:code)?\s*[:\-]?\s*([A-Z]{4}0[A-Z0-9]{6})")
IFSC_RE = re.compile(r"\b([A-Z]{4})0[A-Z0-9]{6}\b")
HOLDER_RE = re.compile(r"(?i)account\s*(?:holder\s*)?name\s*[:\-]?\s*([A-Za-z][A-Za-z .,&']{2,80})")
HOLDER_AFTER_ACCT_RE = re.compile(r"[0-9]{6,20}\s*(?:\([A-Z]{3}\))?\s*-\s*([A-Z][A-Z .]{2,60})")
DATE_TXT = r"(?:\d{1,2}[/\-. ](?:\d{1,2}|[A-Za-z]{3})[/\-. ]\d{2,4}|\d{4}-\d{2}-\d{2})"
SUMMARY_LABELS = {  # statement summary block: label -> field
    "opening": "opening_balance", "totaldebit": "total_debits", "totalcredit": "total_credits",
    "closing": "closing_balance", "debits": "total_debits", "credits": "total_credits",
}
AMOUNT_TOKEN_RE = re.compile(r"\(?-?\d{1,3}(?:,\d{2,3})*\.\d{2}\)?(?:\s*(?:Cr|Dr|CR|DR))?")
PERIOD_RE = re.compile(rf"(?i)(?:from|period)\s*:?\s*({DATE_TXT})\s*(?:to|-)\s*:?\s*({DATE_TXT})")

ITR_MARKER_RE = re.compile(r"(?i)INDIAN\s+INCOME\s+TAX\s+RETURN")
ITR_FORM_RE = re.compile(r"\bITR\s*-?\s*([1-7])\b")
AY_RE = re.compile(r"\b(20\d{2})\s*-\s*(\d{2}|20\d{2})\b")
PAN_RE = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
ACK_RE = re.compile(r"(?i)Acknowledgement\s*Number\s*:?\s*(\d{10,20})")
FILED_RE = re.compile(r"(?i)Date\s*of\s*Filing\s*:?\s*(\d{1,2}-[A-Za-z]{3}-\d{4})")
NAME_PART_RE = {
    part: re.compile(rf"{part} Name\s+([A-Z][A-Z .]*?)\s*(?:\||\(|$)")
    for part in ("First", "Middle", "Last")
}
# (field, label regex); the value is the last numeric cell of the first row matching the label
ITR_AMOUNTS = [
    ("gross_total_income", re.compile(r"(?i)Gross Total Income")),
    ("total_deductions", re.compile(r"(?i)\bTotal deductions\b")),
    ("total_income", re.compile(r"(?<!Gross )(?<!GROSS )Total Income")),
    ("tax_on_total_income", re.compile(r"(?i)Tax payable on total income")),
    ("total_tax_and_cess", re.compile(r"(?i)Total Tax,? and Cess")),
    ("total_taxes_paid", re.compile(r"(?i)Total Taxes Paid")),
    ("refund", re.compile(r"(?i)\bRefund\s*\(")),
    ("tax_payable", re.compile(r"(?i)\bAmount payable\s*\(")),
    ("deduction_80C", re.compile(r"\b80C\s*-")),
    ("deduction_80D", re.compile(r"\b80D\s*-")),
]
ITR_LABELS = {
    "gross_total_income": "Gross total income", "total_deductions": "Total deductions",
    "total_income": "Total (taxable) income", "tax_on_total_income": "Tax on total income",
    "total_tax_and_cess": "Total tax and cess", "total_taxes_paid": "Total taxes paid",
    "refund": "Refund", "tax_payable": "Tax payable", "deduction_80C": "80C deduction",
    "deduction_80D": "80D deduction",
}


# --------------------------------------------------------------------------------------------
@dataclass
class Line:
    page: int
    text: str
    cells: list[str] | None  # table row cells, or None for a text line


def iter_lines(markdown: str) -> list[Line]:
    out, page = [], 1
    for raw in markdown.splitlines():
        s = re.sub(r"[ 	]+", " ", CID_RE.sub(" ", raw)).strip()
        if m := PAGE_RE.match(s):
            page = int(m.group(1))
            continue
        if not s or re.fullmatch(r"\|?(\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*\|?", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")] if s.startswith("|") else None
        out.append(Line(page, s.lstrip("#").strip(), cells))
    return out


def parse_amount(text: str) -> float | None:
    t = text.strip().replace(" ", "")
    if not t or not re.search(r"\d", t):
        return None
    negative = (t.startswith("(") and t.endswith(")")) or t.startswith("-") or t.upper().endswith("DR")
    digits = re.sub(r"(?i)(cr|dr)$", "", t).strip("()-")
    digits = digits.replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def fmt_inr(value: float | None) -> str | None:
    """1234567.5 -> '12,34,567.50' (Indian digit grouping)."""
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    whole, frac = f"{abs(value):.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return sign + ",".join(groups + [tail]) + ("" if frac == "00" else f".{frac}")


def parse_date(text: str) -> date | None:
    t = re.sub(r"\s+", " ", text.strip())[:11]
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y",
                "%d-%b-%y", "%d %b %y", "%d/%b/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    m = re.match(rf"({DATE_TXT})", t)
    if m and m.group(1) != t:
        return parse_date(m.group(1))
    return None


def last_number(line: Line, label: re.Pattern | None = None) -> str | None:
    """The value of a labelled row: its last numeric cell to the right of the label, else a number
    ending the last filled cell (forms often print "Total Income 6,75,330" in a single cell).
    Cells left of the label are serial numbers or row codes ("| 2 | Tax payable ... |"), never values."""
    if line.cells is not None:
        start = 0
        if label is not None:
            start = next((i for i, c in enumerate(line.cells) if label.search(c)), 0)
        after = line.cells[start:]
        # the row's serial number is sometimes repeated as a code right of the label ("| 2 | ... | 2 |")
        serial = next((c for c in line.cells[:start] if c), None) if label is not None else None
        for cell in reversed(after[1:] if label is not None else after):
            if NUM_CELL_RE.match(cell) and cell != serial:
                return cell
        filled = [c for c in after if c]
        text = filled[-1] if filled else ""
    else:
        text = line.text
    m = TRAILING_NUM_RE.search(text)
    return m.group(1) if m and re.search(r"[A-Za-z]", text[: m.start()]) else None


# --------------------------------------------------------------------------------------------
def _norm(h: str) -> str:
    return re.sub(r"[^a-z]", "", h.lower())


def _columns(header: list[str]) -> dict[str, int] | None:
    names = [_norm(h) for h in header]
    cols: dict[str, int] = {}
    for i, n in enumerate(names):
        if "balance" in n or n in ("closingbal", "bal"):
            cols.setdefault("balance", i)
        elif "date" in n and "value" not in n:
            cols.setdefault("date", i)
        elif "withdrawal" in n or "debit" in n or n == "dr":
            cols.setdefault("debit", i)
        elif "deposit" in n or "credit" in n or n == "cr":
            cols.setdefault("credit", i)
        elif any(w in n for w in ("narration", "description", "particular", "remark", "details")):
            cols.setdefault("narration", i)
    if "date" not in cols:  # statements that only have a value-date column
        for i, n in enumerate(names):
            if "date" in n or n in ("valuedt", "txndt"):
                cols["date"] = i
                break
    return cols if {"balance", "date"} <= cols.keys() else None


def _bank_for(lines: list[Line], first_page: int) -> str | None:
    header = [ln for ln in lines if ln.page == first_page and ln.cells is None]
    for ln in header:
        if m := IFSC_LABEL_RE.search(ln.text):
            return IFSC_BANKS.get(m.group(1)[:4], m.group(1)[:4])
    squashed = re.sub(r"[^A-Z]", "", " ".join(ln.text for ln in header).upper())
    for key, name in BANK_NAMES.items():
        if key in squashed:
            return name
    for ln in header:
        if m := IFSC_RE.search(ln.text):
            if m.group(1) in IFSC_BANKS:
                return IFSC_BANKS[m.group(1)]
    section_text = re.sub(r"[^A-Z]", "", " ".join(ln.text for ln in lines).upper())
    hits = {name: section_text.count(key) for key, name in BANK_NAMES.items() if key in section_text}
    hits |= {name: section_text.count(short) for short, name in (("ICICI", "ICICI Bank"), ("HDFC", "HDFC Bank"))
             if short in section_text and name not in hits}
    return max(hits, key=hits.get) if hits else None


def bank_statement_facts(lines: list[Line]) -> dict | None:
    starts: list[tuple[int, str]] = []
    for ln in lines:
        if m := ACCOUNT_RE.search(ln.text):
            number = m.group(1)
            if not starts or starts[-1][1] != number:
                if number not in {n for _, n in starts}:
                    starts.append((ln.page, number))
    if not starts:
        return None
    last_page = max(ln.page for ln in lines)
    accounts = []
    for i, (page, number) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else last_page
        end = max(end, page)
        sec = [ln for ln in lines if page <= ln.page <= end]
        acct: dict = {"account_number": number, "pages": f"{page}-{end}" if end > page else str(page),
                      "page_start": page, "page_end": end, "bank": _bank_for(sec, page)}
        for ln in sec:
            if "holder" not in acct:
                if m := HOLDER_RE.search(ln.text):
                    acct["holder"] = m.group(1).strip(" ,")
                elif number in ln.text and (m := HOLDER_AFTER_ACCT_RE.search(ln.text)):
                    acct["holder"] = m.group(1).strip()
            if "period" not in acct and (m := PERIOD_RE.search(ln.text)):
                acct["period"] = f"{m.group(1)} to {m.group(2)}"
        acct |= _transactions(sec, acct.get("period"))
        if printed := _statement_summary(sec):
            acct["statement_summary"] = printed
            _cross_check(acct)
        if acct.get("transactions") and not acct.get("row_check"):
            acct.setdefault("warnings", []).append(
                "debit/credit columns could not be read (e.g. scanned pages); balances and totals are unreliable")
        accounts.append(acct)
    return {"kind": "bank_statement", "accounts": accounts}


def _statement_summary(sec: list[Line]) -> dict | None:
    """Figures the statement prints itself: a label line ("Opening Balance  Total Debit  Total Credit
    Closing Balance") followed by a line of amounts, or "Closing Balance : 12,345.67" style lines."""
    for i, ln in enumerate(sec):
        squashed = re.sub(r"[^a-z]", "", ln.text.lower())
        if "opening" in squashed and "closing" in squashed and i + 1 < len(sec):
            order = []
            for m in re.finditer(r"opening|totaldebit|totalcredit|closing|debits|credits", squashed):
                field = SUMMARY_LABELS[m.group(0)]
                if field not in order:
                    order.append(field)
            amounts = AMOUNT_TOKEN_RE.findall(sec[i + 1].text)
            if len(amounts) >= len(order) >= 2:
                return {f: fmt_inr(parse_amount(a)) for f, a in zip(order, amounts)}
    found = {}
    for ln in sec:
        for label, field in (("opening balance", "opening_balance"), ("closing balance", "closing_balance")):
            if field not in found and (m := re.search(label + r"\s*:?\s*(" + AMOUNT_TOKEN_RE.pattern + ")", ln.text, re.I)):
                found[field] = fmt_inr(parse_amount(m.group(1)))
    return found or None


def _cross_check(acct: dict) -> None:
    """Flag computed figures that disagree with the statement's own summary."""
    printed = acct["statement_summary"]
    diffs = [k for k in ("closing_balance", "total_debits", "total_credits")
             if printed.get(k) and acct.get(k) and printed[k] != acct[k]]
    if diffs:
        acct.setdefault("warnings", []).append(
            "computed " + ", ".join(k.replace("_", " ") for k in diffs) + " differ from the statement's printed "
            "summary; the file may hold overlapping or out-of-order statements - prefer the printed figures")


def _window(period: str | None) -> tuple[date, date]:
    """Plausible transaction dates: the statement period +-45 days, else 2000..next year."""
    today = date.today()
    lo, hi = date(2000, 1, 1), date(today.year + 1, 12, 31)
    if period and " to " in period:
        a, b = (parse_date(x) for x in period.split(" to ", 1))
        if a and b:
            from datetime import timedelta

            lo, hi = a - timedelta(days=45), b + timedelta(days=45)
    return lo, hi


def _transactions(sec: list[Line], period: str | None = None) -> dict:
    lo, hi = _window(period)
    rows: list[tuple[date, int, dict]] = []
    cols: dict[str, int] | None = None
    width = 0
    order = 0
    for ln in sec:
        if ln.cells is None:
            continue
        found = _columns(ln.cells)
        if found:
            cols, width = found, len(ln.cells)
            continue
        if not cols or len(ln.cells) != width:
            continue
        d = parse_date(ln.cells[cols["date"]])
        bal = parse_amount(ln.cells[cols["balance"]])
        if d is None or bal is None or not lo <= d <= hi:
            continue
        rec = {"balance": bal,
               "debit": parse_amount(ln.cells[cols["debit"]]) if "debit" in cols else None,
               "credit": parse_amount(ln.cells[cols["credit"]]) if "credit" in cols else None,
               "narration": ln.cells[cols["narration"]][:80] if "narration" in cols else ""}
        rows.append((d, order, rec))
        order += 1
    if not rows:
        return {}
    in_doc_order = [r for _, _, r in rows]
    if rows[0][0] > rows[-1][0]:  # newest-first statement
        rows = [(d, -o, r) for d, o, r in rows]
    rows.sort(key=lambda x: (x[0], x[1]))
    first, last = rows[0], rows[-1]
    debits = sum(abs(r["debit"]) for _, _, r in rows if r["debit"])
    credits = sum(abs(r["credit"]) for _, _, r in rows if r["credit"])

    def top(kind: str) -> list[dict]:
        biggest = sorted((x for x in rows if x[2][kind]), key=lambda x: abs(x[2][kind]), reverse=True)[:3]
        return [{"date": d.strftime("%d %b %Y"), "amount": fmt_inr(abs(r[kind])), "narration": r["narration"]}
                for d, _, r in biggest]

    checked = ok = 0
    for seq in (in_doc_order, in_doc_order[::-1]):
        c = k = 0
        for prev, cur in zip(seq, seq[1:]):
            if cur["debit"] is None and cur["credit"] is None:
                continue
            c += 1
            k += abs(prev["balance"] - abs(cur["debit"] or 0) + abs(cur["credit"] or 0) - cur["balance"]) < 0.015
        if c and k / c > (ok / checked if checked else -1):
            checked, ok = c, k
    return {
        "row_check": f"{ok / checked:.0%} of {checked} rows reconcile (previous balance - debit + credit = balance)"
                     if checked else None,
        "transactions": len(rows),
        "first_transaction": first[0].strftime("%d %b %Y"),
        "last_transaction": last[0].strftime("%d %b %Y"),
        "closing_balance": fmt_inr(last[2]["balance"]),
        "total_debits": fmt_inr(debits),
        "total_credits": fmt_inr(credits),
        "largest_debits": top("debit"),
        "largest_credits": top("credit"),
    }


# --------------------------------------------------------------------------------------------
def itr_facts(lines: list[Line]) -> dict | None:
    """One entry per return: a file often bundles several years' returns (e.g. a 29-page file holding
    three ITR-1s). Every ITR page repeats its acknowledgement number, which delimits the returns."""
    if not any(ITR_MARKER_RE.search(ln.text) for ln in lines[:80]):
        return None
    ack_by_page: dict[int, str] = {}
    for ln in lines:
        if ln.page not in ack_by_page and (m := ACK_RE.search(ln.text)):
            ack_by_page[ln.page] = m.group(1)
    pages = sorted({ln.page for ln in lines})
    segments: list[list[int]] = []
    current_ack = None
    for pg in pages:
        ack = ack_by_page.get(pg)
        starts_new = any(ITR_MARKER_RE.search(ln.text) for ln in lines if ln.page == pg) and (
            ack is None or ack != current_ack)
        if not segments or (ack is not None and ack != current_ack) or starts_new:
            segments.append([pg])
            current_ack = ack if ack is not None else current_ack
        else:
            segments[-1].append(pg)
    returns = []
    for seg in segments:
        seg_lines = [ln for ln in lines if seg[0] <= ln.page <= seg[-1]]
        r = _one_return(seg_lines)
        r |= {"pages": f"{seg[0]}-{seg[-1]}" if seg[-1] > seg[0] else str(seg[0]),
              "page_start": seg[0], "page_end": seg[-1]}
        returns.append(r)
    return {"kind": "itr", "returns": returns}


def _one_return(lines: list[Line]) -> dict:
    first_page = [ln for ln in lines if ln.page == lines[0].page]
    facts: dict = {}
    # An explicit "Form Number | ITR-3" row (ITR-V acknowledgement pages) wins; rows that list several
    # forms ("ITR-1(SAHAJ), ITR-2, ITR-3, ...") are boilerplate and never identify the return.
    for ln in first_page:
        if re.search(r"(?i)form\s*number", ln.text) and (m := ITR_FORM_RE.search(ln.text.replace("ITR", " ITR"))):
            facts["form"] = f"ITR-{m.group(1)}"
            break
    for ln in first_page:
        forms = ITR_FORM_RE.findall(ln.text.replace("ITR", " ITR"))
        if "form" not in facts and len(forms) == 1:
            facts["form"] = f"ITR-{forms[0]}"
            if m2 := AY_RE.search(ln.text):  # the AY sits on the same row as the form name
                facts["assessment_year"] = f"{m2.group(1)}-{m2.group(2)[-2:]}"
    for ln in first_page:
        if "assessment_year" not in facts and re.search(r"(?i)assessment\s*year", ln.text) and (m := AY_RE.search(ln.text)):
            facts["assessment_year"] = f"{m.group(1)}-{m.group(2)[-2:]}"
    if "assessment_year" not in facts:
        for ln in first_page:
            if m := AY_RE.search(ln.text):
                facts["assessment_year"] = f"{m.group(1)}-{m.group(2)[-2:]}"
                break
    joined = " | ".join(ln.text for ln in first_page)
    if m := PAN_RE.search(joined):
        facts["pan"] = m.group(1)
    if m := ACK_RE.search(joined):
        facts["acknowledgement_number"] = m.group(1)
    if m := FILED_RE.search(joined):
        facts["date_of_filing"] = m.group(1)
    parts = [m.group(1).strip() for rx in NAME_PART_RE.values() if (m := rx.search(joined)) and m.group(1).strip()]
    if parts:
        facts["name"] = " ".join(parts)
    else:  # ITR-V layout: "| Name | | RAHUL MEHTA |"
        for ln in first_page:
            filled = [c for c in (ln.cells or []) if c]
            if len(filled) >= 2 and filled[0].lower() == "name" and re.fullmatch(r"[A-Z][A-Z .]{2,60}", filled[1]):
                facts["name"] = filled[1]
                break
    amounts: dict[str, str] = {}
    for key, rx in ITR_AMOUNTS:
        for ln in lines:
            if rx.search(ln.text) and (value := last_number(ln, rx)) is not None:
                amounts[key] = value
                break
    facts["amounts"] = amounts
    return facts


# --------------------------------------------------------------------------------------------
def extract_facts(markdown: str) -> dict | None:
    """Facts for a bank statement or an ITR, detected from the content; None for other documents."""
    lines = iter_lines(markdown)
    if not lines:
        return None
    return itr_facts(lines) or bank_statement_facts(lines)


def section_for_page(facts: dict | None, page: int) -> dict | None:
    """The account (bank statement) or return (ITR) that a page belongs to."""
    for item in (facts or {}).get("accounts", []) + (facts or {}).get("returns", []):
        if item["page_start"] <= page <= item["page_end"]:
            return item
    return None


def summary(facts: dict | None) -> str:
    """One-line summary for lists and prompts."""
    if not facts:
        return ""
    if facts["kind"] == "itr":
        rets = facts.get("returns", [])
        years = ", ".join(f"{r.get('form', 'ITR')} AY {r.get('assessment_year', '?')}" for r in rets)
        names = sorted({r["name"] for r in rets if r.get("name")})
        return f"{len(rets)} return{'s' if len(rets) != 1 else ''}: {years}" + (f" · {', '.join(names)}" if names else "")
    accts = facts.get("accounts", [])
    names = ", ".join(f"{a.get('bank') or 'Unknown bank'} …{a['account_number'][-4:]}" for a in accts)
    return f"{len(accts)} account{'s' if len(accts) != 1 else ''}: {names}"


def render(facts: dict | None, source: str, year: str | None = None) -> str:
    """Compact text block given to the model alongside retrieved passages. With `year`, only that
    assessment year's return is included, so a year-scoped question can't drift to other years."""
    if not facts:
        return ""
    out = [f"{source}:"]
    if facts["kind"] == "itr":
        returns = [r for r in facts["returns"] if not year or r.get("assessment_year") == year]
        if not returns:
            return ""
        out.append(f"  {len(facts['returns'])} income tax return(s) in this file"
                   + (f"; showing AY {year} only" if year else ""))
        for i, r in enumerate(returns, 1):
            head = [f"{k.replace('_', ' ')}: {r[k]}" for k in ("form", "assessment_year", "name", "pan",
                    "date_of_filing", "acknowledgement_number") if r.get(k)]
            out.append(f"  {i}. pages {r['pages']}; " + "; ".join(head))
            amounts = "; ".join(f"{ITR_LABELS.get(k, k)}: {v}" for k, v in r.get("amounts", {}).items())
            if amounts:
                out.append(f"     {amounts}")
    else:
        out.append(f"  {len(facts['accounts'])} account(s) in this file")
        for i, a in enumerate(facts["accounts"], 1):
            bits = [f"{a.get('bank') or 'Unknown bank'} a/c {a['account_number']}", f"pages {a['pages']}"]
            for k in ("holder", "period", "transactions", "first_transaction", "last_transaction"):
                if a.get(k) not in (None, ""):
                    bits.append(f"{k.replace('_', ' ')}: {a[k]}")
            if a.get("statement_summary"):
                bits.append("printed summary: " + ", ".join(f"{k.replace('_', ' ')} {v}"
                                                             for k, v in a["statement_summary"].items()))
            for k in ("closing_balance", "total_debits", "total_credits"):
                if a.get(k) not in (None, ""):
                    bits.append(f"computed {k.replace('_', ' ')}: {a[k]}")
            if a.get("row_check"):
                bits.append(f"row check: {a['row_check']}")
            out.append(f"  {i}. " + "; ".join(str(b) for b in bits))
            for w in a.get("warnings", []):
                out.append(f"     WARNING: {w}")
            for kind in ("largest_debits", "largest_credits"):
                if a.get(kind):
                    items = "; ".join(f"{t['amount']} on {t['date']} ({t['narration']})" if t["narration"]
                                      else f"{t['amount']} on {t['date']}" for t in a[kind])
                    out.append(f"     {kind.replace('_', ' ')}: {items}")
    return "\n".join(out)


class FactsStore:
    """One JSON file per document under <data_dir>/facts."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.json"

    def put(self, doc_id: str, source: str, facts: dict | None) -> None:
        self._path(doc_id).write_text(json.dumps({"source": source, "facts": facts}, ensure_ascii=False),
                                      encoding="utf-8")

    def get(self, doc_id: str) -> dict | None:
        p = self._path(doc_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def delete(self, doc_id: str) -> None:
        self._path(doc_id).unlink(missing_ok=True)

    def all(self) -> dict[str, dict]:
        """doc_id -> {"source", "facts"} for every document with stored facts."""
        out = {}
        for p in sorted(self.root.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        return out
