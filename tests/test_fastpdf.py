import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from finpipe.fastpdf import extract_pages
from finpipe.pipeline import _runs

ROWS = [["Date", "Narration", "Withdrawal", "Deposit", "Balance"],
        ["01/04/24", "Opening balance", "", "", "1,00,000.00"],
        ["02/04/24", "NEFT to landlord", "32,000.00", "", "68,000.00"],
        ["05/04/24", "Salary credit", "", "1,25,450.75", "1,93,450.75"]]


def _ruled_table(c, rows, y, inner_rules=True):
    t = Table(rows, colWidths=[70, 180, 80, 80, 90])
    style = [("BOX", (0, 0), (-1, -1), 0.8, colors.black), ("LINEBEFORE", (1, 0), (-1, -1), 0.8, colors.black),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black)]
    if inner_rules:
        style.append(("INNERGRID", (0, 0), (-1, -1), 0.5, colors.black))
    t.setStyle(TableStyle(style))
    w, h = t.wrapOn(c, 500, 700)
    t.drawOn(c, 40, y - h)


@pytest.fixture
def statement_pdf(tmp_path):
    path = tmp_path / "statement.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    # page 1: fully ruled table + a heading
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "Account Statement")
    c.setFont("Helvetica", 10)
    c.drawString(40, 780, "Account No: 1234567890")
    _ruled_table(c, ROWS, 760)
    c.showPage()
    # page 2: column rules only (records not separated by horizontal lines)
    c.setFont("Helvetica", 10)
    _ruled_table(c, [ROWS[0], ["\n".join(r[i] for r in ROWS[1:]) for i in range(5)]], 800, inner_rules=False)
    c.showPage()
    # page 3: unruled columns of figures -> must fall back to Docling
    c.setFont("Helvetica", 10)
    for i in range(10):
        c.drawString(40, 780 - i * 16, f"0{i + 1}/05/24   Transfer {i}      {i + 1},000.00      {50 + i},250.50")
    c.showPage()
    c.save()
    return str(path)


def test_ruled_table_heading_and_fallback(statement_pdf):
    pages, fallback = extract_pages(statement_pdf, 1, 3)
    md = dict(pages)
    assert fallback == [3]
    assert "## Account Statement" in md[1]
    assert "Account No: 1234567890" in md[1]
    assert "| Date | Narration | Withdrawal | Deposit | Balance |" in md[1]
    assert "| 05/04/24 | Salary credit |  | 1,25,450.75 | 1,93,450.75 |" in md[1]


def test_merged_rows_are_split_per_record(statement_pdf):
    md = dict(extract_pages(statement_pdf, 2, 2)[0])[2]
    rows = [line for line in md.splitlines() if line.startswith("| 0")]
    assert rows == [
        "| 01/04/24 | Opening balance |  |  | 1,00,000.00 |",
        "| 02/04/24 | NEFT to landlord | 32,000.00 |  | 68,000.00 |",
        "| 05/04/24 | Salary credit |  | 1,25,450.75 | 1,93,450.75 |",
    ]


def _unruled_pdf(path, balances):
    """Statement with no ruling lines: header, then rows whose details wrap onto the next line."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 9)
    for x, label in ((40, "Date"), (120, "Transaction Details"), (330, "Debits"), (410, "Credits"), (490, "Balance")):
        c.drawString(x, 780, label)
    rows = [("01 Apr 2025", "NEFT FROM ACME LTD", "", "50,000.00"), ("03 Apr 2025", "UPI TO GROCER", "1,250.50", ""),
            ("07 Apr 2025", "RTGS TO LANDLORD", "32,000.00", ""), ("09 Apr 2025", "INTEREST", "", "12.00")] * 2
    y = 760
    for (date, text, dr, cr), bal in zip(rows, balances):
        c.drawString(40, y, date)
        c.drawString(120, y, text)
        for right, amount in ((360, dr), (445, cr), (530, bal)):
            if amount:
                c.drawRightString(right, y, amount)
        c.drawString(120, y - 11, "BRANCH : MUMBAI FORT")  # wrapped details line
        y -= 30
    c.showPage()
    c.save()
    return str(path)


def test_unruled_table_rebuilt_from_columns(tmp_path):
    good = ["1,50,000.00", "1,48,749.50", "1,16,749.50", "1,16,761.50",
            "1,66,761.50", "1,65,511.00", "1,33,511.00", "1,33,523.00"]  # each = previous - debit + credit
    pdf = _unruled_pdf(tmp_path / "unruled.pdf", good)
    pages, fallback = extract_pages(pdf, 1, 1)
    assert fallback == []
    md = dict(pages)[1]
    assert "| Date | Transaction Details | Debits | Credits | Balance |" in md
    assert "| 03 Apr 2025 | UPI TO GROCER BRANCH : MUMBAI FORT | 1,250.50 |  | 1,48,749.50 |" in md
    assert "| 09 Apr 2025 | INTEREST BRANCH : MUMBAI FORT |  | 12.00 | 1,16,761.50 |" in md


def test_unruled_table_that_does_not_reconcile_goes_to_docling(tmp_path):
    pdf = _unruled_pdf(tmp_path / "bad.pdf", ["1,50,000.00", "9,99,999.00", "1,23,456.00", "7,77,777.00"] * 2)
    assert extract_pages(pdf, 1, 1)[1] == [1]


def test_runs():
    assert _runs([9, 3, 4, 5]) == [(3, 5), (9, 9)]
    assert _runs([]) == []
