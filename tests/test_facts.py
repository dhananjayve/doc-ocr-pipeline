from finpipe.chunker import est_tokens
from finpipe.facts import extract_facts, fmt_inr, render, section_for_page, summary

BANK = """<!-- page: 1 -->

Account Name : Ms. ASHA RAO
Account Number :(cid:9)00000012345678901
RTGS/NEFT IFSC: SBIN0000572
### Account Statement from 01 Apr 2024 to 31 Mar 2025

| Txn Date | Description | Debit | Credit | Balance |
|---|---|---|---|---|
| 02 Apr 2024 | Opening credit |  | 10,000.00 | 10,000.00 |
| 05 Apr 2024 | NEFT to HDFC0001208 landlord | 4,000.00 |  | 6,000.00 |

<!-- page: 2 -->

| 09 Apr 2024 | UPI refund |  | 250.50 | 6,250.50 |

<!-- page: 3 -->

HDFC BANK LIMITED
Account No : 00099988877766
Statement From : 01/04/2024 To : 31/03/2025

| Date | Narration | Withdrawal Amt. | Deposit Amt. | Closing Balance |
|---|---|---|---|---|
| 31/03/25 | Interest |  | 12.00 | 1,00,012.00 |
| 01/04/24 | Salary |  | 1,00,000.00 | 1,00,000.00 |
"""

ITR = """<!-- page: 1 -->

### Acknowledgement Number : 111122223333444 Date of Filing : 28-Jul-2023*

| FORM | INDIAN INCOME TAX RETURN | Assessment Year |
|---|---|---|
| ITR1 | SAHAJ | 2023 - 24 |
| (A1) PAN ABCDE1234F | (A2) First Name ANITA | (A3) Last Name SHARMA |

<!-- page: 2 -->

| B4 | Gross Total Income (B1+B2+B3) | B4 | 8,53,332 |
|---|---|---|---|
| C1 | 80C - Life insurance premia | 1,50,000 | 1,50,000 |
| Total Income 6,75,330 |  |  |  |
| D14 | Refund (D12 - D11) | D14 | 40 |

<!-- page: 3 -->

### Acknowledgement Number : 555566667777888 Date of Filing : 25-Jul-2024*

| FORM | INDIAN INCOME TAX RETURN | Assessment Year |
|---|---|---|
| ITR1 | SAHAJ | 2024-25 |

<!-- page: 4 -->

| Acknowledgement Number : 555566667777888 |  |
|---|---|
| Total Income (B4-C21) | 11,81,410 |
"""


def test_bank_accounts_and_balances():
    facts = extract_facts(BANK)
    assert facts["kind"] == "bank_statement"
    sbi, hdfc = facts["accounts"]
    assert (sbi["bank"], sbi["account_number"], sbi["pages"]) == ("State Bank of India", "00000012345678901", "1-2")
    assert sbi["holder"] == "Ms. ASHA RAO"
    assert sbi["transactions"] == 3  # continuation rows on page 2 are counted
    assert sbi["closing_balance"] == "6,250.50"
    assert (sbi["total_debits"], sbi["total_credits"]) == ("4,000", "10,250.50")
    assert (hdfc["bank"], hdfc["pages"]) == ("HDFC Bank", "3")
    assert hdfc["closing_balance"] == "1,00,012"  # newest-first rows sorted by date
    assert summary(facts) == "2 accounts: State Bank of India …8901, HDFC Bank …7766"
    assert section_for_page(facts, 2)["account_number"] == "00000012345678901"


def test_itr_returns_split_by_acknowledgement():
    facts = extract_facts(ITR)
    assert facts["kind"] == "itr"
    first, second = facts["returns"]
    assert (first["form"], first["assessment_year"], first["pages"]) == ("ITR-1", "2023-24", "1-2")
    assert (first["pan"], first["name"], first["date_of_filing"]) == ("ABCDE1234F", "ANITA SHARMA", "28-Jul-2023")
    assert first["amounts"]["gross_total_income"] == "8,53,332"
    assert first["amounts"]["total_income"] == "6,75,330"  # label and value in one cell
    assert first["amounts"]["deduction_80C"] == "1,50,000"
    assert first["amounts"]["refund"] == "40"
    assert (second["assessment_year"], second["pages"]) == ("2024-25", "3-4")
    assert second["amounts"]["total_income"] == "11,81,410"
    assert section_for_page(facts, 4)["assessment_year"] == "2024-25"
    text = render(facts, "itr.pdf")
    assert "2 income tax return(s)" in text and "Total (taxable) income: 11,81,410" in text


SUMMARY_BANK = """<!-- page: 1 -->

ACCOUNT NO : 10012345678
STATEMENT PERIOD : 2025-04-01 TO 2026-03-31
Opening Balance Total Debit Total Credit Closing Balance
(1,000.00) 500.00 200.00 (1,300.00)

| Transaction Date | Particulars | Debit | Credit | Balance |
|---|---|---|---|---|
| 01-Apr-2025 | UPI | 500.00 |  | (1,500.00) |
| 02-Apr-2025 | NEFT |  | 200.00 | (1,300.00) |
| 10-Apr-2824 | misread row |  | 1.00 | (1,299.00) |
"""


def test_printed_summary_wins_and_bad_dates_are_dropped():
    acct = extract_facts(SUMMARY_BANK)["accounts"][0]
    assert acct["period"] == "2025-04-01 to 2026-03-31"
    assert acct["statement_summary"] == {"opening_balance": "-1,000", "total_debits": "500",
                                         "total_credits": "200", "closing_balance": "-1,300"}
    assert acct["transactions"] == 2                       # the year-2824 row is outside the period
    assert acct["closing_balance"] == "-1,300"             # computed agrees with printed: no warning
    assert "warnings" not in acct
    assert acct["row_check"].startswith("100% of 1 rows")


def test_other_documents_have_no_facts():
    assert extract_facts("<!-- page: 1 -->\n\n# Lease deed\n\nMonthly rent 32,000") is None


def test_fmt_inr():
    assert fmt_inr(30748828.62) == "3,07,48,828.62"
    assert fmt_inr(1000.0) == "1,000"
    assert fmt_inr(-512.5) == "-512.50"


def test_est_tokens_counts_digits():
    assert est_tokens("12,480.60") == 9           # every digit and separator is a token
    assert est_tokens("closing balance") == 4     # letter runs ~4 chars/token: 2 + 2
