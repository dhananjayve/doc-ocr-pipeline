"""Document categories and types. Drives the upload dropdown, chunk metadata, model routing and
type-specific instructions added to the SLM prompt."""

from __future__ import annotations

from dataclasses import asdict, dataclass

CATEGORIES = {"finance": "Finance", "property": "Property", "other": "Other"}


@dataclass(frozen=True)
class DocType:
    key: str
    category: str
    label: str
    hint: str = ""


DOC_TYPES: dict[str, DocType] = {t.key: t for t in [
    # --- finance -------------------------------------------------------------------
    DocType("bank_statement", "finance", "Bank statement (single or multi-bank combined)",
            "This file may combine statements from several banks and accounts. Identify each bank and account "
            "number separately and never mix transactions or balances across accounts. Withdrawals/debits reduce "
            "the balance, deposits/credits increase it. Give dates exactly as printed."),
    DocType("ais", "finance", "Annual Information Statement (AIS / TIS)",
            "Indian Income Tax AIS/TIS: amounts are grouped by information category (TDS/TCS, SFT, interest, "
            "dividend, securities, etc.) and may show both 'reported by source' and 'accepted/modified' values. "
            "Say which value you report and the financial year it belongs to."),
    DocType("form16", "finance", "Form 16 / 16A (TDS certificate)",
            "Form 16 Part A holds employer TAN/PAN and quarter-wise TDS deducted and deposited; Part B holds the "
            "salary breakup, exemptions u/s 10, standard deduction, Chapter VI-A deductions (80C, 80D, ...), total "
            "taxable income and tax payable. Always state the assessment year."),
    DocType("form26as", "finance", "Form 26AS (tax credit statement)",
            "Form 26AS lists TDS/TCS by deductor TAN, section and transaction date, plus advance/self-assessment "
            "tax and refunds. Report deductor name and section with each amount."),
    DocType("itr", "finance", "Income tax return / ITR-V acknowledgement",
            "Report the ITR form, assessment year, gross total income, deductions, total income, tax paid and "
            "refund or demand exactly as printed."),
    DocType("salary_slip", "finance", "Salary slip",
            "Separate earnings from deductions; distinguish gross pay from net pay and state the pay period."),
    DocType("invoice", "finance", "Invoice / GST bill",
            "Report GSTIN, invoice number and date, taxable value, CGST/SGST/IGST and invoice total separately."),
    DocType("annual_report", "finance", "Annual report / financial statements",
            "Confirm the period column and the unit (thousands, lakhs, crores, millions) before reporting a figure."),
    DocType("finance_other", "finance", "Other financial document"),
    # --- property -----------------------------------------------------------------
    DocType("sale_deed", "property", "Sale deed / title deed",
            "Identify seller(s), buyer(s), property description and boundaries, survey/plot number, area with "
            "units, consideration amount, stamp duty, registration number and date."),
    DocType("land_record", "property", "Land record (khasra, khatauni, 7/12, jamabandi)",
            "Report survey/khasra number, owner names and shares, area with units, land classification and any "
            "mutation or encumbrance entries. Keep the local-language terms as printed."),
    DocType("lease", "property", "Lease / rent agreement",
            "Identify lessor, lessee, premises, term with start and end dates, rent, escalation, security deposit, "
            "lock-in, notice period and termination clauses."),
    DocType("encumbrance", "property", "Encumbrance certificate",
            "List every registered transaction with document number, date, parties and nature, in date order; "
            "say explicitly if the certificate reports nil encumbrance."),
    DocType("property_tax", "property", "Property tax receipt / assessment",
            "Report property ID, owner, assessment year, annual value, tax amount, penalties and payment date."),
    DocType("property_other", "property", "Other property document"),
    # --- other --------------------------------------------------------------------
    DocType("other", "other", "Other document"),
]}


def get(key: str | None) -> DocType:
    return DOC_TYPES.get(key or "other", DOC_TYPES["other"])


def as_options() -> list[dict]:
    """Dropdown payload: categories with their document types."""
    return [
        {"category": cat, "label": label,
         "types": [asdict(t) for t in DOC_TYPES.values() if t.category == cat]}
        for cat, label in CATEGORIES.items()
    ]
