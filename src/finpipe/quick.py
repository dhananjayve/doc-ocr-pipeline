"""Instant answers from stored document facts, with no language model.

Questions like "how many accounts?", "list the accounts with closing balances", "top debits and
credits" or "total income and refund for AY 2024-25" are fully answered by the facts read at upload
(see facts.py). Formatting those facts takes milliseconds; having the model copy them out took
minutes on CPU/iGPU and risked copying mistakes. Anything that needs reasoning, a specific
transaction or a date lookup still goes to the model.
"""

from __future__ import annotations

import re

from finpipe.facts import ITR_LABELS

# Questions that need reasoning or a specific line are never answered from facts alone.
MODEL_ONLY_RE = re.compile(
    r"(?i)\b(why|explain|reason|how come|analy[sz]e|compare|trend|pattern|suspicious|unusual|should|advice|"
    r"recommend|predict|summari[sz]e the transactions|narration|to whom|from whom|which transaction|"
    r"on \d{1,2}[/\-.]|between|after|before|salary from|paid to|received from)\b"
    r"|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
)
BANK_RE = re.compile(
    r"(?i)\b(accounts?|a/cs?|banks?|closing balance|balances?|top|largest|biggest|highest|major|total debits?|"
    r"total credits?|debits?|credits?|withdrawals?|deposits?|statement period|period|holders?|how many)\b")
TOP_RE = re.compile(r"(?i)\b(top|largest|biggest|highest|major|maximum|max)\b")
ITR_RE = re.compile(
    r"(?i)\b(income|tax|refund|deductions?|80c|80d|returns?|itr|assessment year|ay|pan|filing|filed|"
    r"acknowledg\w*|taxable|taxes paid|payable)\b")
AY_Q_RE = re.compile(r"(?i)\b(?:ay|a\.y\.|assessment year)?\s*(20\d{2})\s*[-/]\s*(\d{2}|20\d{2})\b")


def _year_in(question: str) -> str | None:
    m = AY_Q_RE.search(question)
    return f"{m.group(1)}-{m.group(2)[-2:]}" if m else None


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) if c not in (None, "") else "—" for c in r) + " |" for r in rows]
    return "\n".join(out)


def _short(text: str, n: int = 48) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def bank_answer(source: str, facts: dict, question: str) -> str:
    accts = facts["accounts"]
    parts = [f"**{source}** has **{len(accts)} account{'s' if len(accts) != 1 else ''}**:", ""]

    def figure(a: dict, key: str) -> str | None:
        # the statement's own printed summary wins over figures computed from the rows
        printed = (a.get("statement_summary") or {}).get(key)
        return f"{printed} (printed)" if printed else a.get(key)

    parts.append(_table(
        ["#", "Bank", "Account no.", "Holder", "Period", "Txns", "Closing balance", "Total debits", "Total credits",
         "Row check", "Pages"],
        [[i, a.get("bank") or "Unknown", a["account_number"], a.get("holder"), a.get("period"), a.get("transactions"),
          figure(a, "closing_balance"), figure(a, "total_debits"), figure(a, "total_credits"),
          (a.get("row_check") or "").split(" (")[0], a["pages"]]
         for i, a in enumerate(accts, 1)],
    ))
    warnings = [f"- **{a.get('bank') or 'Account'} …{a['account_number'][-4:]}**: {w}"
                for a in accts for w in a.get("warnings", [])]
    if warnings:
        parts += ["", "**Check before relying on these figures:**", *warnings]
    if TOP_RE.search(question):
        want_debit = re.search(r"(?i)debit|withdraw|spent|paid", question) or not re.search(r"(?i)credit|deposit", question)
        want_credit = re.search(r"(?i)credit|deposit|receiv", question) or not re.search(r"(?i)debit|withdraw", question)
        for kind, label, wanted in (("largest_debits", "Largest debits", want_debit),
                                    ("largest_credits", "Largest credits", want_credit)):
            if not wanted:
                continue
            rows = [[f"{a.get('bank') or 'Unknown'} …{a['account_number'][-4:]}", t["date"], t["amount"], _short(t["narration"])]
                    for a in accts for t in a.get(kind, [])]
            if rows:
                parts += ["", f"**{label}** (top 3 per account):", "", _table(["Account", "Date", "Amount", "Description"], rows)]
    return "\n".join(parts)


def itr_answer(source: str, facts: dict, question: str, year: str | None) -> str | None:
    year = _year_in(question) or year
    returns = [r for r in facts["returns"] if not year or r.get("assessment_year") == year]
    if not returns:
        return None
    returns = sorted(returns, key=lambda r: r.get("assessment_year", ""))
    keys = [k for k in ITR_LABELS if any(k in r.get("amounts", {}) for r in returns)]
    head = ["Field"] + [f"AY {r.get('assessment_year', '?')}" for r in returns]
    rows = [["Form"] + [r.get("form") for r in returns],
            ["Name"] + [r.get("name") for r in returns],
            ["PAN"] + [r.get("pan") for r in returns],
            ["Date of filing"] + [r.get("date_of_filing") for r in returns]]
    rows += [[ITR_LABELS[k]] + [r.get("amounts", {}).get(k) for r in returns] for k in keys]
    rows += [["Pages in file"] + [r["pages"] for r in returns]]
    scope = f"AY {year}" if year else f"{len(returns)} return{'s' if len(returns) != 1 else ''}"
    return f"**{source}** · {scope}:\n\n" + _table(head, rows)


def quick_answer(question: str, entries: list[dict], year: str | None = None) -> str | None:
    """Markdown answer built only from facts, or None when the question needs the model.

    `entries` are FactsStore records ({"source", "facts"}) for the documents in scope.
    """
    q = question.strip()
    if not q or MODEL_ONLY_RE.search(q):
        return None
    blocks = []
    for entry in entries:
        facts = entry.get("facts") or {}
        if facts.get("kind") == "bank_statement" and BANK_RE.search(q):
            blocks.append(bank_answer(entry["source"], facts, q))
        elif facts.get("kind") == "itr" and ITR_RE.search(q):
            if block := itr_answer(entry["source"], facts, q, year):
                blocks.append(block)
    if not blocks:
        return None
    note = ("\n\n_Read directly from the documents at upload; no model involved. "
            "Use **Ask the model** for explanations or specific transactions._")
    return "\n\n---\n\n".join(blocks) + note
