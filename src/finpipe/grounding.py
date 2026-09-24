"""Numeric grounding check: flag numbers in an SLM answer that don't appear in the retrieved sources.

Small models occasionally transpose or invent digits. Every figure in the answer is normalised
("$(1,234.50)" -> -1234.5) and looked up in the numbers found in the context. Figures that are not
found verbatim are reported as unverified; they may be legitimate calculations (growth rates,
sums), so they are flagged, not removed.
"""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]")
NUMBER_RE = re.compile(r"\(?-?[$€£₹¥]?\s?\d[\d,]*(?:\.\d+)?\)?%?")


def normalize(token: str) -> float | None:
    t = re.sub(r"[$€£₹¥\s]", "", token)
    negative = (t.startswith("(") and t.endswith(")")) or t.startswith("-") or t.startswith("(-")
    digits = re.sub(r"[^\d.]", "", t)
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def extract_numbers(text: str) -> dict[float, str]:
    out: dict[float, str] = {}
    for m in NUMBER_RE.finditer(CITATION_RE.sub(" ", text)):
        raw = m.group(0).strip().rstrip(",")
        value = normalize(raw)
        if value is not None:
            out.setdefault(value, raw)
    return out


def unverified_numbers(answer: str, context: str, ignore_below: float = 10) -> list[str]:
    """Numbers in `answer` absent from `context` (ignoring small integers like list ordinals)."""
    ctx = extract_numbers(context)
    ctx_abs = {abs(v) for v in ctx}
    flagged = []
    for value, raw in extract_numbers(answer).items():
        if abs(value) < ignore_below and float(value).is_integer():
            continue
        if value in ctx or abs(value) in ctx_abs:  # sign conventions differ: (1,200) vs "decrease of 1,200"
            continue
        flagged.append(raw)
    return flagged
