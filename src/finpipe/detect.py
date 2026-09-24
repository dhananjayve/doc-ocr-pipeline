"""Per-page triage of a PDF into digital / hybrid / scanned, and planning of parse shards.

Classifying every page (not the whole file) lets one document mix routes: native text-layer
extraction for digital pages (exact decimals, no OCR cost), region OCR for digital pages that
embed scanned images, and full-page OCR for scanned pages.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from finpipe.config import Settings


class PageKind(str, Enum):
    DIGITAL = "digital"   # trustworthy text layer: no OCR
    HYBRID = "hybrid"     # text layer + large embedded images: OCR only bitmap regions
    SCANNED = "scanned"   # no / garbage text layer: full-page OCR


@dataclass(frozen=True)
class PageStats:
    text_chars: int
    garbage_ratio: float
    image_coverage: float


@dataclass(frozen=True)
class Shard:
    kind: PageKind
    start: int  # 1-based, inclusive
    end: int    # 1-based, inclusive
    engine: str = "auto"  # "auto" follows settings; "docling" forces Docling (fast-route fallback)

    @property
    def pages(self) -> int:
        return self.end - self.start + 1


def _garbage_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(
        1 for c in chars
        if c == "�" or unicodedata.category(c) in ("Co", "Cn", "Cc")
    )
    return bad / len(chars)


BACKGROUND_TEXT_CHARS = 100  # an image with this much native text printed over it is a background


def _image_coverage(page: pdfium.PdfPage, textpage: pdfium.PdfTextPage) -> float:
    """Share of the page covered by images that may hold unread text.

    Watermarks, logos and form backgrounds (e.g. the emblem behind every ITR page) carry the real text
    on top of them in the text layer; OCR-ing them only costs time, so they don't count.
    """
    width, height = page.get_size()
    page_area = max(width * height, 1.0)
    area = 0.0
    for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=3):
        bounds = obj.get_bounds if hasattr(obj, "get_bounds") else obj.get_pos  # pypdfium2 v5 / v4
        left, bottom, right, top = bounds()
        overlaid = len("".join(textpage.get_text_bounded(left, bottom, right, top).split()))
        if overlaid >= BACKGROUND_TEXT_CHARS:
            continue
        area += max(0.0, right - left) * max(0.0, top - bottom)
    return min(area / page_area, 1.0)


def page_stats(page: pdfium.PdfPage) -> PageStats:
    textpage = page.get_textpage()
    try:
        stripped = "".join(textpage.get_text_range().split())
        coverage = _image_coverage(page, textpage)
    finally:
        textpage.close()
    return PageStats(
        text_chars=len(stripped),
        garbage_ratio=_garbage_ratio(stripped),
        image_coverage=coverage,
    )


def classify(stats: PageStats, settings: Settings) -> PageKind:
    if stats.text_chars < settings.min_text_chars or stats.garbage_ratio > settings.garbage_ratio:
        return PageKind.SCANNED
    if stats.image_coverage > settings.hybrid_image_coverage:
        return PageKind.HYBRID
    return PageKind.DIGITAL


def classify_pdf(path: str | Path, settings: Settings) -> list[PageKind]:
    pdf = pdfium.PdfDocument(str(path))
    try:
        kinds = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                kinds.append(classify(page_stats(page), settings))
            finally:
                page.close()
        return kinds
    finally:
        pdf.close()


def plan_shards(kinds: list[PageKind], max_pages: int, workers: int = 1, min_pages: int = 4) -> list[Shard]:
    """Group contiguous pages of the same kind into evenly sized shards.

    The shard size is capped at `max_pages` and, when there are enough pages, at pages/workers so every
    worker gets a share (51 pages on 4 workers -> 13+13+13+12, not 24+24+3). Shards never drop below
    `min_pages` because each one carries some fixed per-call overhead.
    """
    if not kinds:
        return []
    limit = max_pages if max_pages > 0 else len(kinds)
    if workers > 1:
        limit = min(limit, max(min_pages, math.ceil(len(kinds) / workers)))
    runs: list[tuple[PageKind, int, int]] = []
    start = 1
    for i in range(1, len(kinds) + 1):
        if i == len(kinds) or kinds[i] != kinds[i - 1]:
            runs.append((kinds[i - 1], start, i))
            start = i + 1
    shards: list[Shard] = []
    for kind, first, last in runs:
        length = last - first + 1
        n = math.ceil(length / limit)
        if workers > 1 and kind is not PageKind.DIGITAL:
            # OCR pages cost ~20 s each: spread every OCR run over all workers (2+ pages per shard),
            # instead of letting e.g. 10 scanned pages run on one worker while the others sit idle.
            n = max(n, min(workers, length // 2 or 1))
        elif workers > 1:
            n = max(1, min(n, length // min_pages))
        base, extra = divmod(length, n)
        s = first
        for j in range(n):
            size = base + (1 if j < extra else 0)
            shards.append(Shard(kind=kind, start=s, end=s + size - 1))
            s += size
    return shards
