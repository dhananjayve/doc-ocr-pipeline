import io

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from finpipe.detect import PageKind, PageStats, Shard, classify, classify_pdf, plan_shards

D, H, S = PageKind.DIGITAL, PageKind.HYBRID, PageKind.SCANNED


def _png() -> ImageReader:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 500), "white").save(buf, "PNG")
    buf.seek(0)
    return ImageReader(buf)


@pytest.fixture
def mixed_pdf(tmp_path):
    path = tmp_path / "mixed.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    # page 1: digital statement
    for i in range(20):
        c.drawString(72, 700 - i * 14, f"Revenue line {i}: $1,{i:03d}.50  Net income (2,{i:03d})")
    c.showPage()
    # page 2: scanned (image only)
    c.drawImage(_png(), 0, 0, width=w, height=h)
    c.showPage()
    # page 3: digital text + large embedded image (e.g. a scanned exhibit)
    c.drawString(72, 750, "Exhibit 99.1 - see scanned statement below. " * 2)
    c.drawImage(_png(), 50, 50, width=w - 100, height=h - 150)
    c.showPage()
    c.save()
    return path


def test_classify_pdf_routes_each_page(mixed_pdf, settings):
    assert classify_pdf(mixed_pdf, settings) == [D, S, H]


def test_classify_thresholds(settings):
    assert classify(PageStats(5, 0.0, 0.0), settings) is S
    assert classify(PageStats(500, 0.9, 0.0), settings) is S  # broken font encoding
    assert classify(PageStats(500, 0.0, 0.8), settings) is H
    assert classify(PageStats(500, 0.0, 0.1), settings) is D


def test_plan_shards_groups_and_caps():
    kinds = [D] * 5 + [S] * 2 + [D] * 3
    assert plan_shards(kinds, 3) == [Shard(D, 1, 3), Shard(D, 4, 5), Shard(S, 6, 7), Shard(D, 8, 10)]
    assert plan_shards(kinds, 0) == [Shard(D, 1, 5), Shard(S, 6, 7), Shard(D, 8, 10)]
    assert plan_shards([], 10) == []


def test_plan_shards_balances_across_workers():
    shards = plan_shards([D] * 51, 24, workers=4)
    assert [s.pages for s in shards] == [13, 13, 13, 12]
    assert (shards[0].start, shards[-1].end) == (1, 51)
    assert [s.pages for s in plan_shards([D] * 6, 24, workers=4)] == [6]  # min_pages keeps tiny docs whole


def test_scanned_pages_spread_over_workers():
    shards = plan_shards([D] * 548 + [S] * 10, 24, workers=4)
    scanned = [s for s in shards if s.kind is S]
    assert [s.pages for s in scanned] == [3, 3, 2, 2]
    assert (scanned[0].start, scanned[-1].end) == (549, 558)
    assert [s.pages for s in plan_shards([S] * 3, 24, workers=4)] == [3]  # too few pages to split
