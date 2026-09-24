"""Docling conversion, run inside worker processes.

Each worker builds (lazily, once) one DocumentConverter per page kind:
  digital -> do_ocr=False: text straight from the PDF font layer, TableFormer for table structure
  hybrid  -> do_ocr=True:  OCR only on bitmap regions, native text everywhere else
  scanned -> do_ocr=True, force_full_page_ocr=True
The default OCR engine is RapidOCR, which runs PaddleOCR's PP-OCR models on ONNX Runtime.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from finpipe.config import Settings
from finpipe.detect import PageKind, Shard

log = logging.getLogger(__name__)

PAGE_MARKER = "<!-- page: {} -->"

_settings: Settings | None = None
_converters: dict[PageKind, object] = {}


@dataclass
class ShardResult:
    doc_id: str
    shard: Shard
    pages: list[tuple[int, str]]  # (absolute 1-based page number, markdown)
    status: str
    error: str | None = None
    fallback: list[int] = field(default_factory=list)  # pages the fast route hands back to Docling


def init_worker(settings: Settings) -> None:
    """ProcessPool initializer: pin thread counts before torch/onnxruntime are imported."""
    global _settings
    _settings = settings
    threads = str(settings.resolved_threads())
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = threads
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)


def _ocr_options(settings: Settings, force_full_page: bool):
    from docling.datamodel import pipeline_options as po

    kwargs: dict = {}
    if hasattr(po, "OcrMode"):  # docling >= 2.1xx
        # PDF_AWARE_LAYOUT_REGIONS skips clusters already covered by native PDF text cells.
        kwargs["mode"] = po.OcrMode.FULL_PAGE if force_full_page else po.OcrMode.PDF_AWARE_LAYOUT_REGIONS
    else:
        kwargs["force_full_page_ocr"] = force_full_page
    if settings.ocr_lang:
        kwargs["lang"] = settings.ocr_lang
    if settings.ocr_engine == "easyocr":
        return po.EasyOcrOptions(**kwargs)
    if settings.ocr_engine == "tesseract":
        return po.TesseractCliOcrOptions(**kwargs)
    return po.RapidOcrOptions(**kwargs)


def _accelerator(settings: Settings):
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    except ImportError:  # docling < 2.40
        from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
    return AcceleratorOptions(
        num_threads=settings.resolved_threads(),
        device=AcceleratorDevice(settings.device),
    )


def build_converter(settings: Settings, kind: PageKind):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    accurate = settings.table_mode == "accurate" or (
        settings.table_mode == "auto" and kind is PageKind.SCANNED
    )
    opts.table_structure_options.mode = TableFormerMode.ACCURATE if accurate else TableFormerMode.FAST
    # Fill TableFormer's predicted cells from the page's text cells (the PDF font layer on digital pages,
    # OCR output on scanned ones). Must stay on for every kind: without it scanned tables come out empty.
    opts.table_structure_options.do_cell_matching = True
    opts.do_ocr = kind is not PageKind.DIGITAL
    if opts.do_ocr:
        opts.ocr_options = _ocr_options(settings, force_full_page=kind is PageKind.SCANNED)
    opts.accelerator_options = _accelerator(settings)
    if settings.document_timeout:
        opts.document_timeout = settings.document_timeout

    fmt_kwargs: dict = {"pipeline_options": opts}
    if settings.pdf_backend == "pypdfium":
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        fmt_kwargs["backend"] = PyPdfiumDocumentBackend
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(**fmt_kwargs)})


def _converter(kind: PageKind):
    if kind not in _converters:
        assert _settings is not None, "init_worker() was not called"
        _converters[kind] = build_converter(_settings, kind)
    return _converters[kind]


def warmup(kinds: tuple[PageKind, ...] = (PageKind.DIGITAL,)) -> int:
    """Build converters and load their models ahead of the first real job; returns the worker pid."""
    from docling.datamodel.base_models import InputFormat

    for kind in kinds:
        _converter(kind).initialize_pipeline(InputFormat.PDF)
    return os.getpid()


def _page_markdown(document, shard: Shard) -> list[tuple[int, str]]:
    page_nos = sorted(document.pages.keys())
    if not page_nos:
        return [(shard.start, document.export_to_markdown())]
    # Docling keeps original page numbers for page_range conversions; fall back to offsets if not.
    in_range = all(shard.start <= p <= shard.end for p in page_nos)
    out = []
    for i, p in enumerate(page_nos):
        absolute = p if in_range else shard.start + i
        out.append((absolute, document.export_to_markdown(page_no=p)))
    return out


def parse_shard(path: str, doc_id: str, shard: Shard) -> ShardResult:
    assert _settings is not None, "init_worker() was not called"
    if shard.kind is PageKind.DIGITAL and shard.engine != "docling" and _settings.digital_engine == "fast":
        return _fast_shard(path, doc_id, shard)
    return _docling_shard(path, doc_id, shard)


def _fast_shard(path: str, doc_id: str, shard: Shard) -> ShardResult:
    from finpipe.fastpdf import extract_pages

    try:
        pages, fallback = extract_pages(path, shard.start, shard.end)
        return ShardResult(doc_id, shard, pages, "fast", fallback=fallback)
    except Exception as exc:  # anything unexpected: let Docling do the whole shard
        log.warning("fast route failed on %s:%s-%s (%s); using Docling", doc_id, shard.start, shard.end, exc)
        return ShardResult(doc_id, shard, [], "fast-failed", fallback=list(range(shard.start, shard.end + 1)))


def _docling_shard(path: str, doc_id: str, shard: Shard) -> ShardResult:
    from docling.datamodel.base_models import ConversionStatus

    try:
        result = _converter(shard.kind).convert(
            path, raises_on_error=False, page_range=(shard.start, shard.end)
        )
        status = result.status
        if status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            errors = "; ".join(str(e.error_message) for e in (result.errors or []))
            return ShardResult(doc_id, shard, [], str(status), errors or "conversion failed")
        return ShardResult(doc_id, shard, _page_markdown(result.document, shard), str(status))
    except Exception as exc:  # keep the pool alive; the pipeline reports the failure
        log.exception("shard %s:%s-%s failed", doc_id, shard.start, shard.end)
        return ShardResult(doc_id, shard, [], "exception", f"{type(exc).__name__}: {exc}")


def assemble_markdown(pages: list[tuple[int, str]]) -> str:
    """Join per-page markdown with page markers the chunker uses for citations."""
    parts = []
    for page_no, md in sorted(pages):
        parts.append(PAGE_MARKER.format(page_no))
        parts.append(md.strip())
    return "\n\n".join(parts) + "\n"
