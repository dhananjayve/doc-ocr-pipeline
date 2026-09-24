"""Command line interface.

    finpipe serve                                   web UI + API on http://127.0.0.1:8765
    finpipe ingest ./statements --type bank_statement   parse + index PDFs/images (skips indexed files)
    finpipe ask "What was diluted EPS in FY2024?"   answer with citations and a numeric grounding check
    finpipe chat                                    interactive Q&A session
    finpipe parse report.pdf -o report.md           PDF -> page-tagged markdown only
    finpipe triage report.pdf                       show per-page digital/hybrid/scanned routing
    finpipe stats                                   what is in the index
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from finpipe.config import Settings


def _pipeline(settings: Settings):
    from finpipe.pipeline import Pipeline

    return Pipeline(settings)


def _rag(settings: Settings):
    from finpipe.rag import build_rag

    return build_rag(settings)


def cmd_ingest(args, settings: Settings) -> int:
    t0 = time.perf_counter()
    reports = _pipeline(settings).ingest(args.paths, force=args.force, doc_type=args.type)
    failed = 0
    print(f"\n{'status':<9} {'pages':>5} {'chunks':>6} {'secs':>7}  file  [page routing]")
    for r in reports:
        failed += r.status == "failed"
        kinds = " ".join(f"{k}={v}" for k, v in r.kinds.items())
        print(f"{r.status:<9} {r.pages:>5} {r.chunks:>6} {r.seconds:>7.1f}  {Path(r.path).name}  [{kinds}]")
        if r.error:
            print(f"          ! {r.error}")
    pages = sum(r.pages for r in reports if r.status in ("ingested", "cached"))
    elapsed = time.perf_counter() - t0
    print(f"\n{len(reports)} file(s), {pages} page(s) in {elapsed:.1f}s"
          + (f" ({pages / elapsed:.2f} pages/s)" if pages else ""))
    return 1 if failed else 0


def _print_answer_footer(answer) -> None:
    from finpipe.rag import source_label

    print("\n\nSources:")
    for i, hit in enumerate(answer.sources, 1):
        print(f"  [{i}] {source_label(hit)}  (score {hit.score:.3f})")
    if answer.unverified:
        print("\nNumbers not found verbatim in the sources (check them, they may be calculations): "
              + ", ".join(answer.unverified))


def _ask(rag, question: str, args) -> None:
    from finpipe.rag import Answer, Prepared

    stream = rag.ask_stream(question, k=args.k, category=args.category, use_model=args.model_only,
                            source=args.source, year=args.year)
    for piece in stream:
        if isinstance(piece, Prepared):
            print(f"(reading {piece.sources} sources, ~{piece.context_tokens} tokens)\n", flush=True)
        elif isinstance(piece, Answer):
            if not piece.sources:
                print(piece.text, end="")
            _print_answer_footer(piece)
            if piece.route == "finance":
                print("(answered by the finance model)")
        else:
            print(piece, end="", flush=True)
    print()


def cmd_ask(args, settings: Settings) -> int:
    _ask(_rag(settings), args.question, args)
    return 0


def cmd_chat(args, settings: Settings) -> int:
    rag = _rag(settings)
    print("finpipe chat — ask about your documents (empty line or Ctrl+C to quit)")
    try:
        while question := input("\n? ").strip():
            _ask(rag, question, args)
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


def cmd_parse(args, settings: Settings) -> int:
    report, markdown = _pipeline(settings).parse_to_markdown(args.pdf)
    if report.status == "failed":
        print(f"failed: {report.error}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"{report.pages} pages {report.kinds} -> {args.output} ({report.seconds:.1f}s)")
    else:
        sys.stdout.write(markdown)
    return 0


def cmd_triage(args, settings: Settings) -> int:
    from finpipe.detect import classify_pdf, plan_shards

    kinds = classify_pdf(args.pdf, settings)
    for shard in plan_shards(kinds, settings.max_pages_per_shard, settings.resolved_workers()):
        print(f"pages {shard.start:>4}-{shard.end:<4} {shard.kind.value}")
    return 0


def cmd_serve(args, settings: Settings) -> int:
    from finpipe.server import serve

    serve(settings, host=args.host, port=args.port)
    return 0


def cmd_stats(args, settings: Settings) -> int:
    from finpipe.store import VectorStore

    store = VectorStore(settings.chroma_dir, settings.collection)
    sources = store.sources()
    print(f"collection '{settings.collection}' at {settings.chroma_dir}: {store.count()} chunks, {len(sources)} document(s)")
    for name, n in sorted(sources.items()):
        print(f"  {n:>6}  {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="finpipe", description="Local financial PDF pipeline with a small language model")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--model", help="SLM model name (overrides FINPIPE_SLM_MODEL)")
    p.add_argument("--workers", type=int, help="parser processes (overrides FINPIPE_WORKERS)")
    p.add_argument("--data-dir", help="cache + index directory (overrides FINPIPE_DATA_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)

    from finpipe.doctypes import CATEGORIES, DOC_TYPES

    s = sub.add_parser("serve", help="run the web UI and API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ingest", help="parse and index PDFs / images / folders")
    s.add_argument("paths", nargs="+")
    s.add_argument("--type", default="other", choices=sorted(DOC_TYPES), help="document type tag")
    s.add_argument("--force", action="store_true", help="re-index files already in the store")
    s.set_defaults(func=cmd_ingest)

    for name, func in (("ask", cmd_ask), ("chat", cmd_chat)):
        s = sub.add_parser(name, help=f"{name} over indexed documents")
        if name == "ask":
            s.add_argument("question")
        s.add_argument("-k", type=int, default=None, help="chunks to retrieve")
        s.add_argument("--source", help="restrict to one file name, e.g. 10-K_2024.pdf")
        s.add_argument("--category", choices=sorted(CATEGORIES), help="restrict to a category")
        s.add_argument("--year", help="restrict to an assessment year, e.g. 2024-25")
        s.add_argument("--model-only", action="store_true",
                       help="always ask the model, even when stored facts answer the question")
        s.set_defaults(func=func)

    s = sub.add_parser("parse", help="convert one PDF to page-tagged markdown")
    s.add_argument("pdf")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_parse)

    s = sub.add_parser("triage", help="show per-page routing (digital / hybrid / scanned)")
    s.add_argument("pdf")
    s.set_defaults(func=cmd_triage)

    sub.add_parser("stats", help="show index contents").set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = {k: v for k, v in {
        "slm_model": args.model, "workers": args.workers, "data_dir": args.data_dir,
    }.items() if v is not None}
    settings = Settings(**overrides)
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
