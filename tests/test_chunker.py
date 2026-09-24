from finpipe.chunker import MarkdownChunker, est_tokens, parse_blocks, split_table
from finpipe.parser import assemble_markdown


def balance_sheet(rows: int) -> str:
    lines = ["| Item | 2024 | 2023 |", "|---|---:|---:|"]
    lines += [f"| Line item {i} | {i * 1000:,}.00 | ({i * 900:,}) |" for i in range(rows)]
    return "\n".join(lines)


def test_page_markers_and_tables_parsed():
    md = assemble_markdown([(2, "## Cash\n\n| a | b |\n|---|---|\n| 1 | 2 |"), (1, "# Report\n\nIntro text.")])
    blocks = parse_blocks(md)
    assert [(b.kind, b.page) for b in blocks] == [
        ("heading", 1), ("text", 1), ("heading", 2), ("table", 2)
    ]


def test_table_never_split_and_keeps_heading_path():
    table = balance_sheet(30)
    md = f"# Annual Report\n\n## Balance Sheet\n\nAmounts in millions.\n\n{table}\n\n## Notes\n\n" + "Note text. " * 80
    chunks = MarkdownChunker(max_tokens=400, min_tokens=50, table_max_tokens=5000).chunk(md)
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) == 1
    assert table in table_chunks[0].text
    assert table_chunks[0].heading_path == ["Annual Report", "Balance Sheet"]
    assert "Amounts in millions." in table_chunks[0].text  # lead-in kept with its table


def test_oversized_table_split_repeats_header():
    table = balance_sheet(200)
    pieces = split_table(table, max_tokens=300)
    assert len(pieces) > 1
    header = "| Item | 2024 | 2023 |\n|---|---:|---:|"
    assert all(p.startswith(header) for p in pieces)
    assert sum(len(p.splitlines()) - 2 for p in pieces) == 200
    assert all(est_tokens(p) <= 320 for p in pieces)


def test_small_sections_are_packed_and_page_span_tracked():
    md = assemble_markdown([(1, "# A\n\nshort one."), (2, "# B\n\nshort two."), (3, "# C\n\nshort three.")])
    chunks = MarkdownChunker(max_tokens=500, min_tokens=100).chunk(md)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.page_start, c.page_end) == (1, 3)
    assert "# B" in c.text and "short three." in c.text


def test_long_text_split_under_limit():
    md = "# Risk Factors\n\n" + " ".join(f"Sentence number {i} about risk." for i in range(400))
    chunks = MarkdownChunker(max_tokens=200, min_tokens=50).chunk(md)
    assert len(chunks) > 3
    assert all(est_tokens(c.text) <= 220 for c in chunks)
    assert all(c.heading_path == ["Risk Factors"] for c in chunks)
    assert chunks[0].embed_text().startswith("[Risk Factors]")
