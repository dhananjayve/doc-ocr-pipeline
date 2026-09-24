# finpipe

A local pipeline for financial and property documents: digital PDFs, scanned PDFs and photos. Pages
are parsed with layout awareness and per-page OCR routing, split into markdown-aware chunks, searched
with hybrid retrieval, and answered by a small language model. An optional FinGPT route handles
finance documents.

## Setup (first time)

Requires Python 3.10+ and [Ollama](https://ollama.com).

```powershell
cd D:\py-apps\doc-pipeline-ocr
python -m venv .venv --system-site-packages   # reuses an existing torch install if you have one
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"                       # add ,fingpt to use FinGPT LoRA adapters
ollama pull qwen3:8b                          # default model (or use one you already have, see below)
```

The first parse downloads Docling's layout, table and OCR models, a few hundred MB.

## Running the application

```powershell
cd D:\py-apps\doc-pipeline-ocr
.\.venv\Scripts\Activate.ps1          # activate the project's virtual env
finpipe --model gemma4 serve          # UI at http://127.0.0.1:8765
```

Stop it with **Ctrl+C**.

- If PowerShell blocks the activate script, run the executable directly instead:
  `.\.venv\Scripts\finpipe.exe --model gemma4 serve`
- Port 8765 is the default because 8000 is often taken by other apps. If a finpipe server is already
  running on it, stop that one (`Get-Process finpipe | Stop-Process`) or pick another port:
  `finpipe serve --port 8766`.
- Leave out `--model` to use the default `qwen3:8b`.

## Web UI

1. Open http://127.0.0.1:8765.
2. Drop in PDFs or images (PNG, JPG, TIFF, BMP, WEBP) and pick a document type:
   - **Finance**: bank statement (single or multi-bank combined), AIS/TIS, Form 16/16A, Form 26AS, ITR, salary slip, invoice/GST, annual report
   - **Property**: sale/title deed, land record (khasra, khatauni, 7/12, jamabandi), lease/rent agreement, encumbrance certificate, property tax
   - **Other**
3. Wait for the job to show **Done**. Digital files take a minute or two; scanned files take longer
   because every page goes through OCR.
4. Ask questions, optionally limited to one category or one document. Answers stream in with cited
   sources and page numbers. Figures that don't appear word-for-word in the sources are flagged, so
   check them (they may be calculations).

Categories and types live in `src/finpipe/doctypes.py`. Each type carries instructions that are
added to the prompt, such as keeping accounts separate in a combined bank statement.

### Key facts and the Year filter

Bank statements and ITRs have fixed layouts, so their key facts are read by rules when the file is
uploaded, without a model (`src/finpipe/facts.py`). This makes questions about the whole document
accurate, where passage search alone would miss parts of it.

- **Bank statements**: one entry per account, including combined multi-bank files. Each entry has the
  bank, account number, holder, period, pages, transaction count, closing balance, total debits and
  credits, and the 3 largest debits and credits.
- **ITRs**: files often bundle several years' returns. They are split by acknowledgement number, and
  each return gets its form, assessment year, name, PAN, filing date and key amounts (gross and
  taxable income, deductions, tax, taxes paid, refund).

**Instant answers.** Questions the facts fully answer are answered from them in under a second,
without a model: account lists, counts, closing balances, largest debits and credits, and ITR
income, tax and refund by year. These are marked "document facts · no model". Click **Ask the model**
to send the same question to the model instead (CLI: `--model-only`). Questions that need reasoning
or a specific transaction ("why…", "on 30/01/24…", "narration of…") always go to the model
(`src/finpipe/quick.py`).

The server loads the embedding model, the index and the language model in the background at startup,
and Ollama keeps the model loaded for `FINPIPE_SLM_KEEP_ALIVE` (default `4h`). Model answers are asked
to be compact and are capped at `FINPIPE_SLM_MAX_TOKENS` (default 800), because writing is the
slowest step on a laptop.

The facts appear under each document in the **Indexed documents** list. They are also sent to the
model with every question about that document, cited as **[F]**. Every chunk is tagged with its
account or assessment year, so the **Assessment year** filter keeps each answer to one return.

## Command line

```powershell
# index files or a folder without the UI
finpipe ingest test_docs\ --type itr
finpipe ingest "test_docs\Sample Bank Statement for AI Analysis.pdf" --type bank_statement

# ask questions
finpipe --model gemma4 ask "What is the closing balance for each account?"
finpipe --model gemma4 ask "Total income declared?" --category finance
finpipe --model gemma4 chat                     # interactive Q&A

# inspect a single file
finpipe triage "test_docs\Detailed ITR-2 Sample.pdf"         # which pages are digital or scanned
finpipe parse "test_docs\Detailed ITR-1 Sample.pdf" -o itr1.md  # convert to markdown only

finpipe stats                                   # what's in the index
finpipe --help                                  # all options
```

Global flags go **before** the command name:

| Flag | Meaning |
|---|---|
| `--model <name>` | Ollama model that answers questions (default `qwen3:8b`) |
| `--workers N` | parser processes (default: up to 4, depending on CPU cores) |
| `--data-dir <path>` | where the index and parse cache are stored (default `.finpipe\`) |
| `-v` | detailed logs |

`ingest --type` accepts any document type key, for example `bank_statement`, `ais`, `form16`,
`form26as`, `itr`, `salary_slip`, `invoice`, `annual_report`, `sale_deed`, `land_record`, `lease`,
`encumbrance`, `property_tax` or `other`. Add `--force` to re-index a file that's already indexed.

## Configuration

Every setting in `src/finpipe/config.py` can be overridden with a `FINPIPE_<NAME>` environment
variable or a `.env` file in the project folder. For example, to avoid typing `--model` every time:

```
FINPIPE_SLM_MODEL=gemma4
```

Other useful settings:

| Setting | Default | Meaning |
|---|---|---|
| `FINPIPE_WORKERS` | auto | parser processes (each holds its own Docling models, ~2 GB RAM) |
| `FINPIPE_DIGITAL_ENGINE` | `fast` | `fast` = geometric table extraction on digital pages; `docling` = every page through Docling |
| `FINPIPE_OCR_ENGINE` | `rapidocr` | `rapidocr`, `easyocr` or `tesseract` |
| `FINPIPE_TABLE_MODE` | `auto` | Docling table model: fast on digital pages, accurate on scanned ones |
| `FINPIPE_SLM_BACKEND` | `ollama` | `ollama`, `openai` (any OpenAI-compatible server) or `hf` (local transformers) |
| `FINPIPE_SLM_BASE_URL` | `http://localhost:11434` | model server address |
| `FINPIPE_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model |
| `FINPIPE_TOP_K` | `6` | chunks retrieved per question |

## How documents are processed

Every page is classified before parsing, so one file can mix routes:

| Page kind | Detected by | Parsed with |
|---|---|---|
| digital | a usable text layer | **fast route** (~0.2–1 s/page, exact digits from the text layer). Ruled tables are rebuilt from their lines with `pdfplumber`. Unruled transaction tables are rebuilt from the header's column positions and the running balance, and are accepted only if at least 80% of rows reconcile (previous balance − debit + credit = balance) (`src/finpipe/unruled.py`). Pages that fail this go to Docling. |
| hybrid | text plus images covering >35% of the page that have no text printed over them | Docling, with OCR only on the image regions |
| scanned | no or garbled text layer, or an uploaded image | Docling with full-page OCR (RapidOCR, which runs PaddleOCR's PP-OCR models on ONNX) and the accurate table model |

Background images with text printed over them, such as the emblem on every ITR page, are treated as
watermarks and don't trigger OCR.

Pages are grouped into batches that run in parallel worker processes, and batches from all files
share the same workers. Scanned pages are spread over every worker, because OCR costs ~20 s per page.

For table chunks, the embedding model sees the words only: payees, narrations, labels and headings.
Number runs are dropped, which keeps every chunk inside the model's 512-token limit. The stored text,
keyword matching and answers still use the full text with all figures.

Bank statement facts prefer the figures the statement prints itself (opening and closing balance,
total debit and credit) over totals computed from the rows. Each account gets a **row check** (the
share of rows whose balances reconcile) and warnings when the two disagree, for example when a file
joins overlapping statements, or when a scanned section's columns couldn't be read.

Measured: a 558-page statement (548 digital pages, many unruled, plus 10 scanned) took ~3.6 minutes
end to end on the laptop above. Before the unruled-table reader, it took over 40 minutes. Parsed output is cached by file content, so an unchanged file is never parsed
twice. The server keeps its workers, and their loaded models, alive between uploads.

On the sample files (bank statement plus three ITRs, 356 digital pages), ingestion took about 150 s
on an 18-thread CPU with no GPU.

## Which model

- **Qwen3-8B** (`qwen3:8b`, default): the strongest of the small models at tables and exact figures,
  and handles Indian tax terms.
- **Gemma 4** (`gemma4`): gave short, correct answers in testing, and runs fully on an Intel Arc iGPU
  (see below). It's the model set in this machine's `.env`.
- **Qwen3-4B** (`qwen3:4b`): smaller, but in this Ollama version it writes its reasoning into the answer
  even with thinking turned off.
- Any other Ollama model works through `--model`.

### Speed: use the GPU, including integrated Intel Arc

On CPU, a laptop reads prompts at only about 35 tokens/s, so each answer takes minutes. Ollama can run
models on an Intel Arc integrated GPU through Vulkan. On an Intel Core Ultra 5 125H this read prompts
4–5× faster and wrote answers 2–3× faster, with first words after about 20 s.

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_IGPU_ENABLE','1','User')
[Environment]::SetEnvironmentVariable('OLLAMA_VULKAN','1','User')
# then quit and restart the Ollama tray app; `ollama ps` should show "100% GPU"
```

Delete both variables to go back to the CPU.

Token counts: Gemma/Qwen/Llama-3 tokenizers count every digit as its own token, so bank statements
cost about 2.5× more tokens than character-based estimates suggest. `est_tokens` in
`src/finpipe/chunker.py` accounts for this, which keeps prompts within `FINPIPE_CONTEXT_MAX_TOKENS`.

## FinGPT for finance documents

When a finance model is configured, questions scoped to **Finance**, or unscoped questions whose best
match is a finance document, are answered by it. Everything else uses the default model.

```
# .env
FINPIPE_FINANCE_SLM_BACKEND=hf
FINPIPE_FINANCE_SLM_MODEL=meta-llama/Llama-2-7b-chat-hf     # gated: accept the licence on Hugging Face
FINPIPE_FINANCE_SLM_ADAPTER=FinGPT/fingpt-mt_llama2-7b_lora
```

Or serve a merged/GGUF FinGPT model through Ollama or vLLM (`FINPIPE_FINANCE_SLM_BACKEND=ollama|openai`
with `FINPIPE_FINANCE_SLM_MODEL=<name>`).

Before you rely on it, note:
- FinGPT's released adapters were trained for financial sentiment, headline and NER tasks on
  US/English data, not for document QA. They know nothing about AIS, Form 16 or Indian bank
  statements. Compare it against the default model on your own documents.
- Llama-2 bases have a 4k-token window, so the finance route's context is capped at
  `FINPIPE_FINANCE_CONTEXT_MAX_TOKENS=2500`.
- The `hf` backend loads a 7B model in-process: roughly 14 GB of RAM in bf16 on CPU, and slow
  without a GPU.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
