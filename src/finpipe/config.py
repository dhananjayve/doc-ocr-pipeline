"""Central configuration. Every field can be overridden with a FINPIPE_<FIELD> env var or a .env file."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINPIPE_", env_file=".env", extra="ignore")

    # --- storage -----------------------------------------------------------------
    data_dir: Path = Path(".finpipe")
    collection: str = "financial_docs"

    # --- parallelism ---------------------------------------------------------------
    workers: int = 0                 # parser processes; 0 = auto (each holds its own Docling models, ~1.5-2 GB RAM)
    threads_per_worker: int = 0      # torch/onnx threads per worker; 0 = cpu_count // workers
    max_pages_per_shard: int = 24    # large PDFs are split into page shards parsed in parallel

    # --- page classification (digital vs scanned) --------------------------------
    min_text_chars: int = 40         # fewer extractable chars than this => page treated as scanned
    garbage_ratio: float = 0.30      # share of unreadable glyphs above which the text layer is distrusted
    hybrid_image_coverage: float = 0.35  # digital page whose images cover more than this also gets region OCR

    # --- Docling -------------------------------------------------------------------
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    ocr_engine: Literal["rapidocr", "easyocr", "tesseract"] = "rapidocr"
    ocr_lang: list[str] = []
    # "auto": TableFormer FAST on digital/hybrid pages (text comes from the PDF, so structure is the only
    # job; ~2x faster, same figures on bank statements) and ACCURATE on scanned pages.
    table_mode: Literal["auto", "accurate", "fast"] = "auto"
    pdf_backend: Literal["dlparse", "pypdfium"] = "dlparse"
    # Digital pages: "fast" rebuilds ruled tables from PDF geometry (pdfplumber, ~0.7 s/page) and sends
    # only pages it can't read confidently to Docling; "docling" runs every page through Docling.
    digital_engine: Literal["fast", "docling"] = "fast"
    document_timeout: Optional[float] = None

    # --- chunking ------------------------------------------------------------------
    chunk_max_tokens: int = 700
    chunk_min_tokens: int = 150
    table_max_tokens: int = 1000     # tables may exceed chunk_max_tokens up to this before being row-split

    # --- embeddings / retrieval ----------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: Optional[str] = None
    embed_batch_size: int = 64
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    top_k: int = 5
    candidate_multiplier: int = 4    # vector candidates fetched = top_k * this, then hybrid re-ranked
    hybrid_alpha: float = 0.6        # weight of vector similarity vs BM25 keyword score
    reranker_model: Optional[str] = None  # e.g. "BAAI/bge-reranker-base" for a cross-encoder pass
    # Prompt reading dominates answer latency on CPU (~30-70 tokens/s for 4-8B models), so keep the
    # context tight; raise this on a GPU.
    context_max_tokens: int = 3000

    # --- small language model --------------------------------------------------------
    # "openai" = any OpenAI-compatible server (vLLM, llama.cpp, LM Studio); "hf" = local transformers (+ LoRA adapter)
    slm_backend: Literal["ollama", "openai", "hf"] = "ollama"
    slm_base_url: str = "http://localhost:11434"
    slm_model: str = "qwen3:8b"
    slm_adapter: Optional[str] = None  # PEFT/LoRA adapter for the "hf" backend
    slm_api_key: Optional[str] = None
    slm_temperature: float = 0.1
    slm_num_ctx: int = 8192
    slm_max_tokens: int = 800   # answers are asked to be compact; writing is the slowest step on CPU/iGPU
    slm_think: Optional[bool] = False  # Ollama "think" flag; None = don't send it
    slm_keep_alive: str = "4h"  # how long Ollama keeps the model loaded after a request ("-1" = forever)
    slm_timeout: float = 900.0  # seconds to wait for model output; CPU prompt reading can take minutes

    # --- finance route (e.g. FinGPT) ---------------------------------------------------
    # Questions over documents tagged "finance" use this model when finance_slm_model is set;
    # unset fields fall back to the default slm_* values. FinGPT example (hf backend):
    #   FINPIPE_FINANCE_SLM_BACKEND=hf
    #   FINPIPE_FINANCE_SLM_MODEL=meta-llama/Llama-2-7b-chat-hf
    #   FINPIPE_FINANCE_SLM_ADAPTER=FinGPT/fingpt-mt_llama2-7b_lora
    finance_slm_backend: Optional[Literal["ollama", "openai", "hf"]] = None
    finance_slm_model: Optional[str] = None
    finance_slm_adapter: Optional[str] = None
    finance_slm_base_url: Optional[str] = None
    finance_context_max_tokens: int = 2500  # Llama-2 based FinGPT models have a 4k window

    # ------------------------------------------------------------------------------
    def finance_settings(self) -> "Settings | None":
        """Settings for the finance model route, or None when no finance model is configured."""
        if not self.finance_slm_model:
            return None
        update = {
            "slm_model": self.finance_slm_model,
            "slm_adapter": self.finance_slm_adapter,
            "context_max_tokens": self.finance_context_max_tokens,
        }
        if self.finance_slm_backend:
            update["slm_backend"] = self.finance_slm_backend
        if self.finance_slm_base_url:
            update["slm_base_url"] = self.finance_slm_base_url
        return self.model_copy(update=update)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def resolved_workers(self) -> int:
        if self.workers > 0:
            return self.workers
        return max(1, min(4, (os.cpu_count() or 2) // 4))

    def resolved_threads(self) -> int:
        if self.threads_per_worker > 0:
            return self.threads_per_worker
        return max(1, (os.cpu_count() or 2) // self.resolved_workers())

    def parser_fingerprint(self) -> str:
        """Hash of everything that changes parser output; part of the parse-cache key."""
        try:
            from importlib.metadata import version

            docling_version = version("docling")
        except Exception:
            docling_version = "unknown"
        keys = ("ocr_engine", "ocr_lang", "table_mode", "pdf_backend", "digital_engine", "min_text_chars",
                "garbage_ratio", "hybrid_image_coverage")
        # bump parser_rev whenever parser.py changes what it produces, to invalidate old caches
        payload = {k: getattr(self, k) for k in keys} | {"docling": docling_version, "parser_rev": 5}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
