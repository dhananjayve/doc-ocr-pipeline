"""Small-language-model clients: Ollama native API, any OpenAI-compatible server, or local
transformers with an optional LoRA adapter (FinGPT)."""

from __future__ import annotations

import json
import threading
from typing import Iterator, Protocol

import httpx

from finpipe.config import Settings

OPEN_TAG, CLOSE_TAG = "<think>", "</think>"


def _partial_suffix(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `tag`."""
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class ThinkFilter:
    """Streaming filter that drops <think>...</think> reasoning blocks (Qwen3, DeepSeek-R1 style)."""

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while True:
            if self.in_think:
                i = self.buf.find(CLOSE_TAG)
                if i < 0:
                    self.buf = self.buf[-(len(CLOSE_TAG) - 1):]
                    return "".join(out)
                self.buf = self.buf[i + len(CLOSE_TAG):]
                self.in_think = False
            else:
                i = self.buf.find(OPEN_TAG)
                if i >= 0:
                    out.append(self.buf[:i])
                    self.buf = self.buf[i + len(OPEN_TAG):]
                    self.in_think = True
                    continue
                keep = _partial_suffix(self.buf, OPEN_TAG)
                out.append(self.buf[: len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return "".join(out)

    def flush(self) -> str:
        rest = "" if self.in_think else self.buf
        self.buf = ""
        return rest


class SLMClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        headers = {"Authorization": f"Bearer {settings.slm_api_key}"} if settings.slm_api_key else {}
        self.http = httpx.Client(
            base_url=settings.slm_base_url.rstrip("/"), timeout=settings.slm_timeout,
            headers=headers, transport=transport,
        )

    # -- public -----------------------------------------------------------------
    def chat(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages)).strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        filt = ThinkFilter()
        started = False
        raw = self._ollama(messages) if self.s.slm_backend == "ollama" else self._openai(messages)
        for piece in raw:
            text = filt.feed(piece)
            if not started:
                text = text.lstrip()
                started = bool(text)
            if text:
                yield text
        tail = filt.flush()
        if tail:
            yield tail

    def warmup(self) -> None:
        """Load the model into memory now (Ollama), so the first question doesn't pay for it."""
        if self.s.slm_backend != "ollama":
            return
        self.http.post("/api/generate", json={
            "model": self.s.slm_model, "prompt": "", "keep_alive": self.s.slm_keep_alive,
            "options": {"num_ctx": self.s.slm_num_ctx},
        })

    # -- backends ---------------------------------------------------------------
    def _ollama(self, messages: list[dict]) -> Iterator[str]:
        body = {
            "model": self.s.slm_model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.s.slm_keep_alive,
            "options": {
                "temperature": self.s.slm_temperature,
                "num_ctx": self.s.slm_num_ctx,
                "num_predict": self.s.slm_max_tokens,
            },
        }
        if self.s.slm_think is not None:
            body["think"] = self.s.slm_think
        with self.http.stream("POST", "/api/chat", json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(f"Ollama error: {data['error']}")
                content = data.get("message", {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    if data.get("done_reason") == "length":
                        yield "\n\n[Answer cut off at the length limit (FINPIPE_SLM_MAX_TOKENS). Ask a narrower question.]"
                    break

    def _openai(self, messages: list[dict]) -> Iterator[str]:
        body = {
            "model": self.s.slm_model,
            "messages": messages,
            "stream": True,
            "temperature": self.s.slm_temperature,
            "max_tokens": self.s.slm_max_tokens,
        }
        with self.http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"SLM server error {r.status_code}: {r.text}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                choices = json.loads(payload).get("choices") or [{}]
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content


class ChatModel(Protocol):
    def chat(self, messages: list[dict]) -> str: ...
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class HFClient:
    """Local transformers model, optionally with a PEFT/LoRA adapter.

    FinGPT publishes LoRA adapters (e.g. FinGPT/fingpt-mt_llama2-7b_lora) over Llama-2 bases. Models
    without a chat template get FinGPT's instruction format: "Instruction: ...\nInput: ...\nAnswer: ".
    """

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.s = settings
        self.tok = AutoTokenizer.from_pretrained(settings.slm_model)
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.slm_model,
            torch_dtype=torch.float16 if cuda else torch.bfloat16,
            device_map="auto" if cuda else None,
            low_cpu_mem_usage=True,
        )
        if settings.slm_adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("LoRA adapters need peft: pip install 'finpipe[fingpt]'") from exc
            self.model = PeftModel.from_pretrained(self.model, settings.slm_adapter)
        self.model.eval()
        self._lock = threading.Lock()  # one generation at a time per loaded model

    def _prompt(self, messages: list[dict]) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return f"Instruction: {system}\nInput: {user}\nAnswer: "

    def chat(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages)).strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self.tok(self._prompt(messages), return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs, streamer=streamer, max_new_tokens=self.s.slm_max_tokens,
            do_sample=self.s.slm_temperature > 0, temperature=max(self.s.slm_temperature, 1e-3),
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        with self._lock:
            worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            worker.start()
            filt = ThinkFilter()
            for piece in streamer:
                if text := filt.feed(piece):
                    yield text
            if tail := filt.flush():
                yield tail
            worker.join()


def make_slm(settings: Settings) -> ChatModel:
    return HFClient(settings) if settings.slm_backend == "hf" else SLMClient(settings)
