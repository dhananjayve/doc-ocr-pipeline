import hashlib
import re

import numpy as np
import pytest

from finpipe.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", workers=1)


class HashEmbedder:
    """Deterministic bag-of-words embedder so store/retrieval tests need no model download."""

    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def embedder():
    return HashEmbedder()
