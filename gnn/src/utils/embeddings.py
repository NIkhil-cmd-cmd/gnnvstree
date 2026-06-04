from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, proj_dim: int = 64, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        raw_dim = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)()
        self.proj = nn.Linear(raw_dim, proj_dim)
        self.proj_dim = proj_dim

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.proj_dim), dtype=np.float32)
        raw = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        with torch.no_grad():
            t = torch.tensor(raw, dtype=torch.float32)
            out = self.proj(t).numpy()
        return out.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
