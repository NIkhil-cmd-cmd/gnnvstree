from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


API_TEXT_FIELDS = (
    "category_name",
    "tool_name",
    "api_name",
    "api_description",
    "description",
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def api_document(api: dict) -> str:
    return " ".join(str(api.get(k, "")) for k in API_TEXT_FIELDS)


def api_document_from_key(key: str) -> str:
    parts = key.split("::")
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1]} {parts[2]}"
    return key


def lexical_overlap(query: str, api: dict) -> float:
    q = set(tokenize(query))
    d = set(tokenize(api_document(api)))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def bm25_scores(query: str, api_list: list[dict]) -> np.ndarray:
    if not api_list:
        return np.zeros(0, dtype=np.float32)
    corpus = [tokenize(api_document(a)) for a in api_list]
    q = tokenize(query)
    scores = BM25Okapi(corpus).get_scores(q)
    return np.asarray(scores, dtype=np.float32)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mx = float(scores.max())
    if mx <= 1e-9:
        return np.zeros_like(scores)
    return scores / mx
