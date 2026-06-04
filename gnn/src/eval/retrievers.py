from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from rank_bm25 import BM25Okapi

from gnn.src.infer.retrieve_tools import GNNRetriever, api_key_from_hf  # noqa: F401


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, api_list: list[dict], path: list[str] | None, top_k: int) -> tuple[list[str], dict[str, float]]:
        ...


class GNNRetrieverWrapper(BaseRetriever):
    name = "shared_gnn"

    def __init__(self, cfg: dict):
        self.retriever = GNNRetriever(cfg)

    def retrieve(self, query: str, api_list: list[dict], path: list[str] | None, top_k: int):
        r = self.retriever.retrieve_from_api_list(query, api_list, top_k=top_k)
        return r.tool_keys, r.timings_ms


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def retrieve(self, query: str, api_list: list[dict], path: list[str] | None, top_k: int):
        t0 = time.perf_counter()
        corpus = []
        keys = []
        for api in api_list:
            keys.append(api_key_from_hf(api))
            text = " ".join(
                str(api.get(k, ""))
                for k in ("category_name", "tool_name", "api_name", "api_description", "description")
            )
            corpus.append(re.findall(r"\w+", text.lower()))
        bm25 = BM25Okapi(corpus)
        q = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        ms = (time.perf_counter() - t0) * 1000
        return [keys[i] for i in order], {"total_retrieval_ms": ms}


class FullListRetriever(BaseRetriever):
    name = "full_list"

    def retrieve(self, query: str, api_list: list[dict], path: list[str] | None, top_k: int):
        keys = [api_key_from_hf(a) for a in api_list[:top_k]]
        return keys, {"total_retrieval_ms": 0.0}


class ToolBenchIRRetriever(BaseRetriever):
    name = "toolbench_ir"

    def __init__(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cpu"
        self.model.to(self.device)

    def _encode(self, texts: list[str]):
        import torch

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state[:, 0, :]
        return out

    def retrieve(self, query: str, api_list: list[dict], path: list[str] | None, top_k: int):
        t0 = time.perf_counter()
        keys = [api_key_from_hf(a) for a in api_list]
        texts = []
        for api in api_list:
            texts.append(
                " ".join(
                    str(api.get(k, ""))
                    for k in ("category_name", "tool_name", "api_name", "api_description")
                )
            )
        q_emb = self._encode([query])
        d_emb = self._encode(texts)
        scores = (q_emb @ d_emb.T).squeeze().cpu().numpy()
        order = scores.argsort()[::-1][:top_k]
        ms = (time.perf_counter() - t0) * 1000
        return [keys[i] for i in order], {"total_retrieval_ms": ms}


def load_benchmark_queries(data_dir: Path, splits: list[str]) -> list[dict]:
    queries = []
    bench_dir = data_dir / "hf_benchmark"
    for split in splits:
        p = bench_dir / f"{split}.json"
        if not p.exists():
            continue
        with open(p) as f:
            rows = json.load(f)
        for row in rows:
            relevant = row["relevant_apis"]
            rel_keys = [api_key_from_hf(a) for a in relevant]
            queries.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "api_list": row["api_list"],
                    "relevant_keys": rel_keys,
                    "path": rel_keys,
                    "split": split,
                }
            )
    return queries
