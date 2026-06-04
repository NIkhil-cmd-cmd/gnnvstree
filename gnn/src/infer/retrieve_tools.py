from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gnn.src.models.shared_tool_gnn import SharedToolGNN
from gnn.src.utils.db import connect, load_graph
from gnn.src.utils.embeddings import Embedder


@dataclass
class RetrievalResult:
    tool_keys: list[str]
    scores: list[float]
    bucket_id: int
    timings_ms: dict[str, float] = field(default_factory=dict)


class GNNRetriever:
    def __init__(self, cfg: dict, checkpoint: Path | None = None):
        self.cfg = cfg
        self.device = "cpu"
        ckpt_path = checkpoint or Path(cfg["artifacts_dir"]) / "model.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        n_buckets = ckpt["n_buckets"]
        c = ckpt["config"]
        self.model = SharedToolGNN(
            n_buckets=n_buckets,
            tool_feat_dim=c["tool_feat_dim"],
            bucket_emb_dim=c["bucket_emb_dim"],
            hidden_dim=c["hidden_dim"],
            dropout=c["dropout"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.embedder = Embedder(cfg["embedder_model"], proj_dim=cfg["tool_feat_dim"])
        self.conn = connect(cfg["sqlite_path"])
        art = Path(cfg["artifacts_dir"])
        kmeans_path = art / "kmeans.pkl"
        self.kmeans = None
        self.cluster_to_bucket: dict[int, int] = {}
        if kmeans_path.exists():
            with open(kmeans_path, "rb") as f:
                self.kmeans = pickle.load(f)
        map_path = art / "cluster_to_bucket.json"
        if map_path.exists():
            with open(map_path) as f:
                self.cluster_to_bucket = {int(k): int(v) for k, v in json.load(f).items()}

    def route_bucket(self, prompt: str, gold_bucket: int | None = None) -> tuple[int, float]:
        t0 = time.perf_counter()
        emb = self.embedder.encode_one(prompt)
        embed_ms = (time.perf_counter() - t0) * 1000
        if self.kmeans is not None:
            t1 = time.perf_counter()
            cluster = int(self.kmeans.predict(emb.reshape(1, -1))[0])
            bid = self.cluster_to_bucket.get(cluster, cluster)
            kmeans_ms = (time.perf_counter() - t1) * 1000
            return bid, embed_ms + kmeans_ms
        return gold_bucket or 0, embed_ms

    def retrieve_from_api_list(
        self,
        prompt: str,
        api_list: list[dict],
        top_k: int = 5,
        gold_bucket: int | None = None,
    ) -> RetrievalResult:
        """Score ToolBench benchmark candidates with shared GNN (per-query graph)."""
        from gnn.src.eval.retrievers import api_key_from_hf
        from torch_geometric.data import Data

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        bucket_id, route_ms = self.route_bucket(prompt, gold_bucket)
        timings["embed_kmeans_ms"] = route_ms

        keys = [api_key_from_hf(a) for a in api_list]
        texts = [
            " ".join(
                str(a.get(k, ""))
                for k in ("category_name", "tool_name", "api_name", "api_description", "description")
            )
            for a in api_list
        ]
        t1 = time.perf_counter()
        feats = self.embedder.encode(texts)
        x = torch.tensor(feats, dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.tool_keys = keys
        timings["sqlite_load_ms"] = (time.perf_counter() - t1) * 1000

        data = data.to(self.device)
        t2 = time.perf_counter()
        with torch.no_grad():
            scores = self.model(data, torch.tensor([min(bucket_id, self.model.bucket_embed.num_embeddings - 1)], device=self.device))
        timings["gnn_forward_ms"] = (time.perf_counter() - t2) * 1000
        timings["total_retrieval_ms"] = (time.perf_counter() - t0) * 1000

        sc = scores.cpu().numpy()
        order = np.argsort(-sc)[:top_k]
        return RetrievalResult(
            tool_keys=[keys[i] for i in order],
            scores=[float(sc[i]) for i in order],
            bucket_id=bucket_id,
            timings_ms=timings,
        )

    def retrieve(
        self,
        prompt: str,
        top_k: int = 5,
        gold_bucket: int | None = None,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        bucket_id, route_ms = self.route_bucket(prompt, gold_bucket)
        timings["embed_kmeans_ms"] = route_ms

        t1 = time.perf_counter()
        data = load_graph(bucket_id, self.conn, self.cfg["tool_feat_dim"])
        timings["sqlite_load_ms"] = (time.perf_counter() - t1) * 1000
        if data is None:
            return RetrievalResult([], [], bucket_id, timings)

        data = data.to(self.device)
        t2 = time.perf_counter()
        with torch.no_grad():
            scores = self.model(data, torch.tensor([bucket_id], device=self.device))
        timings["gnn_forward_ms"] = (time.perf_counter() - t2) * 1000
        timings["total_retrieval_ms"] = (time.perf_counter() - t0) * 1000

        sc = scores.cpu().numpy()
        order = np.argsort(-sc)[:top_k]
        keys = [data.tool_keys[i] for i in order]
        return RetrievalResult(
            tool_keys=keys,
            scores=[float(sc[i]) for i in order],
            bucket_id=bucket_id,
            timings_ms=timings,
        )

    def close(self) -> None:
        self.conn.close()


def api_key_from_hf(api: dict | list) -> str:
    if isinstance(api, (list, tuple)):
        if len(api) >= 3:
            return f"{api[0]}::{api[1]}::{api[2]}"
        if len(api) >= 2:
            return f"{api[0]}::{api[0]}::{api[1]}"
        return str(api[0]) if api else ""
    cat = api.get("category_name") or api.get("category") or ""
    tool = api.get("tool_name") or api.get("tool") or ""
    api_name = api.get("api_name") or api.get("api") or ""
    return f"{cat}::{tool}::{api_name}"


def api_match_key(key: str, other: str) -> bool:
    if key == other:
        return True
    a = key.split("::")
    b = other.split("::")
    if len(a) >= 3 and len(b) >= 2:
        return a[1] == b[0] or a[1] == b[1] or a[2] == b[-1]
    return key in other or other in key
