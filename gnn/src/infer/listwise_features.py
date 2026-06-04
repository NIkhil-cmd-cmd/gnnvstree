from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from gnn.src.infer.api_keys import api_key_from_hf
from gnn.src.models.listwise_rank_gnn import knn_edge_index
from gnn.src.utils.lexical import (
    api_document,
    api_document_from_key,
    bm25_scores,
    lexical_overlap,
    normalize_scores,
)


def key_to_api_dict(key: str) -> dict:
    parts = key.split("::")
    cat = parts[0] if len(parts) > 0 else ""
    tool = parts[1] if len(parts) > 1 else ""
    api = parts[2] if len(parts) > 2 else ""
    return {
        "category_name": cat,
        "tool_name": tool,
        "api_name": api,
        "api_description": api,
        "description": f"{tool} {api}",
    }


def build_listwise_data(
    prompt: str,
    candidates: list[str] | list[dict],
    embedder,
    *,
    labels: list[int] | None = None,
    knn_k: int = 4,
    from_keys: bool = True,
) -> tuple[Data, torch.Tensor | None]:
    if from_keys:
        keys = [str(k) for k in candidates]
        api_list = [key_to_api_dict(k) for k in keys]
    else:
        api_list = list(candidates)  # type: ignore[arg-type]
        keys = [api_key_from_hf(a) for a in api_list]

    bm25 = normalize_scores(bm25_scores(prompt, api_list))
    overlaps = np.array([lexical_overlap(prompt, a) for a in api_list], dtype=np.float32)

    q_emb = embedder.encode_one(prompt)
    texts = [api_document_from_key(k) if from_keys else api_document(a) for k, a in zip(keys, api_list)]
    t_emb = embedder.encode(texts)

    n = len(keys)
    q_tile = np.tile(q_emb, (n, 1))
    extra = np.stack([bm25, overlaps], axis=1)
    x = np.hstack([t_emb, q_tile, extra]).astype(np.float32)

    xt = torch.tensor(x, dtype=torch.float32)
    edge_index = knn_edge_index(xt, k=knn_k)
    data = Data(x=xt, edge_index=edge_index)
    data.tool_keys = keys

    y = None
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float32)
    return data, y
