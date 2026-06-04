from __future__ import annotations

import numpy as np
import torch

from gnn.src.utils.lexical import bm25_scores, normalize_scores


def listwise_scores(
    logits: torch.Tensor,
    prompt: str,
    api_list: list[dict],
    bm25_blend: float,
) -> np.ndarray:
    bm25 = normalize_scores(bm25_scores(prompt, api_list))
    neural = torch.sigmoid(logits).cpu().numpy()
    blend = float(bm25_blend)
    return (1.0 - blend) * neural + blend * bm25


def calibrate_bm25_blend(
    model,
    val_rows,
    embedder,
    cfg,
    device: str,
) -> float:
    """Pick blend weight on held-out ranking val only (not benchmark)."""
    import json

    from gnn.src.infer.listwise_features import build_listwise_data

    keys = [k for k in val_rows]
    if not keys:
        return float(cfg.get("bm25_blend", 0.25))

    best_blend = float(cfg.get("bm25_blend", 0.25))
    best_p1 = -1.0
    knn_k = cfg.get("knn_k", 4)

    for blend in (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        hits = 0
        n = 0
        for row in keys[: min(400, len(keys))]:
            prompt = row[1]
            cand_keys = json.loads(row[2])
            labels = json.loads(row[3])
            api_list = [
                {
                    "category_name": (k.split("::") + ["", "", ""])[0],
                    "tool_name": (k.split("::") + ["", "", ""])[1],
                    "api_name": (k.split("::") + ["", "", ""])[2],
                    "api_description": (k.split("::") + ["", "", ""])[2],
                    "description": k,
                }
                for k in cand_keys
            ]
            data, _ = build_listwise_data(prompt, cand_keys, embedder, labels=None, knn_k=knn_k)
            data = data.to(device)
            with torch.no_grad():
                logits = model(data)
            sc = listwise_scores(logits, prompt, api_list, blend)
            if labels[int(sc.argmax())] > 0:
                hits += 1
            n += 1
        p1 = hits / max(n, 1)
        if p1 > best_p1:
            best_p1 = p1
            best_blend = blend
    print(f"Calibrated bm25_blend={best_blend:.2f} (val P@1={best_p1:.4f})")
    return best_blend
