from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnn.src.models.listwise_rank_gnn import ListwiseRankGNN
from gnn.src.infer.listwise_features import build_listwise_data
from gnn.src.infer.listwise_score import calibrate_bm25_blend
from gnn.src.train.ranking_graph import load_ranking_rows
from gnn.src.utils.config import ensure_dirs, load_config, repo_root
from gnn.src.utils.db import connect
from gnn.src.utils.embeddings import Embedder
from gnn.src.utils.task_dedup import load_benchmark_fingerprints


def ranking_hinge_loss(scores: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    pos = scores[y > 0.5]
    neg = scores[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_tensor(0.0)
    return F.relu(margin - pos.max() + neg.max())


def train_epoch(model, optimizer, rows, embedder, cfg, device) -> float:
    model.train()
    random.shuffle(rows)
    cap = cfg.get("ranking_epoch_samples")
    if cap and len(rows) > cap:
        rows = rows[: int(cap)]
    total = 0.0
    n = 0
    bce = nn.BCEWithLogitsLoss()
    knn_k = cfg.get("knn_k", 4)
    rank_w = cfg.get("ranking_loss_weight", 0.5)
    margin = cfg.get("rank_margin", 0.3)

    for row in rows:
        prompt = row[1]
        keys = json.loads(row[2])
        labels = json.loads(row[3])
        data, y = build_listwise_data(prompt, keys, embedder, labels=labels, knn_k=knn_k)
        if y is None or y.sum() == 0:
            continue
        data = data.to(device)
        y = y.to(device)
        logits = model(data)
        loss = bce(logits, y) + rank_w * ranking_hinge_loss(logits, y, margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, rows, embedder, cfg, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    p1_hits = 0
    n = 0
    bce = nn.BCEWithLogitsLoss()
    knn_k = cfg.get("knn_k", 4)
    rank_w = cfg.get("ranking_loss_weight", 0.5)
    margin = cfg.get("rank_margin", 0.3)

    for row in rows:
        prompt = row[1]
        keys = json.loads(row[2])
        labels = json.loads(row[3])
        data, y = build_listwise_data(prompt, keys, embedder, labels=labels, knn_k=knn_k)
        if y is None or y.sum() == 0:
            continue
        data = data.to(device)
        y = y.to(device)
        logits = model(data)
        loss = bce(logits, y) + rank_w * ranking_hinge_loss(logits, y, margin)
        total_loss += loss.item()
        order = logits.argsort(descending=True)
        if labels[order[0].item()] > 0:
            p1_hits += 1
        n += 1
    return total_loss / max(n, 1), p1_hits / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gnn/configs/cpu_g2_g3.yaml")
    args = parser.parse_args()
    cfg = load_config(repo_root() / args.config)
    ensure_dirs(cfg)
    random.seed(42)
    torch.manual_seed(42)

    device = "cpu"
    conn = connect(cfg["sqlite_path"])
    rows = load_ranking_rows(conn)
    if not rows:
        raise RuntimeError("No ranking_samples. Run: python -m gnn.src.data.build_ranking_dataset")

    exclude = load_benchmark_fingerprints(Path(cfg["data_dir"]))
    rows = [r for r in rows if r[4] not in exclude]
    random.shuffle(rows)
    split = int(len(rows) * (1 - cfg["val_ratio"]))
    train_rows, val_rows = rows[:split], rows[split:]

    embedder = Embedder(cfg["embedder_model"], proj_dim=cfg["tool_feat_dim"])
    # tool_emb + query_emb + bm25 + overlap
    input_dim = cfg["tool_feat_dim"] * 2 + 2
    model = ListwiseRankGNN(
        input_dim=input_dim,
        hidden_dim=cfg.get("listwise_hidden_dim", 64),
        dropout=cfg.get("dropout", 0.15),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best_p1 = -1.0
    patience = 0
    art = Path(cfg["artifacts_dir"])

    for epoch in range(cfg["max_epochs"]):
        tr_loss = train_epoch(model, optimizer, train_rows, embedder, cfg, device)
        va_loss, va_p1 = eval_epoch(model, val_rows, embedder, cfg, device)
        print(f"epoch {epoch + 1} train={tr_loss:.4f} val_loss={va_loss:.4f} val_P@1={va_p1:.4f}")
        if va_p1 > best_p1:
            best_p1 = va_p1
            patience = 0
            torch.save(
                {
                    "model_type": "listwise",
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "config": {
                        "tool_feat_dim": cfg["tool_feat_dim"],
                        "listwise_hidden_dim": cfg.get("listwise_hidden_dim", 64),
                        "dropout": cfg.get("dropout", 0.15),
                        "knn_k": cfg.get("knn_k", 4),
                    },
                },
                art / "model.pt",
            )
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print("Early stopping.")
                break

    blend = float(cfg.get("bm25_blend", 0.25))
    ckpt_path = art / "model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        blend = calibrate_bm25_blend(model, val_rows, embedder, cfg, device)
        ckpt["bm25_blend"] = blend
        torch.save(ckpt, ckpt_path)

    summary = {
        "model_type": "listwise",
        "best_val_p1": best_p1,
        "bm25_blend": blend,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_ranking_samples": len(rows),
    }
    with open(art / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    conn.close()
    print("Listwise training complete:", summary)


if __name__ == "__main__":
    main()
