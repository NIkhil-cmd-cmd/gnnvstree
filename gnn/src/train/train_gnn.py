from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnn.src.models.separate_tool_gnn import SeparateToolGNN
from gnn.src.models.shared_tool_gnn import SharedToolGNN
from gnn.src.utils.config import ensure_dirs, load_config, repo_root
from gnn.src.utils.db import connect, load_graph


def labels_for_sample(sample_row: tuple, n_nodes: int) -> torch.Tensor:
    import json as _json

    label_indices = _json.loads(sample_row[6])
    y = torch.zeros(n_nodes)
    for idx in label_indices:
        if 0 <= idx < n_nodes:
            y[idx] = 1.0
    return y


def load_samples(conn, limit: int | None = None) -> list[tuple]:
    cur = conn.cursor()
    q = "SELECT id, prompt, split_group, bucket_id, bucket_key, tool_keys, label_indices, path_order FROM training_samples"
    rows = cur.execute(q).fetchall()
    if limit:
        random.shuffle(rows)
        rows = rows[:limit]
    return rows


def train_epoch(model, optimizer, loss_fn, batches, conn, feat_dim: int, device: str, is_shared: bool) -> float:
    model.train()
    total = 0.0
    n = 0
    for row in batches:
        bid = row[3]
        data = load_graph(bid, conn, feat_dim)
        if data is None or data.x.size(0) == 0:
            continue
        data = data.to(device)
        y = labels_for_sample(row, data.x.size(0)).to(device)
        bucket_t = torch.tensor([bid], dtype=torch.long, device=device)
        if is_shared:
            scores = model(data, bucket_t)
        else:
            scores = model(data, int(bid))
        loss = loss_fn(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loss_fn, batches, conn, feat_dim: int, device: str, is_shared: bool) -> float:
    model.eval()
    total = 0.0
    n = 0
    for row in batches:
        bid = row[3]
        data = load_graph(bid, conn, feat_dim)
        if data is None or data.x.size(0) == 0:
            continue
        data = data.to(device)
        y = labels_for_sample(row, data.x.size(0)).to(device)
        bucket_t = torch.tensor([bid], dtype=torch.long, device=device)
        if is_shared:
            scores = model(data, bucket_t)
        else:
            scores = model(data, int(bid))
        total += loss_fn(scores, y).item()
        n += 1
    return total / max(n, 1)


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
    rows = load_samples(conn)
    random.shuffle(rows)
    split = int(len(rows) * (1 - cfg["val_ratio"]))
    train_rows, val_rows = rows[:split], rows[split:]

    cur = conn.cursor()
    cur.execute("SELECT MAX(bucket_id) FROM bucket_meta")
    max_bid = cur.fetchone()[0] or 0
    n_buckets = int(max_bid) + 1

    model = SharedToolGNN(
        n_buckets=n_buckets,
        tool_feat_dim=cfg["tool_feat_dim"],
        bucket_emb_dim=cfg["bucket_emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCELoss()

    best_val = float("inf")
    patience = 0
    art = Path(cfg["artifacts_dir"])

    for epoch in range(cfg["max_epochs"]):
        tr_loss = train_epoch(model, optimizer, loss_fn, train_rows, conn, cfg["tool_feat_dim"], device, True)
        va_loss = eval_epoch(model, loss_fn, val_rows, conn, cfg["tool_feat_dim"], device, True)
        print(f"epoch {epoch + 1} train_bce={tr_loss:.4f} val_bce={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            patience = 0
            torch.save(
                {
                    "model_type": "shared",
                    "state_dict": model.state_dict(),
                    "n_buckets": n_buckets,
                    "config": {k: cfg[k] for k in ("tool_feat_dim", "bucket_emb_dim", "hidden_dim", "dropout")},
                },
                art / "model.pt",
            )
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print("Early stopping.")
                break

    bucket_counts: dict[int, int] = {}
    for r in train_rows:
        bucket_counts[r[3]] = bucket_counts.get(r[3], 0) + 1
    top_buckets = sorted(bucket_counts, key=bucket_counts.get, reverse=True)[
        : cfg["separate_gnn_top_buckets"]
    ]
    if top_buckets:
        try:
            sep = SeparateToolGNN(top_buckets, cfg["tool_feat_dim"], cfg["hidden_dim"], cfg["dropout"]).to(device)
            sep_opt = torch.optim.Adam(sep.parameters(), lr=cfg["lr"])
            for epoch in range(min(30, cfg["max_epochs"])):
                tr = [r for r in train_rows if r[3] in top_buckets]
                train_epoch(sep, sep_opt, loss_fn, tr, conn, cfg["tool_feat_dim"], device, False)
            torch.save(
                {"model_type": "separate", "state_dict": sep.state_dict(), "bucket_ids": top_buckets},
                art / "model_separate.pt",
            )
        except Exception as exc:
            print(f"WARN: separate GNN ablation skipped: {exc}")

    summary = {"best_val_bce": best_val, "n_train": len(train_rows), "n_val": len(val_rows), "n_buckets": n_buckets}
    with open(art / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    conn.close()
    print("Training complete:", summary)


if __name__ == "__main__":
    main()
