from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def knn_edge_index(node_x: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Undirected kNN graph on tool embeddings (first half of node features)."""
    n = node_x.size(0)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=node_x.device)
    dim = node_x.size(1) // 2
    t = F.normalize(node_x[:, :dim], p=2, dim=1)
    sim = t @ t.t()
    sim.fill_diagonal_(-1.0)
    k = min(k, n - 1)
    _, nn_idx = sim.topk(k, dim=1)
    src, dst = [], []
    for i in range(n):
        for j in nn_idx[i].tolist():
            src.extend([i, j])
            dst.extend([j, i])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long, device=node_x.device)
    return torch.tensor([src, dst], dtype=torch.long, device=node_x.device)


class ListwiseRankGNN(nn.Module):
    """Query-conditioned listwise ranker (same task as ToolBench api_list re-ranking)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            edge_index = knn_edge_index(x, k=4)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.scorer(x).squeeze(-1)
