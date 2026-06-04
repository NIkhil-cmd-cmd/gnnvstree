from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class _SingleBucketGNN(nn.Module):
    def __init__(self, tool_feat_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(tool_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        node_x = data.x.view(data.x.size(0), -1)
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            n = node_x.size(0)
            edge_index = torch.stack(
                [torch.arange(n, device=node_x.device), torch.arange(n, device=node_x.device)]
            )
        x = F.relu(self.conv1(node_x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(self.scorer(x).squeeze(-1))


class SeparateToolGNN(nn.Module):
    """One small GCN per bucket (ablation vs shared backbone)."""

    def __init__(
        self,
        bucket_ids: list[int],
        tool_feat_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bucket_ids = bucket_ids
        self.models = nn.ModuleDict(
            {str(bid): _SingleBucketGNN(tool_feat_dim, hidden_dim, dropout) for bid in bucket_ids}
        )

    def forward(self, data: Data, bucket_id: int) -> torch.Tensor:
        key = str(int(bucket_id))
        if key not in self.models:
            raise KeyError(f"No separate GNN for bucket {bucket_id}")
        return self.models[key](data)
