from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class SharedToolGNN(nn.Module):
    def __init__(
        self,
        n_buckets: int,
        tool_feat_dim: int,
        bucket_emb_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bucket_embed = nn.Embedding(n_buckets, bucket_emb_dim)
        input_dim = tool_feat_dim + bucket_emb_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data: Data, bucket_id: torch.Tensor) -> torch.Tensor:
        node_x = data.x.view(data.x.size(0), -1)
        b_emb = self.bucket_embed(bucket_id.view(-1)[0])
        b_emb = b_emb.unsqueeze(0).expand(node_x.size(0), -1)
        x = torch.cat([node_x, b_emb], dim=-1)
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            n = node_x.size(0)
            edge_index = torch.stack(
                [torch.arange(n, device=x.device), torch.arange(n, device=x.device)]
            )
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        scores = self.scorer(x).squeeze(-1)
        return torch.sigmoid(scores)
