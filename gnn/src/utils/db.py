from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data


SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bucket_id INTEGER NOT NULL,
    tool_key TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    api_name TEXT,
    description TEXT,
    features BLOB NOT NULL,
    UNIQUE(bucket_id, tool_key)
);

CREATE TABLE IF NOT EXISTS tool_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bucket_id INTEGER NOT NULL,
    source_idx INTEGER NOT NULL,
    target_idx INTEGER NOT NULL,
    weight INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS training_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT UNIQUE,
    prompt TEXT NOT NULL,
    split_group TEXT NOT NULL,
    bucket_id INTEGER NOT NULL,
    bucket_key TEXT,
    tool_keys TEXT NOT NULL,
    label_indices TEXT NOT NULL,
    path_order TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    prompt_text TEXT NOT NULL,
    embedding BLOB,
    bucket_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bucket_meta (
    bucket_id INTEGER PRIMARY KEY,
    bucket_key TEXT UNIQUE NOT NULL,
    split_group TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ranking_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT UNIQUE NOT NULL,
    prompt TEXT NOT NULL,
    task_fingerprint TEXT NOT NULL,
    bucket_id INTEGER NOT NULL,
    candidate_keys TEXT NOT NULL,
    label_mask TEXT NOT NULL,
    path_order TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ranking_fingerprint ON ranking_samples(task_fingerprint);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def features_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def blob_to_features(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy().reshape(-1, dim)


def load_graph(bucket_id: int, conn: sqlite3.Connection, feat_dim: int) -> Data | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, tool_key, tool_name, features FROM tool_nodes "
        "WHERE bucket_id = ? ORDER BY id",
        (bucket_id,),
    )
    nodes = cur.fetchall()
    if not nodes:
        return None
    node_ids = [n[0] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    tool_keys = [n[1] for n in nodes]
    tool_names = [n[2] for n in nodes]
    feats = []
    for n in nodes:
        f = blob_to_features(n[3], feat_dim)
        feats.append(f.reshape(-1)[:feat_dim])
    x = torch.tensor(np.stack(feats), dtype=torch.float)
    cur.execute(
        "SELECT source_idx, target_idx FROM tool_edges WHERE bucket_id = ?",
        (bucket_id,),
    )
    edges = cur.fetchall()
    if edges:
        edge_index = torch.tensor(
            [[id_to_idx[s], id_to_idx[t]] for s, t in edges if s in id_to_idx and t in id_to_idx],
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.tool_keys = tool_keys
    data.tool_names = tool_names
    data.bucket_id = bucket_id
    return data


def save_json_field(obj: Any) -> str:
    return json.dumps(obj)
