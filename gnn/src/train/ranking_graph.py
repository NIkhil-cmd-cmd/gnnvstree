from __future__ import annotations

from gnn.src.utils.db import connect


def load_ranking_rows(conn) -> list[tuple]:
    cur = conn.cursor()
    return cur.execute(
        "SELECT id, prompt, candidate_keys, label_mask, task_fingerprint FROM ranking_samples"
    ).fetchall()
