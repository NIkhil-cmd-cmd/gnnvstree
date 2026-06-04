from __future__ import annotations

import math
from typing import Sequence


def _as_set(items: Sequence[str]) -> set[str]:
    return {str(x) for x in items}


def _matches(item: str, rel: str) -> bool:
    if item == rel:
        return True
    from gnn.src.infer.retrieve_tools import api_match_key

    return api_match_key(item, rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    rel = list(relevant)
    return sum(1 for x in top if any(_matches(x, r) for r in rel)) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = list(relevant)
    if not rel:
        return 0.0
    top = retrieved[:k]
    hit = sum(1 for r in rel if any(_matches(x, r) for x in top))
    return hit / len(rel)


def f1_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = list(relevant)
    top = retrieved[:k]
    return 1.0 if any(any(_matches(x, r) for x in top) for r in rel) else 0.0


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = list(relevant)
    for i, item in enumerate(retrieved, start=1):
        if any(_matches(item, r) for r in rel):
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = list(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, x in enumerate(retrieved[:k])
        if any(_matches(x, r) for r in rel)
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = list(relevant)
    if not rel:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for i, x in enumerate(retrieved, start=1):
        if any(_matches(x, r) for r in rel):
            hits += 1
            prec_sum += hits / i
    return prec_sum / len(rel) if rel else 0.0


def path_recall_at_k(retrieved: Sequence[str], path: Sequence[str], k: int) -> float:
    path_list = list(path)
    if not path_list:
        return 0.0
    top = retrieved[:k]
    hit = sum(1 for p in path_list if any(_matches(x, p) for x in top))
    return hit / len(path_list)


def path_f1_at_k(retrieved: Sequence[str], path: Sequence[str], k: int) -> float:
    return f1_at_k(retrieved, path, k)


def ordered_path_match_at_k(retrieved: Sequence[str], path: Sequence[str], k: int) -> float:
    """Longest subsequence match length / len(path)."""
    if not path:
        return 0.0
    top = list(retrieved[:k])
    i = j = 0
    while i < len(top) and j < len(path):
        if _matches(top[i], path[j]):
            j += 1
        i += 1
    return j / len(path)


def workflow_coverage(retrieved: Sequence[str], path: Sequence[str], k: int) -> float:
    path_list = list(path)
    top = retrieved[:k]
    if not path_list:
        return 0.0
    return 1.0 if all(any(_matches(x, p) for x in top) for p in path_list) else 0.0


def next_tool_acc_at_1(retrieved: Sequence[str], path: Sequence[str], prefix_len: int = 0) -> float:
    if prefix_len >= len(path):
        return 0.0
    if not retrieved:
        return 0.0
    return 1.0 if _matches(retrieved[0], path[prefix_len]) else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def aggregate_timings(timings_list: list[dict[str, float]]) -> dict[str, float]:
    keys = set()
    for t in timings_list:
        keys.update(t.keys())
    out = {}
    for k in keys:
        vals = [t[k] for t in timings_list if k in t]
        if vals:
            out[f"{k}_p50"] = percentile(vals, 50)
            out[f"{k}_p95"] = percentile(vals, 95)
            out[f"{k}_p99"] = percentile(vals, 99)
            out[f"{k}_mean"] = sum(vals) / len(vals)
    return out
