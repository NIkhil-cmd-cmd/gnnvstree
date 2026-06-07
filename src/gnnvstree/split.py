"""Deterministic trace splitting."""

from __future__ import annotations

import hashlib

from gnnvstree.trace import Trace


def split_traces(
    traces: list[Trace],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 19,
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    train: list[Trace] = []
    val: list[Trace] = []
    test: list[Trace] = []
    for trace in traces:
        bucket = _bucket(trace.task_id, seed)
        if bucket < train_ratio:
            train.append(trace)
        elif bucket < train_ratio + val_ratio:
            val.append(trace)
        else:
            test.append(trace)

    if not test and len(traces) > 1:
        test.append(train.pop())
    if not train and test:
        train.append(test.pop(0))
    return train, val, test


def _bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64

