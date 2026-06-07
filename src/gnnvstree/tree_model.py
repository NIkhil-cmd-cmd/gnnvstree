"""Prefix-tree logic model for next-tool prediction."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gnnvstree.trace import Trace


@dataclass
class Prediction:
    tool: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    explanation: str = ""


@dataclass
class TreeNode:
    count: int = 0
    terminal_count: int = 0
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    next_counts: Counter[str] = field(default_factory=Counter)


class TreeLogicAdapter:
    """A deterministic prefix tree with suffix fallback for tool traces."""

    def __init__(self, max_suffix: int = 6, min_confidence: float = 0.0):
        self.max_suffix = max_suffix
        self.min_confidence = min_confidence
        self.root = TreeNode()
        self.global_next_counts: Counter[str] = Counter()
        self.tool_counts: Counter[str] = Counter()
        self.transition_counts: Counter[tuple[str, str]] = Counter()
        self.trace_count = 0
        self.training_seconds = 0.0

    def fit(self, traces: list[Trace]) -> dict[str, Any]:
        start = time.perf_counter()
        self.root = TreeNode()
        self.global_next_counts.clear()
        self.tool_counts.clear()
        self.transition_counts.clear()
        self.trace_count = 0
        for trace in traces:
            self.update(trace)
        self.training_seconds = time.perf_counter() - start
        return self.metrics()

    def update(self, trace: Trace) -> dict[str, Any]:
        names = trace.tool_names
        if len(names) < 2:
            return {"status": "skipped", "reason": "trace_too_short"}

        self.trace_count += 1
        self.tool_counts.update(names)
        for idx in range(len(names) - 1):
            self.transition_counts[(names[idx], names[idx + 1])] += 1
            self.global_next_counts[names[idx + 1]] += 1

        for start_idx in range(len(names)):
            node = self.root
            node.count += 1
            suffix = names[start_idx : min(len(names), start_idx + self.max_suffix)]
            for offset, tool in enumerate(suffix):
                node = node.children.setdefault(tool, TreeNode())
                node.count += 1
                next_index = start_idx + offset + 1
                if next_index < len(names):
                    node.next_counts[names[next_index]] += 1
            node.terminal_count += 1
        return {"status": "updated", "trace_length": len(names)}

    def predict_next(self, prefix: list[str] | tuple[str, ...], context: dict[str, Any] | None = None) -> Prediction | None:
        if not self.trace_count:
            return None

        normalized_prefix = tuple(prefix)
        node = self._best_suffix_node(normalized_prefix)
        counts = node.next_counts if node and node.next_counts else self.global_next_counts
        if not counts:
            return None

        total = sum(counts.values())
        ranked = [(tool, count / total) for tool, count in counts.most_common()]
        top_tool, confidence = ranked[0]
        if confidence < self.min_confidence:
            return None
        suffix_length = self._matched_suffix_length(normalized_prefix)
        return Prediction(
            tool=top_tool,
            confidence=confidence,
            alternatives=ranked[1:5],
            explanation=f"matched suffix length {suffix_length}; {counts[top_tool]} of {total} continuations",
        )

    def explain_prediction(self, prefix: list[str] | tuple[str, ...], context: dict[str, Any] | None = None) -> str:
        prediction = self.predict_next(prefix, context)
        return prediction.explanation if prediction else "no prediction"

    def metrics(self) -> dict[str, Any]:
        depths: list[int] = []
        branch_factors: list[int] = []
        leaf_count = 0

        def visit(node: TreeNode, depth: int) -> None:
            nonlocal leaf_count
            depths.append(depth)
            branch_factors.append(len(node.children))
            if not node.children:
                leaf_count += 1
            for child in node.children.values():
                visit(child, depth + 1)

        visit(self.root, 0)
        node_count = len(depths)
        return {
            "framework": "tree",
            "trace_count": self.trace_count,
            "unique_tools": len(self.tool_counts),
            "transition_count": sum(self.transition_counts.values()),
            "unique_transitions": len(self.transition_counts),
            "node_count": node_count,
            "leaf_count": leaf_count,
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / node_count if node_count else 0.0,
            "avg_branch_factor": sum(branch_factors) / node_count if node_count else 0.0,
            "max_branch_factor": max(branch_factors) if branch_factors else 0,
            "training_seconds": self.training_seconds,
            "top_tools": self.tool_counts.most_common(20),
            "top_transitions": [
                {"source": src, "target": dst, "count": count}
                for (src, dst), count in self.transition_counts.most_common(20)
            ],
            "model_json_bytes": len(json.dumps(self.to_dict())),
        }

    def to_dict(self) -> dict[str, Any]:
        def encode(node: TreeNode) -> dict[str, Any]:
            return {
                "count": node.count,
                "terminal_count": node.terminal_count,
                "next_counts": dict(node.next_counts),
                "children": {name: encode(child) for name, child in node.children.items()},
            }

        return {
            "max_suffix": self.max_suffix,
            "min_confidence": self.min_confidence,
            "trace_count": self.trace_count,
            "tree": encode(self.root),
        }

    def _best_suffix_node(self, prefix: tuple[str, ...]) -> TreeNode | None:
        max_length = min(len(prefix), self.max_suffix)
        for length in range(max_length, 0, -1):
            node = self.root
            matched = True
            for tool in prefix[-length:]:
                child = node.children.get(tool)
                if child is None:
                    matched = False
                    break
                node = child
            if matched and node.next_counts:
                return node
        return None

    def _matched_suffix_length(self, prefix: tuple[str, ...]) -> int:
        max_length = min(len(prefix), self.max_suffix)
        for length in range(max_length, 0, -1):
            node = self.root
            matched = True
            for tool in prefix[-length:]:
                child = node.children.get(tool)
                if child is None:
                    matched = False
                    break
                node = child
            if matched and node.next_counts:
                return length
        return 0

