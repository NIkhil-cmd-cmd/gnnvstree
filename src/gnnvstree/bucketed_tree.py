"""Bucketed start-state action trie for successful tool/action traces."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gnnvstree.buckets import BucketAssignment, BucketRouter
from gnnvstree.trace import Trace
from gnnvstree.tree_model import Prediction


@dataclass
class ActionTrieNode:
    action: str = ""
    count: int = 0
    terminal_count: int = 0
    children: dict[str, "ActionTrieNode"] = field(default_factory=dict)
    pruned: bool = False
    prune_reasons: list[str] = field(default_factory=list)

    def score_children(self) -> list[tuple[str, float]]:
        candidates = [(name, child.count) for name, child in self.children.items() if not child.pruned]
        total = sum(count for _, count in candidates)
        if total <= 0:
            return []
        return sorted(((name, count / total) for name, count in candidates), key=lambda item: item[1], reverse=True)


class BucketedActionTreeAdapter:
    """Task-bucketed start-state action trie.

    Unlike ``TreeLogicAdapter``, this model inserts only complete traces from
    START. It does not insert every suffix at the root. That keeps the learned
    tree aligned with workflow branches inside each task bucket.
    """

    def __init__(
        self,
        n_buckets: int | None = None,
        min_bucket_traces: int = 25,
        min_confidence: float = 0.0,
        random_state: int = 42,
    ):
        self.router = BucketRouter(
            n_buckets=n_buckets,
            random_state=random_state,
            min_bucket_traces=min_bucket_traces,
        )
        self.min_confidence = min_confidence
        self.bucket_roots: dict[int, ActionTrieNode] = {}
        self.bucket_assignments: dict[str, BucketAssignment] = {}
        self.trace_count = 0
        self.successful_trace_count = 0
        self.action_counts: Counter[str] = Counter()
        self.transition_counts: Counter[tuple[int, str, str]] = Counter()
        self.training_seconds = 0.0

    def fit(self, traces: list[Trace]) -> dict[str, Any]:
        start = time.perf_counter()
        fit_traces = [trace for trace in traces if trace.success and len(trace.tool_names) >= 1]
        if not fit_traces:
            raise ValueError("BucketedActionTreeAdapter.fit requires at least one successful trace")

        self.router.fit(fit_traces)
        self.bucket_roots = {bucket_id: ActionTrieNode(action=f"bucket_{bucket_id}") for bucket_id in self.router.bucket_sizes}
        self.bucket_assignments.clear()
        self.trace_count = 0
        self.successful_trace_count = 0
        self.action_counts.clear()
        self.transition_counts.clear()

        for trace in fit_traces:
            self.update(trace)
        self.training_seconds = time.perf_counter() - start
        return self.metrics()

    def update(self, trace: Trace) -> dict[str, Any]:
        if not trace.success or not trace.tool_names:
            return {"status": "skipped", "reason": "unsuccessful_or_empty_trace"}

        assignment = self.router.route(trace.task_text or trace.task_id)
        root = self.bucket_roots.setdefault(assignment.bucket_id, ActionTrieNode(action=f"bucket_{assignment.bucket_id}"))
        self.bucket_assignments[trace.trace_id] = assignment
        self.trace_count += 1
        self.successful_trace_count += 1
        self.action_counts.update(trace.tool_names)

        root.count += 1
        node = root
        previous: str | None = None
        for action in trace.tool_names:
            child = node.children.setdefault(action, ActionTrieNode(action=action))
            child.count += 1
            if previous is not None:
                self.transition_counts[(assignment.bucket_id, previous, action)] += 1
            previous = action
            node = child
        node.terminal_count += 1
        return {"status": "updated", "bucket_id": assignment.bucket_id, "trace_length": len(trace.tool_names)}

    def recommend_next(
        self,
        task_text: str,
        prefix: list[str] | tuple[str, ...],
        candidate_actions: set[str] | None = None,
    ) -> Prediction | None:
        if not self.trace_count:
            return None

        assignment = self.router.route(task_text)
        node = self._node_for_prefix(assignment.bucket_id, prefix)
        if node is None:
            return None

        ranked = node.score_children()
        if candidate_actions is not None:
            ranked = [(action, score) for action, score in ranked if action in candidate_actions]
        if not ranked:
            return None

        top_action, confidence = ranked[0]
        if confidence < self.min_confidence:
            return None
        return Prediction(
            tool=top_action,
            confidence=confidence,
            alternatives=ranked[1:5],
            explanation=f"bucket {assignment.bucket_id}; routed confidence {assignment.confidence:.3f}",
        )

    def predict_next(self, prefix: list[str] | tuple[str, ...], context: dict[str, Any] | None = None) -> Prediction | None:
        context = context or {}
        task_text = str(context.get("task_text") or "")
        if not task_text:
            return None
        candidates = context.get("candidate_actions")
        candidate_set = set(candidates) if candidates is not None else None
        return self.recommend_next(task_text, prefix, candidate_set)

    def metrics(self) -> dict[str, Any]:
        depths: list[int] = []
        branch_factors: list[int] = []
        leaf_count = 0

        def visit(node: ActionTrieNode, depth: int) -> None:
            nonlocal leaf_count
            depths.append(depth)
            branch_factors.append(len(node.children))
            if not node.children:
                leaf_count += 1
            for child in node.children.values():
                visit(child, depth + 1)

        for root in self.bucket_roots.values():
            visit(root, 0)

        node_count = len(depths)
        router_metrics = self.router.metrics()
        return {
            "framework": "bucketed_action_tree",
            "trace_count": self.trace_count,
            "successful_trace_count": self.successful_trace_count,
            "bucket_count": router_metrics["bucket_count"],
            "bucket_sizes": router_metrics["bucket_sizes"],
            "supported_bucket_count": router_metrics["supported_bucket_count"],
            "unique_actions": len(self.action_counts),
            "transition_count": sum(self.transition_counts.values()),
            "unique_transitions": len(self.transition_counts),
            "node_count": node_count,
            "leaf_count": leaf_count,
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / node_count if node_count else 0.0,
            "avg_branch_factor": sum(branch_factors) / node_count if node_count else 0.0,
            "max_branch_factor": max(branch_factors) if branch_factors else 0,
            "training_seconds": self.training_seconds,
            "top_actions": self.action_counts.most_common(20),
            "router": router_metrics,
            "model_json_bytes": len(json.dumps(self.to_dict())),
        }

    def to_dict(self) -> dict[str, Any]:
        def encode(node: ActionTrieNode) -> dict[str, Any]:
            return {
                "action": node.action,
                "count": node.count,
                "terminal_count": node.terminal_count,
                "pruned": node.pruned,
                "prune_reasons": node.prune_reasons,
                "children": {name: encode(child) for name, child in node.children.items()},
            }

        return {
            "min_confidence": self.min_confidence,
            "trace_count": self.trace_count,
            "router": self.router.to_dict(),
            "buckets": {str(bucket_id): encode(root) for bucket_id, root in sorted(self.bucket_roots.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BucketedActionTreeAdapter":
        model = cls(min_confidence=float(data.get("min_confidence", 0.0)))
        model.trace_count = int(data.get("trace_count", 0))

        # The persisted router currently stores metrics only. Recreate enough of
        # the object for inspection; runtime routing for loaded artifacts is
        # handled by script-side nearest-centroid substitutes until the vectorizer
        # is persisted.
        model.router.bucket_sizes = {
            int(bucket_id): int(size)
            for bucket_id, size in data.get("router", {}).get("metrics", {}).get("bucket_sizes", {}).items()
        }

        def decode(node_data: dict[str, Any]) -> ActionTrieNode:
            node = ActionTrieNode(
                action=str(node_data.get("action", "")),
                count=int(node_data.get("count", 0)),
                terminal_count=int(node_data.get("terminal_count", 0)),
                pruned=bool(node_data.get("pruned", False)),
                prune_reasons=[str(reason) for reason in node_data.get("prune_reasons", [])],
            )
            node.children = {
                str(name): decode(child)
                for name, child in node_data.get("children", {}).items()
            }
            return node

        model.bucket_roots = {
            int(bucket_id): decode(root)
            for bucket_id, root in data.get("buckets", {}).items()
        }
        return model

    def _node_for_prefix(self, bucket_id: int, prefix: list[str] | tuple[str, ...]) -> ActionTrieNode | None:
        node = self.bucket_roots.get(bucket_id)
        if node is None:
            return None
        for action in prefix:
            node = node.children.get(action)
            if node is None or node.pruned:
                return None
        return node
