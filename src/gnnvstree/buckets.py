"""Task bucketing for bucketed action trees."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gnnvstree.trace import Trace


@dataclass(frozen=True)
class BucketAssignment:
    bucket_id: int
    confidence: float
    distances: tuple[float, ...] = ()


class BucketRouter:
    """Route task text into K-means buckets.

    This first implementation uses TF-IDF features to avoid introducing a heavy
    embedding model into the core path. The class is intentionally narrow so a
    sentence-embedding provider can replace the vectorizer later without
    changing the tree adapter.
    """

    def __init__(
        self,
        n_buckets: int | None = None,
        random_state: int = 42,
        min_bucket_traces: int = 25,
        max_buckets: int = 64,
    ):
        self.requested_buckets = n_buckets
        self.random_state = random_state
        self.min_bucket_traces = min_bucket_traces
        self.max_buckets = max_buckets
        self.vectorizer: Any | None = None
        self.kmeans: Any | None = None
        self.bucket_sizes: dict[int, int] = {}
        self.top_terms_by_bucket: dict[int, list[str]] = {}

    @property
    def bucket_count(self) -> int:
        return len(self.bucket_sizes)

    def fit(self, traces: list[Trace]) -> dict[str, Any]:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not traces:
            raise ValueError("BucketRouter.fit requires at least one trace")

        texts = [_task_text(trace) for trace in traces]
        n_buckets = self._choose_bucket_count(len(traces))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = self.vectorizer.fit_transform(texts)
        self.kmeans = KMeans(n_clusters=n_buckets, random_state=self.random_state, n_init=20)
        labels = self.kmeans.fit_predict(matrix)

        sizes: dict[int, int] = {}
        for label in labels:
            bucket_id = int(label)
            sizes[bucket_id] = sizes.get(bucket_id, 0) + 1
        self.bucket_sizes = dict(sorted(sizes.items()))
        self.top_terms_by_bucket = self._top_terms_by_bucket()
        return self.metrics()

    def route(self, task_text: str) -> BucketAssignment:
        if self.vectorizer is None or self.kmeans is None:
            raise RuntimeError("BucketRouter must be fit before route")

        matrix = self.vectorizer.transform([task_text or "empty task"])
        distances = tuple(float(value) for value in self.kmeans.transform(matrix)[0])
        bucket_id = int(self.kmeans.predict(matrix)[0])
        confidence = _distance_confidence(distances)
        return BucketAssignment(bucket_id=bucket_id, confidence=confidence, distances=distances)

    def metrics(self) -> dict[str, Any]:
        sizes = list(self.bucket_sizes.values())
        supported = [size for size in sizes if size >= self.min_bucket_traces]
        return {
            "bucket_count": self.bucket_count,
            "bucket_sizes": self.bucket_sizes,
            "supported_bucket_count": len(supported),
            "under_supported_bucket_count": len(sizes) - len(supported),
            "min_bucket_size": min(sizes) if sizes else 0,
            "max_bucket_size": max(sizes) if sizes else 0,
            "avg_bucket_size": sum(sizes) / len(sizes) if sizes else 0.0,
            "top_terms_by_bucket": self.top_terms_by_bucket,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_buckets": self.requested_buckets,
            "random_state": self.random_state,
            "min_bucket_traces": self.min_bucket_traces,
            "max_buckets": self.max_buckets,
            "metrics": self.metrics(),
        }

    def _choose_bucket_count(self, trace_count: int) -> int:
        if trace_count <= 1:
            return 1
        if self.requested_buckets is not None:
            requested = self.requested_buckets
        else:
            requested = max(4, int(math.sqrt(trace_count / 2)))
        requested = max(1, min(requested, self.max_buckets, trace_count))

        # Keep buckets reasonably supported for first-pass training. This is a
        # soft sizing rule, not pruning.
        while requested > 1 and trace_count / requested < self.min_bucket_traces:
            requested -= 1
        return requested

    def _top_terms_by_bucket(self, limit: int = 8) -> dict[int, list[str]]:
        if self.vectorizer is None or self.kmeans is None:
            return {}
        terms = self.vectorizer.get_feature_names_out()
        top_terms: dict[int, list[str]] = {}
        for bucket_id, centroid in enumerate(self.kmeans.cluster_centers_):
            order = centroid.argsort()[::-1]
            top_terms[bucket_id] = [str(terms[index]) for index in order[:limit] if centroid[index] > 0]
        return top_terms


def _task_text(trace: Trace) -> str:
    return trace.task_text or trace.task_id or trace.trace_id


def _distance_confidence(distances: tuple[float, ...]) -> float:
    if not distances:
        return 0.0
    ordered = sorted(distances)
    if len(ordered) == 1:
        return 1.0
    nearest, second = ordered[0], ordered[1]
    if second <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - nearest / second))
