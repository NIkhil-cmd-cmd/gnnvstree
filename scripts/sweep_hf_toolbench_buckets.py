#!/usr/bin/env python3
"""Sweep task-bucketing choices on HF ToolBench traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from gnnvstree.hf_toolbench import iter_hf_toolbench_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--ks", default="8,16,32,64")
    parser.add_argument("--min-bucket-traces", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf-toolbench-bucket-sweep"))
    parser.add_argument("--top-terms", type=int, default=8)
    args = parser.parse_args()

    traces, stats = iter_hf_toolbench_traces(limit=args.limit)
    texts = [trace.task_text or trace.task_id for trace in traces]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    dense = matrix.toarray()
    terms = vectorizer.get_feature_names_out()

    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    results: list[dict[str, Any]] = []
    for k in ks:
        if k < 2 or k >= len(traces):
            continue
        results.append(_summarize_kmeans(matrix, terms, k, args.min_bucket_traces, args.top_terms))
        results.append(_summarize_agglomerative(dense, matrix, terms, k, args.min_bucket_traces, args.top_terms))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "sweep_report.json", {"extraction": stats.to_dict(), "results": results})

    print(json.dumps({"output_dir": str(args.output_dir), "extraction": stats.to_dict(), "results": _compact(results)}, indent=2))


def _summarize_kmeans(matrix, terms, k: int, min_bucket_traces: int, top_terms: int) -> dict[str, Any]:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)
    distances = model.transform(matrix)
    margins = _nearest_margin(distances)
    centers = model.cluster_centers_
    return _summary(
        method="kmeans",
        k=k,
        labels=labels,
        min_bucket_traces=min_bucket_traces,
        silhouette=_silhouette(matrix, labels, metric="cosine"),
        route_confidence=margins,
        top_terms=_top_terms_from_centers(centers, terms, top_terms),
    )


def _summarize_agglomerative(dense, matrix, terms, k: int, min_bucket_traces: int, top_terms: int) -> dict[str, Any]:
    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    labels = model.fit_predict(dense)
    centers = _centers_from_labels(matrix, labels, k)
    distances = cosine_distances(matrix, centers)
    margins = _nearest_margin(distances)
    return _summary(
        method="hierarchical_average_cosine",
        k=k,
        labels=labels,
        min_bucket_traces=min_bucket_traces,
        silhouette=_silhouette(matrix, labels, metric="cosine"),
        route_confidence=margins,
        top_terms=_top_terms_from_centers(centers, terms, top_terms),
    )


def _summary(
    *,
    method: str,
    k: int,
    labels,
    min_bucket_traces: int,
    silhouette: float,
    route_confidence: list[float],
    top_terms: dict[int, list[str]],
) -> dict[str, Any]:
    sizes = Counter(int(label) for label in labels)
    size_values = list(sizes.values())
    supported = [size for size in size_values if size >= min_bucket_traces]
    return {
        "method": method,
        "k": k,
        "silhouette": silhouette,
        "bucket_sizes": dict(sorted(sizes.items())),
        "supported_bucket_count": len(supported),
        "under_supported_bucket_count": len(size_values) - len(supported),
        "min_bucket_size": min(size_values),
        "max_bucket_size": max(size_values),
        "avg_bucket_size": sum(size_values) / len(size_values),
        "route_confidence_mean": float(np.mean(route_confidence)) if route_confidence else 0.0,
        "route_confidence_p10": _percentile(route_confidence, 10),
        "route_confidence_p50": _percentile(route_confidence, 50),
        "top_terms_by_bucket": top_terms,
    }


def _silhouette(matrix, labels, metric: str) -> float:
    if len(set(labels)) <= 1:
        return 0.0
    return float(silhouette_score(matrix, labels, metric=metric))


def _nearest_margin(distances) -> list[float]:
    margins = []
    for row in np.asarray(distances):
        ordered = sorted(float(value) for value in row)
        if len(ordered) < 2:
            margins.append(1.0)
            continue
        nearest, second = ordered[0], ordered[1]
        margins.append(max(0.0, min(1.0, 1.0 - nearest / (second + 1e-9))))
    return margins


def _centers_from_labels(matrix, labels, k: int):
    centers = []
    for bucket_id in range(k):
        indices = np.where(labels == bucket_id)[0]
        if len(indices) == 0:
            centers.append(np.zeros(matrix.shape[1]))
            continue
        centers.append(np.asarray(matrix[indices].mean(axis=0)).reshape(-1))
    return np.vstack(centers)


def _top_terms_from_centers(centers, terms, limit: int) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for bucket_id, center in enumerate(np.asarray(centers)):
        order = center.argsort()[::-1]
        result[bucket_id] = [str(terms[index]) for index in order[:limit] if center[index] > 0]
    return result


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def _compact(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "method",
        "k",
        "silhouette",
        "supported_bucket_count",
        "under_supported_bucket_count",
        "min_bucket_size",
        "max_bucket_size",
        "route_confidence_mean",
        "route_confidence_p10",
        "route_confidence_p50",
    )
    return [{key: result[key] for key in keys} for result in results]


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
