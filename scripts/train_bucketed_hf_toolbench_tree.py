#!/usr/bin/env python3
"""Train a bucketed start-state action tree on HF ToolBench traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnnvstree.bucketed_tree import BucketedActionTreeAdapter
from gnnvstree.hf_toolbench import iter_hf_toolbench_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tuandunghcmut/toolbench-v1")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-actions", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--min-bucket-traces", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf-toolbench-bucketed-tree"))
    parser.add_argument(
        "--tool-only",
        action="store_true",
        help="Ignore action inputs and train on action/tool names only.",
    )
    args = parser.parse_args()

    traces, stats = iter_hf_toolbench_traces(
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.limit,
        min_actions=args.min_actions,
        include_action_inputs=not args.tool_only,
    )
    if not traces:
        raise SystemExit("No traces extracted; cannot train.")

    model = BucketedActionTreeAdapter(
        n_buckets=args.buckets,
        min_bucket_traces=args.min_bucket_traces,
    )
    metrics = model.fit(traces)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "model.json", model.to_dict())
    _write_json(args.output_dir / "training_metrics.json", metrics)
    _write_json(args.output_dir / "bucket_metrics.json", metrics["router"])
    _write_json(args.output_dir / "extraction_summary.json", stats.to_dict())
    _write_json(args.output_dir / "trace_sample.json", [trace.to_dict() for trace in traces[:10]])

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows_scanned": stats.rows_scanned,
                "traces_extracted": stats.traces_extracted,
                "action_count": stats.action_count,
                "parsed_input_rate": stats.to_dict()["parsed_input_rate"],
                "bucket_count": metrics["bucket_count"],
                "supported_bucket_count": metrics["supported_bucket_count"],
                "bucket_sizes": metrics["bucket_sizes"],
                "node_count": metrics["node_count"],
                "leaf_count": metrics["leaf_count"],
                "max_depth": metrics["max_depth"],
                "unique_actions": metrics["unique_actions"],
            },
            indent=2,
        )
    )


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
