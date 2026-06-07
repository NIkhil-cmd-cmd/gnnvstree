"""Command-line runner for tree-framework evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnnvstree.loaders import load_jsonl_traces, load_live_runs, load_toolbench_queries
from gnnvstree.metrics import evaluate_model
from gnnvstree.split import split_traces
from gnnvstree.tree_model import TreeLogicAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the tree logic adapter on tool traces.")
    parser.add_argument("--toolbench", action="append", type=Path, default=[], help="ToolBench/StableToolBench query JSON file.")
    parser.add_argument("--live-runs", action="append", type=Path, default=[], help="Pareto-style runs.json file.")
    parser.add_argument("--jsonl", action="append", type=Path, default=[], help="Canonical JSONL trace file.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tree-eval"))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-suffix", type=int, default=6)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    traces = []
    for path in args.toolbench:
        traces.extend(load_toolbench_queries(path))
    for path in args.live_runs:
        traces.extend(load_live_runs(path))
    for path in args.jsonl:
        traces.extend(load_jsonl_traces(path))

    if not traces:
        raise SystemExit("No traces loaded. Pass --toolbench, --live-runs, or --jsonl.")

    train, val, test = split_traces(traces, seed=args.seed)
    eval_test = [*val, *test] if val else test
    model = TreeLogicAdapter(max_suffix=args.max_suffix, min_confidence=args.min_confidence)
    report = evaluate_model(model, train, eval_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "report.json", report.to_dict())
    _write_json(args.output_dir / "training_metrics.json", report.training)
    _write_json(args.output_dir / "test_metrics.json", report.testing)
    _write_json(args.output_dir / "structure_metrics.json", report.structure)
    _write_json(
        args.output_dir / "split_summary.json",
        {
            "train": [trace.trace_id for trace in train],
            "validation": [trace.trace_id for trace in val],
            "test": [trace.trace_id for trace in test],
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "top1_accuracy": report.testing["top1_accuracy"]}, indent=2))


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

