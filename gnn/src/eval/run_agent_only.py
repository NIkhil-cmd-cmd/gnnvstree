"""Run live multi-step agent benchmark only (skips retrieval re-eval)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn.src.eval.retrievers import GNNRetrieverWrapper, load_benchmark_queries
from gnn.src.eval.run_benchmark import run_agent_benchmark, write_report
from gnn.src.utils.config import ensure_dirs, load_config, repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gnn/configs/cpu_g2_g3.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from gnn/results/agent_episodes.jsonl (skip finished query/condition pairs)",
    )
    args = parser.parse_args()
    cfg = load_config(repo_root() / args.config)
    ensure_dirs(cfg)
    results_dir = Path(cfg["results_dir"])

    report_path = results_dir / "benchmark_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {"retrieval": {}, "config": {}}

    queries = load_benchmark_queries(Path(cfg["data_dir"]), cfg["hf_benchmark_splits"])
    gnn_wrapper = GNNRetrieverWrapper(cfg)
    print("Running live ReAct-style agent benchmark (GNN vs BM25)...")
    report["agent"] = run_agent_benchmark(
        cfg, queries, gnn_wrapper, results_dir=results_dir, resume=args.resume
    )
    write_report(results_dir, report)
    gnn_wrapper.retriever.close()
    print(f"Done. See {results_dir}/REPORT.md and agent_episodes.jsonl")


if __name__ == "__main__":
    main()
