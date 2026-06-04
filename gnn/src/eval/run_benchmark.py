from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnn.src.agent.gemini_agent import GeminiToolAgent
from gnn.src.eval import metrics as M
from gnn.src.eval.metrics import _matches
from gnn.src.eval.retrievers import (
    BM25Retriever,
    FullListRetriever,
    GNNRetrieverWrapper,
    ToolBenchIRRetriever,
    load_benchmark_queries,
)
from gnn.src.infer.retrieve_tools import api_key_from_hf
from gnn.src.utils.config import ensure_dirs, load_config, repo_root


def eval_retriever(retriever, queries: list[dict], ks: list[int]) -> dict:
    per_query = []
    timings = []
    for q in queries:
        retrieved, tms = retriever.retrieve(
            q["query"], q["api_list"], q.get("path"), top_k=max(ks)
        )
        timings.append(tms)
        rel = q["relevant_keys"]
        path = q.get("path") or rel
        row = {"query_id": q["query_id"], "split": q["split"], "retriever": retriever.name}
        for k in ks:
            row[f"P@{k}"] = M.precision_at_k(retrieved, rel, k)
            row[f"R@{k}"] = M.recall_at_k(retrieved, rel, k)
            row[f"F1@{k}"] = M.f1_at_k(retrieved, rel, k)
            row[f"Hit@{k}"] = M.hit_at_k(retrieved, rel, k)
            row[f"PathRecall@{k}"] = M.path_recall_at_k(retrieved, path, k)
            row[f"PathF1@{k}"] = M.path_f1_at_k(retrieved, path, k)
            row[f"OrderedPath@{k}"] = M.ordered_path_match_at_k(retrieved, path, k)
            row[f"WorkflowCov@{k}"] = M.workflow_coverage(retrieved, path, k)
        row["MRR"] = M.mrr(retrieved, rel)
        row["MAP"] = M.average_precision(retrieved, rel)
        row["NDCG@5"] = M.ndcg_at_k(retrieved, rel, 5)
        row["NDCG@10"] = M.ndcg_at_k(retrieved, rel, 10)
        row["NextToolAcc@1"] = M.next_tool_acc_at_1(retrieved, path, 0)
        per_query.append(row)

    agg: dict[str, float] = {"retriever": retriever.name, "n_queries": len(per_query)}
    if per_query:
        for key in per_query[0]:
            if key in ("query_id", "split", "retriever"):
                continue
            vals = [r[key] for r in per_query]
            agg[key] = sum(vals) / len(vals)
    agg.update(M.aggregate_timings(timings))
    return {"aggregate": agg, "per_query_sample": per_query[:20]}


def run_agent_benchmark_dry(cfg: dict, queries: list[dict], gnn_wrapper: GNNRetrieverWrapper) -> dict:
    """Deterministic agent proxy (no API): picks top-1 from retriever."""
    sample = queries[: cfg["gemini_eval_samples"]]
    episodes = []
    for q in sample:
        for condition, retriever in (
            ("baseline_bm25", BM25Retriever()),
            ("gnn_agent", gnn_wrapper),
        ):
            keys, tms = retriever.retrieve(q["query"], q["api_list"], q.get("path"), cfg["top_k"])
            selected = keys[: max(1, len(q["relevant_keys"]))]
            rel = q["relevant_keys"]
            path = q.get("path") or rel
            episodes.append(
                {
                    "query_id": q["query_id"],
                    "condition": condition,
                    "selected_tools": selected,
                    "selection_accuracy": 1.0
                    if selected and any(_matches(selected[0], r) for r in rel)
                    else 0.0,
                    "multi_tool_selection_recall": sum(
                        1 for r in rel if any(_matches(s, r) for s in selected)
                    )
                    / max(len(rel), 1),
                    "path_success": 1.0
                    if all(any(_matches(s, p) for s in selected) for p in path)
                    else 0.0,
                    "hallucination_rate": 0.0,
                    "tools_retrieved": len(keys),
                    "tools_called": len(selected),
                    "unnecessary_tool_calls": 0,
                    "steps_to_complete": len(selected),
                    "episode_success": 1.0
                    if all(any(_matches(s, r) for s in selected) for r in rel)
                    else 0.0,
                    "wall_clock_s": tms.get("total_retrieval_ms", 0) / 1000,
                    "gemini_latency_ms": 0.0,
                    "retrieval_latency_ms": tms.get("total_retrieval_ms", 0),
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "model": "dry_run",
                }
            )
    agg: dict[str, dict[str, float]] = defaultdict(dict)
    by_cond: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_cond[ep["condition"]].append(ep)
    for cond, eps in by_cond.items():
        n = len(eps)
        for field in (
            "selection_accuracy",
            "multi_tool_selection_recall",
            "path_success",
            "hallucination_rate",
            "episode_success",
            "wall_clock_s",
            "gemini_latency_ms",
            "retrieval_latency_ms",
            "tools_retrieved",
            "tools_called",
        ):
            agg[cond][field] = sum(e[field] for e in eps) / n if n else 0.0
        agg[cond]["n_episodes"] = n
    return {"aggregate": agg, "episodes_sample": episodes[:10], "mode": "dry_run"}


def run_agent_benchmark(
    cfg: dict,
    queries: list[dict],
    gnn_wrapper: GNNRetrieverWrapper,
    *,
    allow_dry_run: bool = False,
) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        if allow_dry_run:
            print("WARN: No GEMINI_API_KEY — running --dry-run-agent (not real Gemini)")
            return run_agent_benchmark_dry(cfg, queries, gnn_wrapper)
        raise RuntimeError(
            "GEMINI_API_KEY is required for real agent evaluation. "
            "Get a free key at https://aistudio.google.com/ then run:\n"
            "  export GEMINI_API_KEY='your-key'\n"
            "  python -m gnn.src.eval.run_benchmark --config gnn/configs/cpu_g2_g3.yaml\n"
            "Or pass --dry-run-agent only for offline testing (not real LLM calls)."
        )

    agent = GeminiToolAgent(
        cfg["gemini_model"],
        cfg["gemini_daily_budget"],
        fallback_models=cfg.get("gemini_fallback_models"),
        min_request_interval_s=float(cfg.get("gemini_min_request_interval_s", 4.0)),
    )
    sample = queries[: cfg["gemini_eval_samples"]]
    episodes = []
    quota_error: str | None = None
    bm25 = BM25Retriever()
    for q in sample:
        if quota_error:
            break
        for condition, retriever in (("baseline_bm25", bm25), ("gnn_agent", gnn_wrapper)):
            keys, tms = retriever.retrieve(q["query"], q["api_list"], q.get("path"), cfg["top_k"])
            retrieval_ms = tms.get("total_retrieval_ms", 0.0)
            ranked = [(k, 1.0 - i * 0.1) for i, k in enumerate(keys)]
            try:
                ep = agent.run_episode(
                    q["query_id"],
                    q["query"],
                    ranked,
                    q["relevant_keys"],
                    q.get("path") or q["relevant_keys"],
                    condition,
                    retrieval_latency_ms=retrieval_ms,
                )
                episodes.append(ep.__dict__)
            except RuntimeError as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    quota_error = str(e)
                    print(f"Gemini quota hit after {len(episodes)} episodes: {e}")
                    break
                raise
    agg: dict[str, dict[str, float]] = defaultdict(dict)
    by_cond: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_cond[ep["condition"]].append(ep)
    for cond, eps in by_cond.items():
        n = len(eps)
        for field in (
            "selection_accuracy",
            "multi_tool_selection_recall",
            "path_success",
            "hallucination_rate",
            "episode_success",
            "wall_clock_s",
            "gemini_latency_ms",
            "retrieval_latency_ms",
            "tools_retrieved",
            "tools_called",
            "unnecessary_tool_calls",
            "steps_to_complete",
            "total_tokens",
            "cost_usd",
        ):
            agg[cond][field] = sum(e[field] for e in eps) / n if n else 0.0
        agg[cond]["n_episodes"] = n
    out = {
        "aggregate": agg,
        "episodes": episodes,
        "mode": "live_gemini",
        "model": cfg["gemini_model"],
        "models_tried": agent.models_to_try,
    }
    if quota_error:
        out["partial"] = True
        out["quota_error"] = quota_error
    return out


def write_report(results_dir: Path, report: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# GNN ToolBench Benchmark Report\n"]
    for name, block in report.get("retrieval", {}).items():
        lines.append(f"## Retrieval: {name}\n")
        for k, v in block.get("aggregate", {}).items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    if "agent" in report and not report["agent"].get("skipped"):
        mode = report["agent"].get("mode", "unknown")
        lines.append(f"## Gemini Agent (mode: {mode})\n")
        if report["agent"].get("model"):
            lines.append(f"- model: {report['agent']['model']}\n")
        for cond, stats in report["agent"].get("aggregate", {}).items():
            lines.append(f"### {cond}\n")
            for k, v in stats.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
            lines.append("")
    with open(results_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gnn/configs/cpu_g2_g3.yaml")
    parser.add_argument("--skip-ir", action="store_true", help="Skip ToolBench IR (slow download)")
    parser.add_argument("--skip-agent", action="store_true", help="Skip agent eval entirely")
    parser.add_argument(
        "--dry-run-agent",
        action="store_true",
        help="Offline fake agent (no API). Default requires GEMINI_API_KEY for real calls.",
    )
    args = parser.parse_args()

    cfg = load_config(repo_root() / args.config)
    ensure_dirs(cfg)
    results_dir = Path(cfg["results_dir"])
    ks = cfg["retrieval_ks"]

    queries = load_benchmark_queries(Path(cfg["data_dir"]), cfg["hf_benchmark_splits"])
    if not queries:
        raise RuntimeError("No benchmark queries. Run download_toolbench first.")

    report: dict = {"retrieval": {}, "config": {k: cfg[k] for k in ("top_k", "gemini_model") if k in cfg}}
    retrievers = [
        BM25Retriever(),
        FullListRetriever(),
        GNNRetrieverWrapper(cfg),
    ]
    if not args.skip_ir:
        try:
            retrievers.insert(0, ToolBenchIRRetriever(cfg["toolbench_ir_model"]))
        except Exception as e:
            print(f"WARN: ToolBench IR unavailable: {e}")

    gnn_wrapper = GNNRetrieverWrapper(cfg)
    for ret in retrievers:
        print(f"Evaluating retriever: {ret.name}")
        t0 = time.perf_counter()
        report["retrieval"][ret.name] = eval_retriever(ret, queries, ks)
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    if not args.skip_agent:
        try:
            report["agent"] = run_agent_benchmark(
                cfg, queries, gnn_wrapper, allow_dry_run=args.dry_run_agent
            )
        except Exception as e:
            report["agent"] = {"mode": "failed", "error": str(e), "episodes": []}
            print(f"Agent benchmark failed: {e}")
    else:
        report["agent"] = {
            "skipped": True,
            "reason": "Agent eval skipped via --skip-agent",
        }

    write_report(results_dir, report)

    csv_path = results_dir / "retrieval_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["retriever"] + sorted(
            {k for block in report["retrieval"].values() for k in block["aggregate"] if k != "retriever"}
        )
        writer.writerow(headers)
        for name, block in report["retrieval"].items():
            agg = block["aggregate"]
            writer.writerow([name] + [agg.get(h, "") for h in headers[1:]])

    print(f"Results written to {results_dir}")
    gnn_wrapper.retriever.close()


if __name__ == "__main__":
    main()
