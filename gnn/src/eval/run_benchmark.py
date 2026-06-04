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

from gnn.src.agent.tool_agent import ToolWorkflowAgent, hypothesis_summary
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
                    "llm_latency_ms": 0.0,
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


def _episode_to_dict(ep) -> dict:
    d = ep.__dict__.copy()
    d.pop("trace", None)
    return d


def _load_agent_episodes_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def run_agent_benchmark(
    cfg: dict,
    queries: list[dict],
    gnn_wrapper: GNNRetrieverWrapper,
    *,
    allow_dry_run: bool = False,
    results_dir: Path | None = None,
    resume: bool = False,
) -> dict:
    provider = (cfg.get("llm_provider") or "anthropic").lower()
    has_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        if provider == "anthropic"
        else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    )
    if not has_key:
        if allow_dry_run:
            print("WARN: No API key — running --dry-run-agent (not a real LLM)")
            return run_agent_benchmark_dry(cfg, queries, gnn_wrapper)
        raise RuntimeError(
            f"Set {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'GEMINI_API_KEY'} "
            "for real multi-step agent eval. See gnn/README.md"
        )

    agent = ToolWorkflowAgent(cfg)
    sample = queries[: cfg["gemini_eval_samples"]]
    partial_error: str | None = None
    bm25 = BM25Retriever()
    jsonl_path = (results_dir / "agent_episodes.jsonl") if results_dir else None
    episodes: list[dict] = []
    done_keys: set[tuple[str, str]] = set()
    if jsonl_path and resume:
        episodes = _load_agent_episodes_jsonl(jsonl_path)
        done_keys = {(e["query_id"], e["condition"]) for e in episodes}
        if episodes:
            print(f"Resuming agent eval: {len(episodes)} episodes already in {jsonl_path.name}")
    elif jsonl_path:
        jsonl_path.unlink(missing_ok=True)

    total = len(sample) * 2
    n_done = len(done_keys)
    for qi, q in enumerate(sample):
        if partial_error:
            break
        for condition, retriever in (("baseline_bm25", bm25), ("gnn_agent", gnn_wrapper)):
            ep_key = (q["query_id"], condition)
            if ep_key in done_keys:
                continue
            keys, tms = retriever.retrieve(q["query"], q["api_list"], q.get("path"), cfg["top_k"])
            retrieval_ms = tms.get("total_retrieval_ms", 0.0)
            ranked = [(k, 1.0 - i * 0.1) for i, k in enumerate(keys)]
            n_done += 1
            print(
                f"[agent {n_done}/{total}] {condition} query={q['query_id']} "
                f"retrieved={len(keys)} llm={getattr(agent.llm, 'model', '?')}",
                flush=True,
            )
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
                row = _episode_to_dict(ep)
                episodes.append(row)
                if jsonl_path:
                    with open(jsonl_path, "a") as jf:
                        jf.write(json.dumps(row) + "\n")
                print(
                    f"  -> tools={ep.selected_tools} recall={ep.multi_tool_selection_recall:.2f} "
                    f"success={ep.episode_success:.0f} llm_calls={ep.llm_calls}",
                    flush=True,
                )
            except Exception as e:
                err = str(e)
                if "budget exceeded" in err.lower() or any(
                    x in err.lower() for x in ("429", "503", "quota", "unavailable", "rate")
                ):
                    partial_error = err
                    print(f"Stopped after {len(episodes)} episodes: {e}")
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
            "llm_latency_ms",
            "retrieval_latency_ms",
            "tools_retrieved",
            "tools_called",
            "unnecessary_tool_calls",
            "steps_to_complete",
            "total_tokens",
            "cost_usd",
            "llm_calls",
        ):
            agg[cond][field] = sum(e.get(field, 0) for e in eps) / n if n else 0.0
        agg[cond]["n_episodes"] = n

    hyp = hypothesis_summary(agg)
    print(f"\n=== Hypothesis: {hyp['verdict']} ===")
    for m, c in hyp["comparison"].items():
        print(f"  {m}: GNN={c['gnn']:.3f} BM25={c['bm25']:.3f} delta={c['delta']:+.3f}")

    out = {
        "aggregate": agg,
        "episodes": episodes,
        "mode": "live_react_agent",
        "llm_provider": provider,
        "model": getattr(agent.llm, "model", cfg.get("anthropic_model") or cfg.get("gemini_model")),
        "agent_max_steps": cfg.get("agent_max_steps", 5),
        "hypothesis": hyp,
    }
    if partial_error:
        out["partial"] = True
        out["error"] = partial_error
    if hasattr(agent.llm, "estimate_cost_usd"):
        out["estimated_cost_usd"] = agent.llm.estimate_cost_usd()
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
    if "agent" in report and not report["agent"].get("skipped") and report["agent"].get("mode") != "failed":
        ag = report["agent"]
        lines.append(f"## Agent eval ({ag.get('mode', '?')})\n")
        lines.append(f"- provider: {ag.get('llm_provider', '?')}")
        lines.append(f"- model: {ag.get('model', '?')}")
        if ag.get("estimated_cost_usd") is not None:
            lines.append(f"- estimated_cost_usd: {ag['estimated_cost_usd']:.4f}")
        if ag.get("hypothesis"):
            lines.append(f"- **{ag['hypothesis']['verdict']}**\n")
        for cond, stats in ag.get("aggregate", {}).items():
            lines.append(f"### {cond}\n")
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {v:.4f}")
            lines.append("")
    elif report.get("agent", {}).get("mode") == "failed":
        lines.append(f"## Agent eval failed\n\n{report['agent'].get('error', '')}\n")
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
                cfg,
                queries,
                gnn_wrapper,
                allow_dry_run=args.dry_run_agent,
                results_dir=results_dir,
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
