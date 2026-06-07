"""Trace loaders for ToolBench-style data and live Pareto-style artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from gnnvstree.trace import Trace, make_trace, normalize_name


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _api_key(tool_name: str, api_name: str) -> str:
    return normalize_name(f"{tool_name}_{api_name}")


def _candidate_names(api_list: Iterable[dict[str, Any]]) -> list[str]:
    names = []
    for api in api_list:
        tool_name = api.get("tool_name") or api.get("tool") or api.get("tool_name_standardized") or ""
        api_name = api.get("api_name") or api.get("name") or ""
        if tool_name or api_name:
            names.append(_api_key(str(tool_name), str(api_name)))
    return names


def _relevant_names(item: dict[str, Any]) -> list[str]:
    names = []
    for pair in item.get("relevant APIs", []) or item.get("relevant_apis", []):
        if isinstance(pair, list) and len(pair) >= 2:
            names.append(_api_key(str(pair[0]), str(pair[1])))
        elif isinstance(pair, dict):
            names.append(_api_key(str(pair.get("tool_name", "")), str(pair.get("api_name", ""))))
    return names


def load_toolbench_queries(path: str | Path, split: str = "train", source: str = "toolbench") -> list[Trace]:
    """Load ToolBench/StableToolBench query files into API-level traces.

    Important: candidate ``api_list`` entries are recorded as candidates only.
    Ground-truth trace steps come from ``relevant APIs``.
    """

    data = _read_json(Path(path))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of ToolBench query objects in {path}")

    traces: list[Trace] = []
    for index, item in enumerate(data):
        relevant = _relevant_names(item)
        candidate_tools = _candidate_names(item.get("api_list", []))
        if not relevant:
            continue
        task_id = str(item.get("query_id", index))
        traces.append(
            make_trace(
                source=source,
                task_id=task_id,
                split=split,
                success=True,
                tool_names=relevant,
                task_text=str(item.get("query", "")),
                candidate_tools=candidate_tools,
                ground_truth_tools=relevant,
                metadata={
                    "query_id": task_id,
                    "has_ordered_answer_trace": False,
                    "raw_relevant_api_count": len(relevant),
                    "candidate_api_count": len(candidate_tools),
                },
            )
        )
    return traces


def load_live_runs(path: str | Path, split: str = "live", source: str = "live_pareto") -> list[Trace]:
    """Load Pareto-style ``runs.json`` artifacts emitted by live agents."""

    data = _read_json(Path(path))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of live run objects in {path}")

    traces: list[Trace] = []
    for index, item in enumerate(data):
        trace = item.get("trace") or []
        if not trace:
            tool_calls = item.get("tool_calls") or []
            trace = [call.get("name") for call in tool_calls if isinstance(call, dict) and call.get("name")]
        if not trace:
            continue
        task = item.get("task") if isinstance(item.get("task"), dict) else {}
        task_id = str(task.get("name") or item.get("index") or index)
        traces.append(
            make_trace(
                source=source,
                task_id=task_id,
                split=split,
                success=bool(item.get("success") or item.get("valid") or item.get("exact_optimal")),
                tool_names=[str(name) for name in trace],
                semantic_steps=[str(step) for step in item.get("semantic_steps", [])],
                task_text=str(task.get("description") or task.get("name") or ""),
                metadata={
                    "phase": item.get("phase"),
                    "index": item.get("index"),
                    "tokens": item.get("tokens", 0),
                    "latency_ms": item.get("latency_ms", 0),
                    "valid": item.get("valid"),
                    "exact_optimal": item.get("exact_optimal"),
                    "error": item.get("error"),
                },
            )
        )
    return traces


def load_jsonl_traces(path: str | Path) -> list[Trace]:
    traces = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            traces.append(
                make_trace(
                    source=str(item.get("source", "jsonl")),
                    task_id=str(item.get("task_id", item.get("trace_id", len(traces)))),
                    split=str(item.get("split", "unknown")),
                    success=bool(item.get("success", True)),
                    tool_names=[str(name) for name in item.get("tool_names", item.get("trace", []))],
                    semantic_steps=[str(step) for step in item.get("semantic_steps", [])],
                    task_text=str(item.get("task_text", "")),
                    candidate_tools=[str(tool) for tool in item.get("candidate_tools", [])],
                    ground_truth_tools=[str(tool) for tool in item.get("ground_truth_tools", [])],
                    metadata=dict(item.get("metadata", {})),
                )
            )
    return traces

