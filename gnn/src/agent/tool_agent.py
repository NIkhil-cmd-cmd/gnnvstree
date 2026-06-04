from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from gnn.src.agent.llm_client import BaseLLMClient, build_client
from gnn.src.eval.metrics import _matches
from gnn.src.infer.retrieve_tools import api_match_key


AGENT_SYSTEM = """You are a multi-step tool-planning agent (like ReAct for APIs).
You must build a workflow to answer the user's request by calling tools ONE AT A TIME.
Rules:
- Only choose tools from the CANDIDATE list (exact string match).
- Call each tool at most once unless necessary.
- When the workflow is complete, respond with {"action": "finish"}.
- Each step: {"action": "call", "tool": "<exact candidate id>"} OR {"action": "finish"}
- Output JSON only, no markdown."""


@dataclass
class AgentEpisodeResult:
    query_id: str
    condition: str
    selected_tools: list[str]
    selection_accuracy: float
    multi_tool_selection_recall: float
    path_success: float
    hallucination_rate: float
    tools_retrieved: int
    tools_called: int
    unnecessary_tool_calls: int
    steps_to_complete: int
    episode_success: float
    wall_clock_s: float
    llm_latency_ms: float
    retrieval_latency_ms: float
    total_tokens: int
    cost_usd: float
    model: str
    llm_calls: int = 0
    trace: list[dict] = field(default_factory=list)
    raw_response: str = ""


def _parse_step(text: str) -> tuple[str, str | None]:
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return "unknown", None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return "unknown", None
    action = str(obj.get("action", "")).lower()
    if action == "finish":
        return "finish", None
    tool = obj.get("tool") or obj.get("tool_name")
    return "call", str(tool).strip() if tool else None


def _match_candidate(picked: str, allowed: list[str]) -> str | None:
    if picked in allowed:
        return picked
    for a in allowed:
        if api_match_key(picked, a) or api_match_key(a, picked):
            return a
    return None


class ToolWorkflowAgent:
    """Multi-step tool selection agent backed by a real LLM."""

    def __init__(self, cfg: dict):
        self.llm: BaseLLMClient = build_client(cfg)
        self.max_steps = int(cfg.get("agent_max_steps", 5))
        self.daily_budget = int(cfg.get("llm_daily_budget", cfg.get("gemini_daily_budget", 100)))
        self._calls = 0

    def _budget_check(self) -> None:
        if self._calls >= self.daily_budget:
            raise RuntimeError(f"LLM call budget exceeded ({self.daily_budget})")

    def run_episode(
        self,
        query_id: str,
        query: str,
        ranked_tools: list[tuple[str, float]],
        relevant_keys: list[str],
        path: list[str],
        condition: str,
        retrieval_latency_ms: float = 0.0,
    ) -> AgentEpisodeResult:
        allowed = [t for t, _ in ranked_tools]
        allowed_set = set(allowed)
        candidates_block = "\n".join(
            f"  {i + 1}. {name} (retriever_score={score:.3f})" for i, (name, score) in enumerate(ranked_tools)
        )
        selected: list[str] = []
        trace: list[dict] = []
        llm_ms = 0.0
        hallucinations = 0
        t0 = time.perf_counter()

        for step in range(self.max_steps):
            self._budget_check()
            used = ", ".join(selected) if selected else "(none)"
            prompt = (
                f"USER QUERY:\n{query}\n\n"
                f"CANDIDATE TOOLS (pick exact id):\n{candidates_block}\n\n"
                f"TOOLS ALREADY CALLED: {used}\n\n"
                f"Step {step + 1}/{self.max_steps}. What is the next action?"
            )
            self._calls += 1
            resp = self.llm.generate_with_retry(prompt, system=AGENT_SYSTEM)
            llm_ms += resp.latency_ms
            action, tool = _parse_step(resp.text)
            trace.append({"step": step + 1, "action": action, "tool": tool, "raw": resp.text[:300]})

            if action == "finish":
                break
            if action == "call" and tool:
                matched = _match_candidate(tool, allowed)
                if matched:
                    if matched not in selected:
                        selected.append(matched)
                else:
                    hallucinations += 1
            # stop early if all gold tools covered
            rel = relevant_keys
            if rel and all(any(_matches(s, r) for s in selected) for r in rel):
                break

        wall = time.perf_counter() - t0
        path_set = path or relevant_keys
        sel_acc = 1.0 if selected and path_set and _matches(selected[0], path_set[0]) else 0.0
        multi_recall = (
            sum(1 for r in relevant_keys if any(_matches(s, r) for s in selected)) / len(relevant_keys)
            if relevant_keys
            else 0.0
        )
        ordered_ok = selected == list(path_set)[: len(selected)] and len(selected) >= len(path_set)
        subset_ok = all(any(_matches(s, p) for s in selected) for p in path_set)
        path_success = 1.0 if ordered_ok else (1.0 if subset_ok else 0.0)
        episode_success = 1.0 if subset_ok else 0.0
        unnecessary = sum(
            1 for s in selected if not any(_matches(s, p) for p in path_set)
        )
        hall_rate = hallucinations / max(self.max_steps, 1)

        cost = 0.0
        if hasattr(self.llm, "estimate_cost_usd"):
            cost = self.llm.estimate_cost_usd()

        return AgentEpisodeResult(
            query_id=query_id,
            condition=condition,
            selected_tools=selected,
            selection_accuracy=sel_acc,
            multi_tool_selection_recall=multi_recall,
            path_success=path_success,
            hallucination_rate=hall_rate,
            tools_retrieved=len(ranked_tools),
            tools_called=len(selected),
            unnecessary_tool_calls=unnecessary,
            steps_to_complete=len(trace),
            episode_success=episode_success,
            wall_clock_s=wall,
            llm_latency_ms=llm_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            total_tokens=self.llm.total_input_tokens + self.llm.total_output_tokens,
            cost_usd=cost,
            model=getattr(self.llm, "model", "unknown"),
            llm_calls=len(trace),
            trace=trace,
            raw_response=trace[-1]["raw"] if trace else "",
        )


def hypothesis_summary(aggregate: dict) -> dict:
    """Compare GNN vs BM25 on metrics that test the hypothesis."""
    gnn = aggregate.get("gnn_agent", {})
    bm25 = aggregate.get("baseline_bm25", {})
    metrics = [
        "multi_tool_selection_recall",
        "path_success",
        "episode_success",
        "selection_accuracy",
    ]
    comparison = {}
    gnn_wins = 0
    for m in metrics:
        gv, bv = gnn.get(m, 0), bm25.get(m, 0)
        delta = gv - bv
        comparison[m] = {"gnn": gv, "bm25": bv, "delta": delta, "gnn_better": delta > 0}
        if delta > 0:
            gnn_wins += 1
    supported = gnn_wins >= 2 and gnn.get("multi_tool_selection_recall", 0) >= bm25.get("multi_tool_selection_recall", 0)
    return {
        "comparison": comparison,
        "gnn_wins_on": gnn_wins,
        "hypothesis_supported": supported,
        "verdict": "GNN retrieval helps the agent" if supported else "GNN retrieval does not beat BM25 on agent metrics (this run)",
    }
