#!/usr/bin/env python3
"""Run a tiny Gemini replay comparison against the bucketed action tree."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gnnvstree.bucketed_tree import BucketedActionTreeAdapter
from gnnvstree.hf_toolbench import iter_hf_toolbench_traces
from gnnvstree.trace import Trace

_LAST_GEMINI_REQUEST_AT = 0.0


@dataclass
class ReplayStep:
    task_id: str
    condition: str
    step: int
    prefix: list[str]
    gold_action: str
    recommended_action: str | None
    recommendation_confidence: float
    selected_action: str | None
    matched_gold: bool
    hallucinated: bool
    raw_response: str
    latency_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--offset", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--buckets", type=int, default=16)
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument(
        "--conditions",
        default="baseline,tree_suggested",
        help="Comma-separated conditions: baseline,tree_suggested,bm25_suggested.",
    )
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemini-min-interval-s", type=float, default=6.0)
    parser.add_argument(
        "--require-recommendation",
        action="store_true",
        help="Select only held-out traces where the tree can recommend at START.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gemini-replay-smoke"))
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini replay.")

    train_traces, _ = iter_hf_toolbench_traces(limit=args.train_limit)
    model = BucketedActionTreeAdapter(n_buckets=args.buckets, min_bucket_traces=25)
    model.fit(train_traces)
    action_counts = Counter(action for trace in train_traces for action in trace.tool_names)
    distractor_pool = [action for action, _ in action_counts.most_common()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    known_conditions = {"baseline", "tree_suggested", "bm25_suggested"}
    unknown = sorted(set(conditions) - known_conditions)
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")

    eval_limit = args.offset + max(500, args.episodes * 100)
    eval_traces, _ = iter_hf_toolbench_traces(limit=eval_limit)
    heldout = []
    skipped_no_recommendation = 0
    for trace in eval_traces[args.offset:]:
        if len(trace.tool_names) < 2:
            continue
        if args.require_recommendation:
            candidates = _candidate_actions(trace)
            recommendation = model.recommend_next(trace.task_text, [], set(candidates))
            if recommendation is None:
                skipped_no_recommendation += 1
                continue
        heldout.append(trace)
        if len(heldout) >= args.episodes:
            break
    if not heldout:
        raise SystemExit("No held-out traces available for replay.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "episodes.jsonl"
    steps: list[ReplayStep] = []
    with log_path.open("w", encoding="utf-8") as handle:
        for episode_idx, trace in enumerate(heldout):
            for condition in conditions:
                prefix: list[str] = []
                for step_idx, gold in enumerate(trace.tool_names[: args.max_steps]):
                    if step_idx == 0:
                        # The tree recommends from START, but first-action
                        # prediction is also noisy. Keep it in the comparison.
                        pass
                    candidates = _candidate_actions(
                        trace,
                        distractor_pool=distractor_pool,
                        distractor_count=args.distractors,
                        seed=args.seed + episode_idx * 1000 + step_idx,
                    )
                    recommendation = _recommend_for_condition(
                        condition=condition,
                        trace=trace,
                        prefix=prefix,
                        candidates=candidates,
                        tree_model=model,
                    )
                    selected, raw, latency = _ask_gemini(
                        api_key=api_key,
                        model=args.model,
                        trace=trace,
                        condition=condition,
                        prefix=prefix,
                        candidates=candidates,
                        recommendation=recommendation.tool if recommendation else None,
                        recommendation_confidence=recommendation.confidence if recommendation else 0.0,
                        min_interval_s=args.gemini_min_interval_s,
                    )
                    matched = selected == gold
                    hallucinated = selected not in set(candidates)
                    row = ReplayStep(
                        task_id=trace.task_id,
                        condition=condition,
                        step=step_idx,
                        prefix=list(prefix),
                        gold_action=gold,
                        recommended_action=recommendation.tool if recommendation else None,
                        recommendation_confidence=recommendation.confidence if recommendation else 0.0,
                        selected_action=selected,
                        matched_gold=matched,
                        hallucinated=hallucinated,
                        raw_response=raw[:1000],
                        latency_ms=latency,
                    )
                    steps.append(row)
                    handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    handle.flush()
                    if not matched:
                        break
                    prefix.append(gold)

    report = _report(steps)
    report["selection"] = {
        "require_recommendation": args.require_recommendation,
        "skipped_no_recommendation": skipped_no_recommendation,
        "heldout_episodes": len(heldout),
        "conditions": conditions,
        "distractors": args.distractors,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **report}, indent=2))


@dataclass
class Recommendation:
    tool: str
    confidence: float


def _candidate_actions(
    trace: Trace,
    *,
    distractor_pool: list[str],
    distractor_count: int,
    seed: int,
) -> list[str]:
    actions = list(dict.fromkeys(trace.tool_names))
    gold_set = set(actions)
    rng = random.Random(seed)
    candidates = list(actions)
    for action in distractor_pool:
        if action not in gold_set and action not in candidates:
            candidates.append(action)
        if len(candidates) >= len(actions) + distractor_count:
            break
    rng.shuffle(candidates)
    return candidates[: max(len(actions), len(actions) + distractor_count)]


def _recommend_for_condition(
    *,
    condition: str,
    trace: Trace,
    prefix: list[str],
    candidates: list[str],
    tree_model: BucketedActionTreeAdapter,
) -> Recommendation | None:
    if condition == "tree_suggested":
        rec = tree_model.recommend_next(trace.task_text, prefix, set(candidates))
        return Recommendation(rec.tool, rec.confidence) if rec else None
    if condition == "bm25_suggested":
        return _bm25_recommend(trace.task_text, prefix, candidates)
    return None


def _bm25_recommend(task_text: str, prefix: list[str], candidates: list[str]) -> Recommendation | None:
    if not candidates:
        return None
    query_tokens = _tokens(task_text + " " + " ".join(_action_text(p) for p in prefix[-2:]))
    docs = [_tokens(_action_text(action)) for action in candidates]
    if not query_tokens:
        return Recommendation(candidates[0], 0.0)
    n_docs = len(docs)
    avgdl = sum(len(doc) for doc in docs) / max(n_docs, 1)
    df = Counter(token for doc in docs for token in set(doc))
    q_counts = Counter(query_tokens)
    scores: list[float] = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc) or 1
        score = 0.0
        for token, qtf in q_counts.items():
            if token not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5))
            denom = tf[token] + 1.5 * (1.0 - 0.75 + 0.75 * dl / max(avgdl, 1e-9))
            score += idf * (tf[token] * 2.5 / denom) * qtf
        scores.append(score)
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    best = scores[best_idx]
    total = sum(max(score, 0.0) for score in scores)
    confidence = best / total if total > 0 else 0.0
    return Recommendation(candidates[best_idx], confidence)


def _action_text(action: str) -> str:
    action = re.sub(r"__[0-9a-f]{10}$", "", action)
    return action.replace("_", " ").replace("::", " ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ask_gemini(
    *,
    api_key: str,
    model: str,
    trace: Trace,
    condition: str,
    prefix: list[str],
    candidates: list[str],
    recommendation: str | None,
    recommendation_confidence: float,
    min_interval_s: float,
) -> tuple[str | None, str, float]:
    import requests

    rec_block = ""
    if condition in {"tree_suggested", "bm25_suggested"} and recommendation:
        label = "Tree" if condition == "tree_suggested" else "BM25"
        rec_block = (
            f"\n{label} recommendation: {recommendation}\n"
            f"Recommendation confidence: {recommendation_confidence:.3f}\n"
            "Prefer the recommendation when it is relevant, but choose another candidate if clearly better.\n"
        )
    prompt = f"""You are choosing the next tool/action for an agent.
Return JSON only with this shape: {{"action": "...", "reason": "short"}}.
Choose exactly one action from the candidate list.

Task:
{trace.task_text}

Current prefix:
{json.dumps(prefix)}

Candidate actions:
{json.dumps(candidates, indent=2)}
{rec_block}
"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = None
    start = time.perf_counter()
    for attempt in range(5):
        _throttle_gemini(min_interval_s)
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
            json=payload,
            timeout=60,
        )
        if response.status_code != 429:
            break
        time.sleep(_retry_delay(response, attempt))
    latency = (time.perf_counter() - start) * 1000
    assert response is not None
    response.raise_for_status()
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0].get("text", "")
    return _parse_action(text), text, latency


def _throttle_gemini(min_interval_s: float = 6.0) -> None:
    global _LAST_GEMINI_REQUEST_AT
    elapsed = time.perf_counter() - _LAST_GEMINI_REQUEST_AT
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)
    _LAST_GEMINI_REQUEST_AT = time.perf_counter()


def _retry_delay(response: Any, attempt: int) -> float:
    try:
        payload = response.json()
        details = payload.get("error", {}).get("details", [])
        for detail in details:
            retry = detail.get("retryDelay")
            if isinstance(retry, str) and retry.endswith("s"):
                return float(retry[:-1]) + 1.0
    except Exception:
        pass
    return min(60.0, 8.0 * (attempt + 1))


def _parse_action(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            action = obj.get("action")
            return str(action).strip() if action else None
        except json.JSONDecodeError:
            pass
    stripped = text.strip()
    return stripped or None


def _report(steps: list[ReplayStep]) -> dict[str, Any]:
    by_condition: dict[str, list[ReplayStep]] = {}
    for step in steps:
        by_condition.setdefault(step.condition, []).append(step)

    result: dict[str, Any] = {"steps": len(steps), "conditions": {}}
    for condition, rows in by_condition.items():
        result["conditions"][condition] = {
            "steps": len(rows),
            "next_action_accuracy": sum(row.matched_gold for row in rows) / max(len(rows), 1),
            "hallucination_rate": sum(row.hallucinated for row in rows) / max(len(rows), 1),
            "avg_latency_ms": sum(row.latency_ms for row in rows) / max(len(rows), 1),
            "recommendation_available_rate": sum(row.recommended_action is not None for row in rows) / max(len(rows), 1),
            "recommendation_acceptance_rate": sum(
                row.selected_action == row.recommended_action
                for row in rows
                if row.recommended_action is not None
            )
            / max(sum(row.recommended_action is not None for row in rows), 1),
        }
    return result


if __name__ == "__main__":
    main()
