"""Metrics for training and testing tool-logic models."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Protocol

from gnnvstree.trace import Trace


class LogicModel(Protocol):
    def fit(self, traces: list[Trace]) -> dict[str, Any]: ...
    def predict_next(self, prefix: list[str] | tuple[str, ...], context: dict[str, Any] | None = None) -> Any: ...
    def metrics(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EvaluationReport:
    dataset: dict[str, Any]
    training: dict[str, Any]
    testing: dict[str, Any]
    sequence: dict[str, Any]
    structure: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataset_metrics(traces: list[Trace]) -> dict[str, Any]:
    lengths = [len(trace.tool_calls) for trace in traces]
    successful = [trace for trace in traces if trace.success]
    tools = Counter(tool for trace in traces for tool in trace.tool_names)
    transitions = Counter(
        (names[idx], names[idx + 1])
        for trace in traces
        for names in [trace.tool_names]
        for idx in range(len(names) - 1)
    )
    by_source = Counter(trace.source for trace in traces)
    by_split = Counter(trace.split for trace in traces)
    duplicate_keys = Counter((trace.task_id, trace.tool_names) for trace in traces)
    return {
        "trace_count": len(traces),
        "successful_trace_count": len(successful),
        "failed_trace_count": len(traces) - len(successful),
        "success_rate": len(successful) / len(traces) if traces else 0.0,
        "unique_task_count": len({trace.task_id for trace in traces}),
        "unique_tool_count": len(tools),
        "transition_count": sum(transitions.values()),
        "unique_transition_count": len(transitions),
        "avg_trace_length": mean(lengths) if lengths else 0.0,
        "median_trace_length": median(lengths) if lengths else 0.0,
        "p90_trace_length": _percentile(lengths, 0.90),
        "empty_trace_rate": sum(1 for length in lengths if length == 0) / len(lengths) if lengths else 0.0,
        "single_step_trace_rate": sum(1 for length in lengths if length == 1) / len(lengths) if lengths else 0.0,
        "duplicate_trace_rate": sum(count - 1 for count in duplicate_keys.values() if count > 1) / len(traces) if traces else 0.0,
        "by_source": dict(by_source),
        "by_split": dict(by_split),
        "top_tools": tools.most_common(25),
        "top_transitions": [
            {"source": src, "target": dst, "count": count}
            for (src, dst), count in transitions.most_common(25)
        ],
    }


def evaluate_model(model: LogicModel, train_traces: list[Trace], test_traces: list[Trace]) -> EvaluationReport:
    train_start = time.perf_counter()
    training = model.fit(train_traces)
    fit_seconds = time.perf_counter() - train_start
    training = dict(training) | {"fit_seconds_measured": fit_seconds}

    prediction_rows = []
    total = correct_top1 = correct_top3 = correct_top5 = covered = 0
    reciprocal_ranks = []
    confidence_values = []
    prediction_latencies = []
    per_tool_total: Counter[str] = Counter()
    per_tool_correct: Counter[str] = Counter()
    unseen_transition_total = unseen_transition_correct = 0
    seen_transitions = {
        (names[idx], names[idx + 1])
        for trace in train_traces
        for names in [trace.tool_names]
        for idx in range(len(names) - 1)
    }

    for trace in test_traces:
        names = trace.tool_names
        for idx in range(len(names) - 1):
            prefix = names[: idx + 1]
            expected = names[idx + 1]
            total += 1
            per_tool_total[expected] += 1
            start = time.perf_counter()
            prediction = model.predict_next(prefix, {"trace": trace})
            prediction_latencies.append(time.perf_counter() - start)
            if prediction is None:
                prediction_rows.append(_prediction_row(trace, prefix, expected, None, []))
                continue

            covered += 1
            ranked = [(prediction.tool, prediction.confidence), *getattr(prediction, "alternatives", [])]
            ranked_tools = [tool for tool, _ in ranked]
            confidence_values.append(float(prediction.confidence))
            if expected in ranked_tools:
                reciprocal_ranks.append(1 / (ranked_tools.index(expected) + 1))
            else:
                reciprocal_ranks.append(0.0)
            if ranked_tools[:1] == [expected]:
                correct_top1 += 1
                per_tool_correct[expected] += 1
            if expected in ranked_tools[:3]:
                correct_top3 += 1
            if expected in ranked_tools[:5]:
                correct_top5 += 1
            if (prefix[-1], expected) not in seen_transitions:
                unseen_transition_total += 1
                unseen_transition_correct += int(ranked_tools[:1] == [expected])
            prediction_rows.append(_prediction_row(trace, prefix, expected, prediction.tool, ranked))

    testing = {
        "examples": total,
        "coverage_rate": covered / total if total else 0.0,
        "abstention_rate": 1 - (covered / total) if total else 0.0,
        "top1_accuracy": correct_top1 / total if total else 0.0,
        "top3_accuracy": correct_top3 / total if total else 0.0,
        "top5_accuracy": correct_top5 / total if total else 0.0,
        "accuracy_when_predicted": correct_top1 / covered if covered else 0.0,
        "mean_reciprocal_rank": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "confidence_mean": mean(confidence_values) if confidence_values else 0.0,
        "confidence_median": median(confidence_values) if confidence_values else 0.0,
        "confidence_p90": _percentile(confidence_values, 0.90),
        "prediction_latency_ms_p50": _percentile(prediction_latencies, 0.50) * 1000,
        "prediction_latency_ms_p90": _percentile(prediction_latencies, 0.90) * 1000,
        "unseen_transition_examples": unseen_transition_total,
        "unseen_transition_top1_accuracy": unseen_transition_correct / unseen_transition_total if unseen_transition_total else 0.0,
        "per_tool_accuracy": {
            tool: per_tool_correct[tool] / count for tool, count in per_tool_total.items()
        },
        "sample_predictions": prediction_rows[:50],
    }

    return EvaluationReport(
        dataset={
            "train": dataset_metrics(train_traces),
            "test": dataset_metrics(test_traces),
            "overlap": overlap_metrics(train_traces, test_traces),
        },
        training=training,
        testing=testing,
        sequence=sequence_metrics(model, test_traces),
        structure=model.metrics(),
    )


def overlap_metrics(train_traces: list[Trace], test_traces: list[Trace]) -> dict[str, Any]:
    train_tools = {tool for trace in train_traces for tool in trace.tool_names}
    test_tools = {tool for trace in test_traces for tool in trace.tool_names}
    train_transitions = {
        (names[idx], names[idx + 1])
        for trace in train_traces
        for names in [trace.tool_names]
        for idx in range(len(names) - 1)
    }
    test_transitions = {
        (names[idx], names[idx + 1])
        for trace in test_traces
        for names in [trace.tool_names]
        for idx in range(len(names) - 1)
    }
    return {
        "tool_overlap_rate": len(train_tools & test_tools) / len(test_tools) if test_tools else 0.0,
        "held_out_tool_count": len(test_tools - train_tools),
        "transition_overlap_rate": len(train_transitions & test_transitions) / len(test_transitions) if test_transitions else 0.0,
        "held_out_transition_count": len(test_transitions - train_transitions),
    }


def sequence_metrics(model: LogicModel, traces: list[Trace]) -> dict[str, Any]:
    exact = 0
    distances = []
    divergence_steps = []
    invalid_repetition = 0
    generated_count = 0
    for trace in traces:
        names = trace.tool_names
        if len(names) < 2:
            continue
        generated = [names[0]]
        for _ in range(len(names) - 1):
            prediction = model.predict_next(tuple(generated), {"trace": trace})
            if prediction is None:
                break
            generated.append(prediction.tool)
        generated_tuple = tuple(generated)
        generated_count += 1
        exact += int(generated_tuple == names)
        distances.append(edit_distance(generated_tuple, names) / max(len(names), 1))
        divergence_steps.append(_divergence_step(generated_tuple, names))
        invalid_repetition += int(_has_repetition_loop(generated_tuple))
    return {
        "evaluated_sequences": generated_count,
        "full_trace_exact_match_rate": exact / generated_count if generated_count else 0.0,
        "avg_normalized_edit_distance": mean(distances) if distances else 0.0,
        "median_divergence_step": median(divergence_steps) if divergence_steps else 0.0,
        "loop_repetition_rate": invalid_repetition / generated_count if generated_count else 0.0,
    }


def edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _prediction_row(trace: Trace, prefix: tuple[str, ...], expected: str, predicted: str | None, ranked: list[tuple[str, float]]) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "prefix": list(prefix),
        "expected": expected,
        "predicted": predicted,
        "ranked": ranked[:5],
    }


def _percentile(values: list[float] | list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _divergence_step(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    for idx, expected in enumerate(right):
        if idx >= len(left) or left[idx] != expected:
            return idx
    return len(right)


def _has_repetition_loop(sequence: tuple[str, ...]) -> bool:
    if len(sequence) < 4:
        return False
    for idx in range(len(sequence) - 3):
        if sequence[idx] == sequence[idx + 2] and sequence[idx + 1] == sequence[idx + 3]:
            return True
    return False

