"""Hugging Face ToolBench conversation trace extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from gnnvstree.trace import Trace, make_trace, normalize_name


@dataclass(frozen=True)
class ExtractionStats:
    rows_scanned: int = 0
    traces_extracted: int = 0
    skipped_no_conversation: int = 0
    skipped_too_short: int = 0
    action_count: int = 0
    parsed_input_count: int = 0

    def to_dict(self) -> dict[str, int | float]:
        extracted = max(self.traces_extracted, 1)
        actions = max(self.action_count, 1)
        return {
            "rows_scanned": self.rows_scanned,
            "traces_extracted": self.traces_extracted,
            "skipped_no_conversation": self.skipped_no_conversation,
            "skipped_too_short": self.skipped_too_short,
            "action_count": self.action_count,
            "parsed_input_count": self.parsed_input_count,
            "avg_actions_per_trace": self.action_count / extracted,
            "parsed_input_rate": self.parsed_input_count / actions,
        }


@dataclass(frozen=True)
class ExtractedAction:
    name: str
    raw_name: str
    input: Any
    raw_input: str
    parsed_input: bool

    @property
    def action_key(self) -> str:
        if self.raw_name == "Finish":
            return normalize_name("Finish")
        return normalize_name(f"{self.raw_name}__{_short_hash(_canonical_json(self.input))}")


def iter_hf_toolbench_traces(
    *,
    dataset_name: str = "tuandunghcmut/toolbench-v1",
    config: str = "default",
    split: str = "train",
    limit: int = 1000,
    min_actions: int = 2,
    include_action_inputs: bool = True,
) -> tuple[list[Trace], ExtractionStats]:
    """Stream ToolBench conversations from HF and extract action-level traces."""

    from datasets import load_dataset

    rows = load_dataset(dataset_name, config, split=split, streaming=True)
    traces: list[Trace] = []
    stats = ExtractionStats()

    for row in rows:
        if stats.rows_scanned >= limit:
            break
        stats = _replace(stats, rows_scanned=stats.rows_scanned + 1)
        trace = trace_from_hf_row(
            row,
            split=split,
            min_actions=min_actions,
            include_action_inputs=include_action_inputs,
        )
        if trace is None:
            if not row.get("conversations"):
                stats = _replace(stats, skipped_no_conversation=stats.skipped_no_conversation + 1)
            else:
                stats = _replace(stats, skipped_too_short=stats.skipped_too_short + 1)
            continue
        traces.append(trace)
        stats = _replace(
            stats,
            traces_extracted=stats.traces_extracted + 1,
            action_count=stats.action_count + len(trace.tool_calls),
            parsed_input_count=stats.parsed_input_count
            + int(trace.metadata.get("parsed_input_count", 0)),
        )

    return traces, stats


def trace_from_hf_row(
    row: dict[str, Any],
    *,
    split: str,
    min_actions: int,
    include_action_inputs: bool,
) -> Trace | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, dict):
        return None

    roles = conversations.get("from") or []
    values = conversations.get("value") or []
    if not roles or not values:
        return None

    task_text = _first_user_text(roles, values)
    actions: list[ExtractedAction] = []
    for role, value in zip(roles, values, strict=False):
        if role != "assistant":
            continue
        actions.extend(extract_actions_from_assistant(str(value)))

    if len(actions) < min_actions:
        return None

    if include_action_inputs:
        tool_names = [action.action_key for action in actions]
    else:
        tool_names = [action.name for action in actions]

    row_id = str(row.get("id") or _short_hash(task_text or "|".join(tool_names)))
    return make_trace(
        source="hf_toolbench",
        task_id=row_id,
        split=split,
        success=True,
        tool_names=tool_names,
        task_text=task_text,
        ground_truth_tools=tool_names,
        metadata={
            "hf_id": row_id,
            "action_count": len(actions),
            "parsed_input_count": sum(1 for action in actions if action.parsed_input),
            "include_action_inputs": include_action_inputs,
            "raw_action_names": [action.raw_name for action in actions],
        },
    )


def extract_actions_from_assistant(text: str) -> list[ExtractedAction]:
    lines = text.splitlines()
    actions: list[ExtractedAction] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("Action:"):
            index += 1
            continue
        raw_name = line.split(":", 1)[1].strip()
        index += 1
        raw_input_lines: list[str] = []
        if index < len(lines) and lines[index].strip().startswith("Action Input:"):
            first = lines[index].split(":", 1)[1].strip()
            if first:
                raw_input_lines.append(first)
            index += 1
            while index < len(lines):
                next_line = lines[index]
                stripped = next_line.strip()
                if stripped.startswith("Thought:") or stripped.startswith("Action:"):
                    break
                raw_input_lines.append(next_line)
                index += 1
        if not raw_name:
            continue
        raw_input = "\n".join(raw_input_lines).strip()
        parsed, parsed_input = _parse_input(raw_input)
        actions.append(
            ExtractedAction(
                name=normalize_name(raw_name),
                raw_name=raw_name,
                input=parsed_input,
                raw_input=raw_input,
                parsed_input=parsed,
            )
        )
    return actions


def _first_user_text(roles: Iterable[Any], values: Iterable[Any]) -> str:
    for role, value in zip(roles, values, strict=False):
        if role == "user":
            return str(value).replace("Begin!", "").strip()
    return ""


def _parse_input(raw_input: str) -> tuple[bool, Any]:
    if not raw_input:
        return False, {}
    try:
        return True, json.loads(raw_input)
    except json.JSONDecodeError:
        return False, raw_input


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _replace(stats: ExtractionStats, **updates: int) -> ExtractionStats:
    data = stats.__dict__.copy()
    data.update(updates)
    return ExtractionStats(**data)
