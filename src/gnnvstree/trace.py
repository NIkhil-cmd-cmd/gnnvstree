"""Canonical trace schema and normalization helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One ordered tool/API/action call in a trace."""

    name: str
    raw_name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Trace:
    """Model-neutral tool trace used by both tree and graph frameworks."""

    trace_id: str
    source: str
    task_id: str
    split: str
    success: bool
    tool_calls: tuple[ToolCall, ...]
    semantic_steps: tuple[str, ...] = ()
    task_text: str = ""
    candidate_tools: tuple[str, ...] = ()
    ground_truth_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(call.name for call in self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return data


def normalize_name(value: str) -> str:
    """Normalize tool/API names while preserving enough identity for metrics."""

    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def stable_trace_id(*parts: Any) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_tool_call(name: str, raw_name: str | None = None, input: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(name=normalize_name(name), raw_name=raw_name or name, input=input or {})


def make_trace(
    *,
    source: str,
    task_id: str,
    split: str,
    success: bool,
    tool_names: list[str] | tuple[str, ...],
    task_text: str = "",
    candidate_tools: list[str] | tuple[str, ...] = (),
    ground_truth_tools: list[str] | tuple[str, ...] = (),
    semantic_steps: list[str] | tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Trace:
    normalized_tools = tuple(make_tool_call(name) for name in tool_names)
    return Trace(
        trace_id=stable_trace_id(source, task_id, split, "|".join(call.name for call in normalized_tools)),
        source=source,
        task_id=str(task_id),
        split=split,
        success=success,
        tool_calls=normalized_tools,
        semantic_steps=tuple(normalize_name(step) for step in semantic_steps),
        task_text=task_text,
        candidate_tools=tuple(normalize_name(tool) for tool in candidate_tools),
        ground_truth_tools=tuple(normalize_name(tool) for tool in ground_truth_tools),
        metadata=metadata or {},
    )

