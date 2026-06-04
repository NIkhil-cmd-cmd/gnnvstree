from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

ALLOWED_GEMINI_MODELS = frozenset(
    {
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
    }
)

DEFAULT_FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]

FORBIDDEN_ENV = ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "VERTEXAI_PROJECT")


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
    gemini_latency_ms: float
    retrieval_latency_ms: float
    total_tokens: int
    cost_usd: float
    model: str
    raw_response: str = ""


class GeminiBudgetGuard:
    def __init__(self, daily_budget: int):
        self.daily_budget = daily_budget
        self.count = 0

    def check(self) -> None:
        if self.count >= self.daily_budget:
            raise RuntimeError(f"Gemini daily budget exceeded ({self.daily_budget})")
        self.count += 1


def validate_gemini_config(model: str) -> str:
    for var in FORBIDDEN_ENV:
        if os.environ.get(var):
            raise RuntimeError(
                f"{var} is set; use AI Studio API key only to avoid billed Vertex usage."
            )
    if model not in ALLOWED_GEMINI_MODELS:
        raise ValueError(
            f"Model {model!r} not allowed. Use one of: {sorted(ALLOWED_GEMINI_MODELS)}"
        )
    if "pro" in model.lower() and "flash" not in model.lower():
        raise ValueError("Gemini Pro models are not allowed on free tier.")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY from https://aistudio.google.com/ (free tier).")
    return api_key


def build_prompt(query: str, ranked_tools: list[tuple[str, float]]) -> str:
    tools_block = "\n".join(f"- {name} (score={score:.3f})" for name, score in ranked_tools)
    return (
        "You are a tool-selection agent. Given the user query and ranked candidate tools, "
        "respond with a JSON object only: {\"tools\": [\"category::tool::api\", ...]} "
        "ordered by execution sequence. Only pick tools from the list.\n\n"
        f"User query: {query}\n\nCandidate tools:\n{tools_block}\n"
    )


def parse_tool_response(text: str, allowed: set[str]) -> tuple[list[str], float]:
    hallucinations = 0
    selected: list[str] = []
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            selected = obj.get("tools") or obj.get("tool_sequence") or []
        else:
            for part in re.findall(r"[\w:]+\:\:[\w:]+(?:\:\:[\w:]+)?", text):
                selected.append(part)
    except json.JSONDecodeError:
        selected = re.findall(r"[\w]+(?:\:\:[\w]+){2}", text)
    clean = []
    for t in selected:
        t = str(t).strip()
        if t in allowed:
            clean.append(t)
        elif t:
            hallucinations += 1
    rate = hallucinations / max(len(selected), 1)
    return clean, rate


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "429" in msg
        or "resourceexhausted" in name
        or "quota" in msg
        or "rate" in msg
        or "limit: 0" in msg
    )


def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    msg = str(exc)
    m = re.search(r"retry in ([\d.]+)s", msg, re.I)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"seconds:\s*([\d.]+)", msg)
    if m:
        return float(m.group(1)) + 1.0
    return min(60.0, 2 ** attempt * 2)


def _token_count(response) -> int:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return 0
    return (
        getattr(meta, "total_token_count", None)
        or getattr(meta, "total_tokens", None)
        or 0
    )


class GeminiToolAgent:
    def __init__(
        self,
        model: str,
        daily_budget: int = 100,
        fallback_models: list[str] | None = None,
        min_request_interval_s: float = 4.0,
    ):
        api_key = validate_gemini_config(model)
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        fallbacks = fallback_models or DEFAULT_FALLBACK_MODELS
        seen = {model}
        self.models_to_try = [model] + [m for m in fallbacks if m not in seen and m in ALLOWED_GEMINI_MODELS]
        self.guard = GeminiBudgetGuard(daily_budget)
        self.min_request_interval_s = min_request_interval_s
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.perf_counter() - self._last_request_at
        if elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)

    def generate(self, prompt: str) -> tuple[str, float, int, str]:
        """Returns (text, latency_ms, total_tokens, model_used)."""
        last_err: BaseException | None = None
        for model_name in self.models_to_try:
            for attempt in range(5):
                self._throttle()
                t1 = time.perf_counter()
                try:
                    resp = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    self._last_request_at = time.perf_counter()
                    text = resp.text or ""
                    return text, (time.perf_counter() - t1) * 1000, _token_count(resp), model_name
                except Exception as e:
                    last_err = e
                    if _is_rate_limit_error(e) and attempt < 4:
                        time.sleep(_retry_delay_seconds(e, attempt))
                        continue
                    if _is_rate_limit_error(e):
                        break
                    raise
        assert last_err is not None
        raise RuntimeError(
            f"Gemini free-tier quota exhausted for models {self.models_to_try}. "
            f"Check https://ai.dev/rate-limit and retry later. Last error: {last_err}"
        ) from last_err

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
        self.guard.check()
        allowed = {t for t, _ in ranked_tools}
        prompt = build_prompt(query, ranked_tools)
        t0 = time.perf_counter()
        text, latency_ms, total_tokens, model_used = self.generate(prompt)
        wall = time.perf_counter() - t0
        selected, hall_rate = parse_tool_response(text, allowed)
        rel = set(relevant_keys)
        path_set = path or relevant_keys
        sel_acc = 1.0 if selected and selected[0] == (path_set[0] if path_set else "") else 0.0
        multi_recall = len(set(selected) & rel) / len(rel) if rel else 0.0
        path_success = 1.0 if selected == list(path_set)[: len(selected)] and len(selected) >= len(path_set) else (
            1.0 if set(path_set) <= set(selected) else 0.0
        )
        unnecessary = len([t for t in selected if t not in set(path_set)])
        episode_success = 1.0 if set(path_set) <= set(selected) else 0.0
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
            steps_to_complete=max(len(selected), 1),
            episode_success=episode_success,
            wall_clock_s=wall,
            gemini_latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            total_tokens=total_tokens,
            cost_usd=0.0,
            model=model_used,
            raw_response=text[:500],
        )
