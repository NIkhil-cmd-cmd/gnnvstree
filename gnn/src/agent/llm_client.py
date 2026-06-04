from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod


def _is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "429" in msg
        or "503" in msg
        or "unavailable" in msg
        or "rate" in msg
        or "quota" in msg
        or "overloaded" in msg
        or "resourceexhausted" in name
    )


def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    msg = str(exc)
    for pat in (r"retry in ([\d.]+)s", r"seconds:\s*([\d.]+)", r"retry_delay.*?(\d+)"):
        m = re.search(pat, msg, re.I)
        if m:
            return float(m.group(1)) + 1.0
    return min(90.0, 2 ** attempt * 3)


class LLMResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int, model: str, latency_ms: float):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.latency_ms = latency_ms

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMClient(ABC):
    def __init__(self, min_interval_s: float = 1.0):
        self.min_interval_s = min_interval_s
        self._last_at = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _throttle(self) -> None:
        elapsed = time.perf_counter() - self._last_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        ...

    def generate_with_retry(self, prompt: str, system: str | None = None, max_attempts: int = 6) -> LLMResponse:
        last: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                self._throttle()
                resp = self.generate(prompt, system=system)
                self._last_at = time.perf_counter()
                self.total_input_tokens += resp.input_tokens
                self.total_output_tokens += resp.output_tokens
                return resp
            except Exception as e:
                last = e
                if _is_transient_error(e) and attempt < max_attempts - 1:
                    time.sleep(_retry_delay_seconds(e, attempt))
                    continue
                raise
        raise last  # type: ignore[misc]


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str, min_interval_s: float = 1.0):
        super().__init__(min_interval_s)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY from https://console.anthropic.com/")
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        # Haiku 4.5 pricing for cost estimate
        self.input_price_per_m = 1.0
        self.output_price_per_m = 5.0

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = self.client.messages.create(**kwargs)
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
        usage = msg.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def estimate_cost_usd(self) -> float:
        return (
            self.total_input_tokens / 1e6 * self.input_price_per_m
            + self.total_output_tokens / 1e6 * self.output_price_per_m
        )


class GeminiClient(BaseLLMClient):
    ALLOWED = frozenset(
        {
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        }
    )

    def __init__(
        self,
        model: str,
        fallback_models: list[str] | None = None,
        min_interval_s: float = 4.0,
    ):
        super().__init__(min_interval_s)
        for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "VERTEXAI_PROJECT"):
            if os.environ.get(var):
                raise RuntimeError(f"{var} set — use AI Studio key only, not Vertex billing.")
        if model not in self.ALLOWED:
            raise ValueError(f"Model {model!r} not in allowlist: {sorted(self.ALLOWED)}")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY from https://aistudio.google.com/")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        fallbacks = fallback_models or ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
        self.models_to_try = [model] + [m for m in fallbacks if m != model and m in self.ALLOWED]

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        contents = f"{system}\n\n{prompt}" if system else prompt
        last_err: BaseException | None = None
        for model_name in self.models_to_try:
            try:
                t0 = time.perf_counter()
                resp = self.client.models.generate_content(model=model_name, contents=contents)
                meta = getattr(resp, "usage_metadata", None)
                inp = getattr(meta, "prompt_token_count", 0) or 0
                out = getattr(meta, "candidates_token_count", 0) or 0
                return LLMResponse(
                    text=resp.text or "",
                    input_tokens=int(inp),
                    output_tokens=int(out),
                    model=model_name,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            except Exception as e:
                last_err = e
                if _is_transient_error(e):
                    continue
                raise
        raise RuntimeError(f"Gemini failed for {self.models_to_try}: {last_err}") from last_err

    def estimate_cost_usd(self) -> float:
        return 0.0


def build_client(cfg: dict) -> BaseLLMClient:
    provider = (cfg.get("llm_provider") or "anthropic").lower()
    interval = float(cfg.get("llm_min_request_interval_s", 1.5))
    if provider == "anthropic":
        return AnthropicClient(cfg.get("anthropic_model", "claude-haiku-4-5-20251001"), interval)
    if provider == "gemini":
        return GeminiClient(
            cfg.get("gemini_model", "gemini-2.5-flash-lite"),
            cfg.get("gemini_fallback_models"),
            interval,
        )
    raise ValueError(f"Unknown llm_provider: {provider}")
