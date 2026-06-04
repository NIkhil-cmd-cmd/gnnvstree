"""Validate Gemini free-tier guards without making API calls."""

from __future__ import annotations

import os
import sys

from gnn.src.agent.gemini_agent import ALLOWED_GEMINI_MODELS, validate_gemini_config


def main() -> None:
    for model in ALLOWED_GEMINI_MODELS:
        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            os.environ["GEMINI_API_KEY"] = "test-key-not-used"
        validate_gemini_config(model)
        print(f"OK allowlist: {model}")
    try:
        validate_gemini_config("gemini-1.5-pro")
        print("FAIL should reject pro")
        sys.exit(1)
    except ValueError:
        print("OK rejects gemini-1.5-pro")
    try:
        from google import genai  # noqa: F401
        print("OK google.genai import")
    except ImportError:
        print("FAIL: pip install google-genai")
        sys.exit(1)
    print("Gemini guard checks passed.")


if __name__ == "__main__":
    main()
