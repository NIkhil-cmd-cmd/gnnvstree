#!/usr/bin/env bash
# Re-run ONLY the live agent benchmark (retrieval must exist in results/ or will still run retrieval in full benchmark).
# Usage:
#   export ANTHROPIC_API_KEY='sk-ant-...'
#   bash gnn/scripts/run_agent_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source gnn/.venv/bin/activate
pip install -q anthropic google-genai 2>/dev/null || true
export LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
python -m gnn.src.eval.run_agent_only --config gnn/configs/cpu_g2_g3.yaml
