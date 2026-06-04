# Shared GNN ToolBench Validation

Validates the shared-bucket GNN tool retriever on **ToolBench G2/G3 multi-tool workflows** using real DFSDT paths and HF benchmark labels.

## Setup

```bash
git checkout gnn
pip install -r gnn/requirements.txt
```

## Pipeline

```bash
# 1. Download/cache ToolBench data (HF required; official zip optional)
python -m gnn.src.data.download_toolbench --config gnn/configs/cpu_g2_g3.yaml

# 2. Build SQLite graphs from real tool paths
python -m gnn.src.data.build_workflow_graphs --config gnn/configs/cpu_g2_g3.yaml

# 3. K-means prompt routing + cluster→bucket map
python -m gnn.src.data.bucket_kmeans --config gnn/configs/cpu_g2_g3.yaml

# 4. Train shared GNN (CPU)
python -m gnn.src.train.train_gnn --config gnn/configs/cpu_g2_g3.yaml

# 5. Benchmark retrieval + real Gemini agent (required for agent section)
export GEMINI_API_KEY=your_ai_studio_key   # https://aistudio.google.com/ — do not paste in chat/logs
python -m gnn.src.eval.run_benchmark --config gnn/configs/cpu_g2_g3.yaml
```

**Agent eval uses real Gemini API calls** unless you explicitly pass `--dry-run-agent` (offline only, not real LLM).

## Gemini (no charges)

- Use **AI Studio** API key only (`GEMINI_API_KEY`).
- Do **not** set `GOOGLE_CLOUD_PROJECT` / Vertex env vars.
- Uses **`google-genai`** SDK (not deprecated `google-generativeai`).
- Default model: `gemini-2.5-flash-lite` with fallbacks in config (see `cpu_g2_g3.yaml`).
- **429 / quota:** waits using API `retry_delay`, tries fallback Flash models, saves partial results if exhausted.
- If you see `limit: 0` for `gemini-2.0-flash`, your key has no free quota on that model — switch config to `gemini-2.5-flash-lite` or check https://ai.dev/rate-limit
- Daily call budget enforced in code (default 100); eval samples default 20 to stay within free RPM.

## Outputs

- `gnn/artifacts/model.pt` — shared GNN checkpoint
- `gnn/artifacts/toolbench.db` — graphs and training samples
- `gnn/results/benchmark_report.json` — all metrics
- `gnn/results/REPORT.md` — summary
- `gnn/results/retrieval_summary.csv`

## Metrics

Retrieval: P@K, R@K, F1@K, Hit@K, MRR, MAP, NDCG@5/10, PathRecall@K, PathF1@K, OrderedPath@K, WorkflowCoverage, NextToolAcc@1, latency p50/p95/p99.

Agent: selection_accuracy, multi_tool_selection_recall, path_success, hallucination_rate, tools_retrieved/called, episode_success, wall_clock_s, token counts, cost_usd=0.
