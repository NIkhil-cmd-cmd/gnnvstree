# Shared GNN ToolBench Validation

Listwise **query-conditioned GNN ranker** trained on the same task as the ToolBench benchmark: re-rank tools inside each query’s `api_list`. Training tasks are deduplicated and **held out from benchmark prompts** (no repeat tasks).

## Design (v2)

| Before | After |
|--------|--------|
| Train on bucket graphs, eval on edgeless `api_list` | Train & eval both rank `api_list` candidates |
| K-means bucket routing at eval | No routing — score candidates directly |
| Tiny shared GNN only | Listwise GNN + BM25/overlap features + kNN graph + optional BM25 residual blend |

## Setup

```bash
git checkout gnn
python3 -m venv gnn/.venv && source gnn/.venv/bin/activate
pip install -r gnn/requirements.txt
```

## Full pipeline

```bash
bash gnn/scripts/run_pipeline.sh
```

Steps:

1. `download_toolbench` — cache benchmark + train; write `benchmark_task_fingerprints.json`
2. `build_workflow_graphs` — bucket graphs (negatives for ranking lists); dedupe train vs benchmark
3. `build_ranking_dataset` — listwise training rows (prompt + candidates + labels)
4. `train_rank_gnn` — listwise ranker → `gnn/artifacts/model.pt`
5. `run_benchmark` — GNN vs BM25 (+ optional agent)

## Agent eval (optional)

```bash
export ANTHROPIC_API_KEY='...'
python -m gnn.src.eval.run_agent_only --config gnn/configs/cpu_g2_g3.yaml
```

## Config highlights (`gnn/configs/cpu_g2_g3.yaml`)

- `ranking_train_cap` — max listwise training queries (deduped)
- `max_candidates` / `num_distractors` — synthetic lists similar to benchmark size
- `bm25_blend` — fixed residual weight on normalized BM25 (not tuned on test)
- `knn_k` — edges among candidates for GCN

## Outputs

- `gnn/results/benchmark_report.json` — retrieval + agent
- `gnn/results/REPORT.md`
- `gnn/data/benchmark_task_fingerprints.json` — prompts excluded from training
