#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source gnn/.venv/bin/activate
CFG=gnn/configs/cpu_g2_g3.yaml

python -m gnn.src.data.download_toolbench --config "$CFG" --skip-gdrive
python -m gnn.src.data.build_workflow_graphs --config "$CFG"
python -m gnn.src.data.bucket_kmeans --config "$CFG"
python -m gnn.src.train.train_gnn --config "$CFG"
python -m gnn.src.eval.run_benchmark --config "$CFG" --skip-ir "$@"
