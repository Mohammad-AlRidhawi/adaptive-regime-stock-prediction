#!/usr/bin/env bash
# Reproduce all tables and figures in the manuscript.
# Total wall-clock time on an NVIDIA A100 40 GB GPU: ~22 hours.

set -euo pipefail

CONFIG="configs/default.yaml"
RUN_DIR="runs/paper"

mkdir -p "$RUN_DIR"

echo "==> Training proposed framework (AE-NodeFormer + SAC)"
python scripts/train.py --config "$CONFIG"

echo "==> Evaluating at 1-day horizon"
python scripts/evaluate.py --config "$CONFIG" --checkpoint "$RUN_DIR/checkpoints/final.pt" --horizon 1

echo "==> Evaluating at 5-day horizon"
python scripts/evaluate.py --config "$CONFIG" --checkpoint "$RUN_DIR/checkpoints/final.pt" --horizon 5

echo "==> Evaluating at 20-day horizon"
python scripts/evaluate.py --config "$CONFIG" --checkpoint "$RUN_DIR/checkpoints/final.pt" --horizon 20

echo "==> Running trading backtest (top-K = 5, 10 bp transaction cost)"
python scripts/backtest.py --config "$CONFIG" --checkpoint "$RUN_DIR/checkpoints/final.pt" --top-k 5 --cost-bps 10

echo "==> Done. Artifacts written to $RUN_DIR"
