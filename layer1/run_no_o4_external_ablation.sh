#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jjhong/gly}"
PYTHON_BIN="${PYTHON_BIN:-/home/jjhong/.conda/envs/one/bin/python}"
GPU_ID="${1:?usage: run_no_o4_external_ablation.sh GPU_ID SEED [SEED ...]}"
shift

if [[ "$#" -eq 0 ]]; then
  echo "at least one seed is required" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
output_dir="results/hard_feasibility_no_o4_external_role"
checkpoint_dir="checkpoints/hard_feasibility_no_o4_external_role"
log_dir="logs/hard_feasibility_no_o4_external_role"
mkdir -p "$output_dir" "$checkpoint_dir" "$log_dir"

for seed in "$@"; do
  log_path="$log_dir/seed${seed}.log"
  echo "starting seed=${seed} gpu=${GPU_ID} at $(date --iso-8601=seconds)" | tee "$log_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m layer1.train_hard_feasibility \
    --seed "$seed" \
    --device cuda \
    --output-dir "$output_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    2>&1 | tee -a "$log_path"
  echo "completed seed=${seed} gpu=${GPU_ID} at $(date --iso-8601=seconds)" | tee -a "$log_path"
done
