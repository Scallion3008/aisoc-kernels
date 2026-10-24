#!/usr/bin/env bash
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
shopt -s nullglob

BENCHES=(
  "0_baseline"
  "1_tb_swizzle"
  "2_pipelined"
  "3_tma"
  "4_autotuning"
)

for bench in "${BENCHES[@]}"; do
  echo "Running bench $bench"
  uv run "kernels/$bench.py"
done
