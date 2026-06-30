#!/bin/bash
#SBATCH --job-name=prop_b30
#SBATCH --partition=gpu
#SBATCH --nodelist=g2c15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --output=/workplace/home/mayongzhe/faultinject/projects/propagation_study/slurm_logs/prop_b30_%j.out
#SBATCH --error=/workplace/home/mayongzhe/faultinject/projects/propagation_study/slurm_logs/prop_b30_%j.err

set -euo pipefail

PROJECT_DIR="/workplace/home/mayongzhe/faultinject/projects/propagation_study"
SCRIPT="${PROJECT_DIR}/run_propagation_study.py"
LOG_DIR="${PROJECT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

BIT=30
STUCK=1
P_LIST="0 16 32 48 64 80 96 112 128 144 160 192 256"

echo "=========================================="
echo "Propagation study: bit=${BIT} stuck-at-${STUCK} on GPU 0 @ $(hostname)"
echo "Date: $(date)"
echo "=========================================="

uv run python "${SCRIPT}" \
    -o "${PROJECT_DIR}/result/bit${BIT}_sa${STUCK}_n200" \
    -g 0 \
    -n 200 \
    --filter-layer 0 \
    --filter-stuck "${STUCK}" \
    --filter-p ${P_LIST} \
    --kernel-groups k,v \
    --filter-bit "${BIT}" \
    2>&1 | tee "${LOG_DIR}/prop_b${BIT}_sa${STUCK}_${SLURM_JOB_ID}.log"

echo "=========================================="
echo "[$(date)] Done."
echo "Output: ${PROJECT_DIR}/result/bit${BIT}_sa${STUCK}_n200/"
echo "=========================================="
