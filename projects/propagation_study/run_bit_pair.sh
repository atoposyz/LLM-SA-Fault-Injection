#!/bin/bash
#SBATCH --job-name=prop_study
#SBATCH --partition=gpu
#SBATCH --nodelist=g2c15
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=/workplace/home/mayongzhe/faultinject/projects/propagation_study/slurm_logs/prop_study_%j.out
#SBATCH --error=/workplace/home/mayongzhe/faultinject/projects/propagation_study/slurm_logs/prop_study_%j.err

set -euo pipefail

PROJECT_DIR="/workplace/home/mayongzhe/faultinject/projects/propagation_study"
SCRIPT="${PROJECT_DIR}/run_propagation_study.py"
LOG_DIR="${PROJECT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

BIT_A="${1}"
BIT_B="${2}"
STUCK="${3:-1}"  # default to stuck_1

P_LIST="0 16 32 48 64 80 96 112 128 144 160 192 256"
COMMON_ARGS="-n 200 --filter-layer 0 --filter-stuck ${STUCK} --filter-p ${P_LIST} --kernel-groups k,v"

echo "=========================================="
echo "Starting propagation study: bit=${BIT_A} & bit=${BIT_B} stuck=${STUCK}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# bit arg1 on GPU 0
echo "[$(date)] Launching bit=${BIT_A} stuck=${STUCK} on GPU 0..."
uv run python "${SCRIPT}" \
    -o "${PROJECT_DIR}/result/bit${BIT_A}_sa${STUCK}_n200" \
    -g 0 \
    ${COMMON_ARGS} \
    --filter-bit "${BIT_A}" \
    > "${LOG_DIR}/prop_study_bit${BIT_A}_sa${STUCK}_${SLURM_JOB_ID}.log" 2>&1 &
PID_1=$!

# bit arg2 on GPU 1
echo "[$(date)] Launching bit=${BIT_B} stuck=${STUCK} on GPU 1..."
uv run python "${SCRIPT}" \
    -o "${PROJECT_DIR}/result/bit${BIT_B}_sa${STUCK}_n200" \
    -g 1 \
    ${COMMON_ARGS} \
    --filter-bit "${BIT_B}" \
    > "${LOG_DIR}/prop_study_bit${BIT_B}_sa${STUCK}_${SLURM_JOB_ID}.log" 2>&1 &
PID_2=$!

echo "PID bit=${BIT_A}: ${PID_1}"
echo "PID bit=${BIT_B}: ${PID_2}"

wait ${PID_1} ${PID_2}

echo "=========================================="
echo "[$(date)] Both bits completed."
echo "Outputs:"
echo "  bit${BIT_A}: ${PROJECT_DIR}/result/bit${BIT_A}_sa${STUCK}_n200/"
echo "  bit${BIT_B}: ${PROJECT_DIR}/result/bit${BIT_B}_sa${STUCK}_n200/"
echo "=========================================="
