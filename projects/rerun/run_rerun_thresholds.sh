#!/bin/bash
#SBATCH --job-name=rerun_thr
#SBATCH --partition=gpu
#SBATCH --nodelist=g2c14
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --output=/workplace/home/mayongzhe/faultinject/projects/rerun/slurm_logs/rerun_thr_%j.out
#SBATCH --error=/workplace/home/mayongzhe/faultinject/projects/rerun/slurm_logs/rerun_thr_%j.err

set -euo pipefail

PROJECT_DIR="/workplace/home/mayongzhe/faultinject"
SCRIPT="${PROJECT_DIR}/projects/rerun/SARerun.py"

# Common args (derived from reference: ReRun_single_bit_kernel-v_layer-35_reg-input_df-WS_cfg-input_bitflip_13_pe_random_sr<=1.2_cons3_detL33affect0.jsonl)
COMMON_ARGS=(
    --layerList 35
    --layerType v
    --reg input
    --dataflow WS
    --injectConfig input_bitflip_13
    --pe -1 -1
    --sr_cmp "<="
    --sr_count_mode consecutive
    --sr_trigger_k 3
    --detectLayer 33
    --interval 50
    --max_tokens 5000
)

THRESHOLDS=(1.1 1.15 1.25 1.3)

run_job() {
    local thr=$1 gpu=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting threshold=${thr} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} uv run python "${SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --sr_threshold "${thr}" \
        > "${PROJECT_DIR}/projects/rerun/slurm_logs/rerun_thr_${thr}_gpu${gpu}.log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished threshold=${thr} on GPU ${gpu}"
}

echo "=== ReRun threshold sweep ==="
echo "Thresholds: ${THRESHOLDS[*]}"
echo "Started at $(date)"

cd "${PROJECT_DIR}"

# Batch 1: thresholds 1.1 (GPU 0) and 1.15 (GPU 1) in parallel
echo ""
echo "=== Batch 1: thr=1.1 (GPU0), thr=1.15 (GPU1) ==="
run_job 1.1 0 &
PID1=$!
run_job 1.15 1 &
PID2=$!
wait $PID1 $PID2
echo "Batch 1 done at $(date)"

# Batch 2: thresholds 1.25 (GPU 0) and 1.3 (GPU 1) in parallel
echo ""
echo "=== Batch 2: thr=1.25 (GPU0), thr=1.3 (GPU1) ==="
run_job 1.25 0 &
PID3=$!
run_job 1.3 1 &
PID4=$!
wait $PID3 $PID4
echo "Batch 2 done at $(date)"

echo ""
echo "=== All thresholds done at $(date) ==="
