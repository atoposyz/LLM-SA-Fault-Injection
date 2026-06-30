#!/bin/bash
#SBATCH --job-name=saskip-upper
#SBATCH --output=/workplace/home/mayongzhe/faultinject/projects/rerun/slurm_logs/saskip_upper_%A_%a.out
#SBATCH --error=/workplace/home/mayongzhe/faultinject/projects/rerun/slurm_logs/saskip_upper_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodelist=g2c12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=40G
#SBATCH --array=0-5%2

# 6个上阈值，并行跑在 g2c12 的 2 个 GPU 上（%2 限制最多 2 个 job 同时运行）
UPPER_THRESHOLDS=(1.8 1.9 2.0 2.1 2.2 2.3)
THR=${UPPER_THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

# 固定参数（从文件名 ReRun_single_bit_kernel-v_layer-35_reg-input_df-WS_cfg-input_bitflip_13_pe_random_sr<=1.2_cons3_detL33affect0.jsonl 提取）
# 下阈值: <=1.2, consecutive, k=3, detect_layer=33
# 上阈值: >=${THR}, consecutive, k=3, detect_layer=0

cd /workplace/home/mayongzhe/faultinject

echo "[$(date)] Starting task ${SLURM_ARRAY_TASK_ID} with upper_threshold=${THR} on $(hostname) GPU=${CUDA_VISIBLE_DEVICES}"

uv run python projects/rerun/SAskip.py \
    --layerList 35 \
    --layerType v \
    --reg input \
    --dataflow WS \
    --injectConfig input_bitflip_13 \
    --pe -1 -1 \
    --detectLayerLower 33 \
    --sr_threshold_lower 1.2 \
    --sr_cmp_lower "<=" \
    --sr_count_mode_lower consecutive \
    --sr_trigger_k_lower 3 \
    --detectLayerUpper 0 \
    --sr_threshold_upper ${THR} \
    --sr_cmp_upper ">=" \
    --sr_count_mode_upper consecutive \
    --sr_trigger_k_upper 3

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished with upper_threshold=${THR}"
