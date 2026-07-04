#!/bin/bash
#SBATCH --job-name=saskip-oracle
#SBATCH --output=/workplace/home/mayongzhe/faultinject/projects/rerun/result/slurm_saskip_oracle_%j.out
#SBATCH --error=/workplace/home/mayongzhe/faultinject/projects/rerun/result/slurm_saskip_oracle_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gpus=0

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /workplace/home/mayongzhe/faultinject

uv run python projects/rerun/detect_garbled.py build-oracle \
  --saskip \
  --oracle projects/rerun/result/SAskip_llm_oracle.jsonl \
  --concurrency 10 \
  projects/rerun/result/SAskip_single_bit_*.jsonl

echo "Done: $(date)"
