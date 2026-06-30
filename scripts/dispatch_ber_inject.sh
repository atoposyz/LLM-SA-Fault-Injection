#!/bin/bash
set -euo pipefail

PROJECT_DIR="/workplace/home/mayongzhe/faultinject"
RESULT_DIR="${PROJECT_DIR}/result"

if [ -n "${MACHINES:-}" ]; then
  read -ra MACHINES <<< "$MACHINES"
else
  MACHINES=(10.4.1.117 10.4.1.118)
fi
GPUS=(0 1)
SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no"

LOCAL_IPS=" $(hostname -I) "

is_local() {
  [[ "$LOCAL_IPS" == *" $1 "* ]]
}

run_remote() {
  local host=$1 cmd=$2
  if is_local "$host"; then
    bash -c "$cmd"
  else
    ssh $SSH_OPTS "$host" "$cmd"
  fi
}

JOBS=(
  "random mixed"
  "IS input"
  "IS weight"
  "IS psum"
  "OS input"
  "OS weight"
  "OS psum"
  "WS input"
  "WS weight"
  "WS psum"
)

# Per-GPU cooldown: after dispatching, block re-dispatch for COOLDOWN seconds
# to allow model loading to allocate VRAM before next GPU check.
COOLDOWN=60
declare -A GPU_LAST_DISPATCH

can_dispatch_to() {
  local key="$1:$2"
  local now last
  now=$(date +%s)
  last="${GPU_LAST_DISPATCH["$key"]:-0}"
  if [ $((now - last)) -lt $COOLDOWN ]; then
    return 1
  fi
  return 0
}

check_gpu_free() {
  local host=$1 gpu_idx=$2

  # Cooldown: skip GPU that was just dispatched to (model still loading)
  if ! can_dispatch_to "$host" "$gpu_idx"; then
    echo "busy"
    return
  fi

  local output
  if ! output=$(run_remote "$host" \
    "util=\$(nvidia-smi -i $gpu_idx --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 100); \
     mem=\$(nvidia-smi -i $gpu_idx --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999); \
     if [ \"\$util\" -lt 10 ] && [ \"\$mem\" -lt 500 ]; then echo free; else echo busy; fi" 2>/dev/null); then
    echo "busy"
    return
  fi
  echo "$output"
}

# ---------------------------------------------------------------------------
# Main dispatch loop
# ---------------------------------------------------------------------------
job_idx=0
total_jobs=${#JOBS[@]}

# Check if a combo already has output (running or finished by a previous dispatch)
combo_already_running() {
  local df=$1 reg=$2
  # If ANY ber file for this combo exists, it's already in progress or done
  local pattern="${RESULT_DIR}/ber_*_df${df}_reg${reg}.jsonl"
  compgen -G "$pattern" > /dev/null 2>&1
}

while [ $job_idx -lt $total_jobs ]; do
  dispatched=0

  for machine in "${MACHINES[@]}"; do
    for gpu in "${GPUS[@]}"; do
      if [ $job_idx -ge $total_jobs ]; then
        break 2
      fi

      read -r df reg <<< "${JOBS[$job_idx]}"

      # Skip combos already running or finished from a previous dispatch
      if combo_already_running "$df" "$reg"; then
        echo "[$(date +%H:%M:%S)] Skipping df=$df reg=$reg (already in progress)"
        job_idx=$((job_idx + 1))
        dispatched=$((dispatched + 1))
        continue
      fi

      status=$(check_gpu_free "$machine" "$gpu")
      if [ "$status" = "free" ]; then
        timestamp=$(date +%H:%M:%S)
        echo "[$timestamp] Dispatching df=$df reg=$reg to $machine:$gpu"

        run_remote "$machine" \
          "nohup ${PROJECT_DIR}/.venv/bin/python3 ${PROJECT_DIR}/projects/qwen3-8b/run_ber_inject.py \
           --gpu ${gpu} --filter-dataflow ${df} --filter-reg ${reg} \
           > /tmp/ber_${df}_${reg}.log 2>&1 &"

        GPU_LAST_DISPATCH["$machine:$gpu"]=$(date +%s)
        job_idx=$((job_idx + 1))
        dispatched=$((dispatched + 1))
      fi
    done
  done

  if [ $dispatched -eq 0 ] && [ $job_idx -lt $total_jobs ]; then
    echo "[$(date +%H:%M:%S)] All GPUs busy, waiting 30 seconds..."
    sleep 30
  fi
done

echo ""
echo "=== All ${total_jobs} jobs dispatched ==="
echo "Check logs at /tmp/ber_*.log on each machine"
echo "Results at: ${RESULT_DIR}/"
