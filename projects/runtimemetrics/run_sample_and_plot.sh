#!/bin/bash
# 用法: uv run bash run_sample_and_plot.sh <sample_id> <pe_row> <pe_col>
# 示例: uv run bash run_sample_and_plot.sh 992 177 216
#
# 流程:
#   1. 用 SARuntimeMetricsRecord.py 生成指定样本+PE的 runtime metrics 记录
#   2. 用 plot_rm_comparison.py 与 baseline 对比出图

set -euo pipefail

SAMPLE_ID="${1:?Usage: $0 <sample_id> <pe_row> <pe_col>}"
PE_ROW="${2:?Missing pe_row}"
PE_COL="${3:?Missing pe_col}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/result"

LAYER_LIST=(35)
LAYER_TYPE="v"
REG="input"
DATAFLOW="WS"
INJECT_CFG="input_bitflip_13"
INTERVAL=50
DETECT_LAYERS=(0 3 6 9 12 15 18 21 24 27 30 33)

BASELINE_FILE="runtime_metrics_record_sidall_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_noinject.jsonl"

INJECT_FILE="runtime_metrics_record_sid${SAMPLE_ID}_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe${PE_ROW}_${PE_COL}_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_inject.jsonl"

# ---- Step 1: 检查是否已有结果，没有则生成 ----
if [ -f "${RESULT_DIR}/${INJECT_FILE}" ]; then
    echo "[SKIP] 记录已存在: ${INJECT_FILE}"
else
    echo "[STEP 1] 生成 runtime metrics 记录 (sample=${SAMPLE_ID}, PE=(${PE_ROW},${PE_COL})) ..."
    uv run python "${SCRIPT_DIR}/SARuntimeMetricsRecord.py" \
        --sampleid "${SAMPLE_ID}" \
        --pe "${PE_ROW}" "${PE_COL}" \
        --layerList "${LAYER_LIST[@]}" \
        --layerType "${LAYER_TYPE}" \
        --reg "${REG}" \
        --dataflow "${DATAFLOW}" \
        --injectConfig "${INJECT_CFG}" \
        --interval "${INTERVAL}" \
        --detectLayers "${DETECT_LAYERS[@]}"
    echo "[STEP 1] 完成"
fi

# ---- Step 2: 画对比图 ----
echo "[STEP 2] 画图 (sample_id=${SAMPLE_ID}) ..."
uv run python "${SCRIPT_DIR}/plot_rm_comparison.py" \
    --result-dir "${RESULT_DIR}" \
    --inject-file "${INJECT_FILE}" \
    --baseline-file "${BASELINE_FILE}" \
    --samples "${SAMPLE_ID}"

echo "[DONE] 图表保存在: ${RESULT_DIR}/plots/sample_${SAMPLE_ID}/"
