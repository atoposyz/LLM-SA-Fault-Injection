#!/bin/bash

# --- 1. 配置区 ---
# !!! 请根据你的脉动阵列大小修改这里的行数和列数 !!!
ARRAY_ROWS=32
ARRAY_COLS=32

# !!! 设置你想同时运行的最大任务数 (根据你的CPU核心数调整) !!!
MAX_PARALLEL_JOBS=4
# CUDA_VISIBLE_DEVICES=1
# !!! 记得修改output_dir
PYTHON_SCRIPT="/workplace/home/mayongzhe/faultinject/projects/bert/SAInjectProPlus32.py"
BASE_ARGS="--layerType all --pos 26 --layerList 0 --injectConfig input_stuck_1"
OUTPUT_DIR="/workplace/home/mayongzhe/faultinject/projects/bert/result/pe_input_all_32x32bit26/"

export HF_HUB_OFFLINE=1

# --- 2. 执行区 ---

mkdir -p "$OUTPUT_DIR"
echo "所有输出文件将保存在: $OUTPUT_DIR"

echo "开始并行执行PE敏感度测试..."
echo "阵列大小: ${ARRAY_ROWS}x${ARRAY_COLS}"
echo "最大并行任务数: $MAX_PARALLEL_JOBS"

# 嵌套循环，遍历所有PE位置
for (( r=0; r<ARRAY_ROWS; r++ ))
do
  for (( c=0; c<ARRAY_COLS; c++ ))
  do
    # 每当达到最大并行任务数时，就等待一个任务完成后再继续
    if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL_JOBS ]]; then
      wait -n  # 等待任意一个后台任务完成
    fi

    # 将任务放入后台执行
    (
      output_file="${OUTPUT_DIR}"
      echo "--- 启动 PE($r, $c) ---"
      
      uv run "$PYTHON_SCRIPT" \
        --outputfile "$output_file" \
        $BASE_ARGS \
        --pe "$r" "$c"
        
      echo "--- PE($r, $c) 执行完毕. ---"
    ) &  # '&' 符号是关键，它让命令在后台运行
  done
done

# 等待所有剩余的后台任务全部完成
wait
echo "所有PE位置的测试已全部完成！"