# Qwen 3.5 故障注入仿真平台

本项目是一个基于 Qwen 3.5 模型的故障注入仿真平台。该平台利用 Systolic Array (SA) 架构抽象，模拟在硬件层级（如卷积/全连接计算单元中的 PE 阵列）发生比特翻转（Bit-flip）或固定存 0/1（Stuck-at 0/1）错误对大模型推理能力的影响。

## 1. Python 文件用途

| 文件名 | 用途描述 |
| :--- | :--- |
| `run_single_bit_inject.py` | **精确单比特注入脚本**。用于在指定的模型层、指定的 Kernel（如 Q, K, V, MLP 等）以及特定的 PE 阵列位置注入单个比特位故障。支持指定寄存器类型（input, weight, psum）。 |
| `run_ber_inject.py` | **大规模 BER 注入脚本**。根据比特错误率（BER）在模型的所有相关层和 Kernel 中随机播撒大量故障。支持混合寄存器注入和动态数据流切换，模拟高负载或恶劣硬件环境下的模型表现。 |
| `SAInjectProRandomex2.py` | **通用注错集成脚本**。集成了随机注错与精准注错逻辑的综合版本，支持在大规模样本集上进行循环测试，并能够自动根据层级架构（如 Qwen 3.5 的 3:1 混合注意力架构）切换注错映射。 |
| `printlayer.py` | **辅助工具脚本**。用于导出 Qwen 3.5 模型的详细层级结构、模块名称以及参数权重形状（Weight Shape），生成 `layernames.txt` 和 `layerstructure.txt`。 |

## 2. 运行示例

项目环境：基于 `transformers` 和 `torch`，数据集通常使用 `rajpurkar/squad` (validation)。

### 2.1 获取基准测试 (Baseline)
不启用故障注入运行模型，以获取原始性能数据：
```bash
python SAInjectProRandomex2.py --outputfile ./result
```
*注：当未在参数中特别声明注错逻辑或注入器处于禁用状态时，脚本会自动保存基准运行结果至 `result/origin_200.jsonl`。*

### 2.2 精确单比特注错示例
在第 0 层的 Query 矩阵 (q) 的权重复比特位注入一个错误：
```bash
python run_single_bit_inject.py \
    --layerList 0 \
    --layerType q \
    --reg weight \
    --injectConfig weight_bitflip_10 \
    --outputfile ./result
```

### 2.3 大规模随机注错 (BER) 示例
在每个 PE 阵列中平均注入 10 个随机故障（混合寄存器类型，随机数据流）：
```bash
python run_ber_inject.py \
    --ber 10 \
    --fixReg mixed \
    --fixDataflow random \
    --outputfile ./result
```

## 3. Baseline & 评估指标

### 3.1 默认配置说明
- **模型**: Qwen/Qwen3.5-4B (Hybrid Attention 架构，3:1 比例的 Linear Attention 与 Self Attention)。
- **硬件抽象**: 模拟 256x256 规模的脉动阵列 (Systolic Array)。
- **数据流映射**: 支持 WS (Weight Stationary), OS (Output Stationary), IS (Input Stationary)。
- **默认精度**: FP32 (权重/激活值计算)。

### 3.2 结果记录格式
注错运行结果将保存为 `.jsonl` 文件，包含以下关键字段用于后续分析：
- `generated_answer`: 受注错影响后生成的文本。
- `reference_answer`: 标准答案，用于计算 Exact Match (EM) 或 F1 分数。
- `fault_pe_row/col`: 记录故障发生的具体 PE 坐标。
- `token_length`: 生成结果的长度（用于评估注错是否导致模型崩溃输出乱码）。

### 3.3 预期 Baseline 表现
在无注入（Clean Model）的情况下，SQuAD 验证集样本生成的 `generated_answer` 应与 `reference_answer` 高度吻合。评估时建议通过计算生成的 EM 值作为性能下降的对比基准。
