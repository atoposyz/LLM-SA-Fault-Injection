# Qwen3-8B BER 多点故障注入脚本设计

## 概述

在 `projects/qwen3-8b/` 下新建 `run_ber_inject.py`，使用 `BER_Fast_SA_FaultInjector` 对 Qwen3-8B 进行多数据流、多寄存器位置、多 BER 梯度的系统化故障注入实验。

## 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--outputfile / -o` | 输出目录 | `./result` |
| `--gpu / -g` | CUDA device（设 `CUDA_VISIBLE_DEVICES`） | `0` |
| `--filter-dataflow` | 限定数据流，空格分隔 | random IS OS WS |
| `--filter-reg` | 限定寄存器，空格分隔 | mixed input weight psum |
| `--filter-ber` | 限定 BER 值，空格分隔 | 1e-9 ~ 1e-4 |
| `--num-samples / -n` | 样本数 | 200 |
| `--no-cache-priority` | 禁用本地缓存优先 | False |

## 组合定义（10 组 × 6 BER = 60 种）

| dataflow | reg |
|---|---|
| random | mixed |
| IS | input / weight / psum |
| OS | input / weight / psum |
| WS | input / weight / psum |

BER 梯度：`1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4`

## 关键设计决策

### 小 BER 概率注入

`total_bits = 256 × 256 × num_regs × 32`。当 `total_bits × BER < 1` 时，不以绝对数量注入，改为每次 hook 调用以 `p = total_bits × BER` 概率注入 1 个随机错。

### 故障类型

每样本随机选择 `stuck_0` 或 `stuck_1`（不含 bitflip）。

### 随机种子重置

每个组合开始前重置 `random.seed(42)` + `torch.manual_seed(42)`，确保跨组合的随机序列一致——同一 `sample_id` 的故障类型选择相同，BER 的空间分布相同，变量仅 dataflow、reg 类型、BER 总量。

### 模型架构

Qwen3-8B 使用标准注意力（每层 q/k/v/o），不需要 3:1 层判断逻辑。

### 断点续跑

输出文件已存在则跳过该组合。

### 输出

每组合一个 JSONL 文件：`ber_{ber}_df{dataflow}_reg{reg}.jsonl`

```json
{"sample_id": "0", "token_length": 128, "reference_answer": "...", "generated_answer": "...", "fault_type": "random_stuck_0_mixed", "dataflow": "WS", "ber": 1e-06, "fix_reg": "weight", "inject_layer": "all", "inject_kernel": "all"}
```

### 并行调度

脚本只负责单 GPU 单进程，通过 `--filter-*` 参数拆分工作，并行调度由外部 shell 管理。示例：

```bash
# GPU 0: random + IS
uv run python run_ber_inject.py --gpu 0 --filter-dataflow random IS &
# GPU 1: OS + WS
uv run python run_ber_inject.py --gpu 1 --filter-dataflow OS WS &
```

## 执行流程

1. 解析参数，`CUDA_VISIBLE_DEVICES=gpu`
2. 加载 `Qwen/Qwen3-8B` + tokenizer（优先本地缓存）
3. 加载 GSM8K，seed=37 取 200 样本
4. 构建组合列表（filter 过滤）
5. 按组合遍历：
   - 重置种子
   - 计算输出文件名，存在则跳过
   - 初始化 `BER_Fast_SA_FaultInjector(precision='bf16')`
   - 遍历样本：
     - 选择 fault_type → `parse_fault_type()`
     - 设置 dataflow（random 模式每样本随机）
     - `init_faults_by_ber(BER)` 或小 BER 概率注入
     - 挂载所有层 q/k/v/o + mlp-gate/up/down 的 hook
     - 生成 → 记录结果 → 清理 hook
