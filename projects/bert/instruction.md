可以，但我建议你**稍微收敛一下目标**。这三个变更的方向是对的，尤其是“bit × position × layer_type”的统一表；但第 3 步“sigmoid 预测 accuracy drop”要谨慎，不要让它变成过拟合正式实验结果。你现在应该把它定位为**校准模型级响应**，而不是“用预实验精确预测正式实验”。

我的判断如下。

## 总体评价

你的行动计划可以改成：

> 构建一个 unified severity predictor，用于对故障配置进行风险排序，而不是逐点回归最终 accuracy drop。

这样更稳。三个变更都可以做，但优先级应该是：

1. **必须做：按 layer type 构建 bit severity table**
2. **必须做：构建 joint severity = bit severity × position coverage**
3. **可以做，但要谨慎：用少量正式实验拟合 nonlinear calibration**

---

## 变更 1：按层类型构建严重性表，可以做，而且很有必要

这个改动是合理的。

你现在的 `build_severity_table.py` 把所有 Linear 层混在一起，这会掩盖不同 operator 的数值分布差异。BERT 里至少应该区分：

```text
attention.query
attention.key
attention.value
attention.output.dense
intermediate.dense
output.dense
```

如果为了简洁，可以先合并成三类：

```text
attention: query / key / value / attention output
intermediate: intermediate.dense
output: output.dense
```

但我更建议保留两级：

```text
layer_type = attention / intermediate / output
op_type = query / key / value / attn_out / ffn_intermediate / ffn_output
```

因为正式注错时，如果你只注入 key/value，那么单独的 key/value 表更有指导性。

这一部分可以在论文里解释为：

```text
We first calibrate bit-level numerical severity for each operator group, since different operators exhibit different value distributions and therefore different sensitivity to bit-level faults.
```

---

## 变更 2：创建 `build_joint_severity.py`，这是最关键的一步

这个必须做。你现在预实验和正式实验不吻合，核心原因就是缺少这个联合表。

但注意，不能只是简单相乘：

[
S_{\text{joint}} = S_{\text{bit}} \times S_{\text{pos}}
]

最好写成：

[
S_{\text{joint}}(b,p,o,m)
=========================

S_{\text{bit}}(b,o,m)
\cdot
C_{\text{pos}}(p,o,m)
]

其中：

* (b)：bit index
* (p)：PE position
* (o)：operator 或 layer type
* (m)：fault mode，比如 input / weight / psum，stuck-at-0 / stuck-at-1
* (S_{\text{bit}})：单元素数值扰动强度
* (C_{\text{pos}})：该 PE 位置实际污染的元素数量或传播范围

这里的重点是：**position coverage 不能只用元素数量，也要区分传播模式。**

例如：

### weight fault

weight fault 在 WS 下通常影响的是映射到某个 PE 的少量权重元素。它的 coverage 可能很小，所以即使 bit severity 很高，joint severity 也会低。

### input fault

input fault 可能沿阵列方向传播，影响多个 MAC 结果，因此 position coverage 应该包含传播长度。

### psum fault

psum fault 不只是污染单个元素，而是污染部分和后续累加链路，所以它的 coverage 应该接近：

[
C_{\text{psum}} \approx \text{affected accumulation length} \times \text{affected output count}
]

所以 `build_joint_severity.py` 不应只做“位严重性 × PE 元素数”，而应根据 fault target 分别定义 coverage。

我建议你输出类似：

```json
{
  "mode": "input_stuck_1",
  "dataflow": "WS",
  "operator": "attention.key",
  "entries": [
    {
      "bit": 30,
      "pe_row": 12,
      "pe_col": 8,
      "bit_severity": 1.0,
      "position_coverage": 0.73,
      "joint_severity": 0.73,
      "rank": 1
    }
  ]
}
```

这样后续可以直接用于 representative sampling。

---

## 变更 3：非线性饱和校准可以做，但不要只用 input stuck-at-1 拟合一个全局 sigmoid

这个地方我最担心。

不同 fault mode 的响应曲线不一定一样：

| 故障类型              | 响应形态可能不同                    |
| ----------------- | --------------------------- |
| input stuck-at-1  | 明显阈值，适合 sigmoid             |
| weight stuck-at-0 | 可能长期接近 0，因为 coverage 太低     |
| psum fault        | 可能更接近阶跃或快速饱和                |
| sign bit fault    | 数值扰动大，但可能被归一化/残差抵消          |
| exponent high bit | 可能直接 NaN/Inf 或 catastrophic |

所以更合理的是：

[
\hat{D} = G_m(S_{\text{joint}})
]

也就是每种 mode 一个校准函数，而不是所有模式共享一个 sigmoid。

如果数据量不够，我建议先不要拟合复杂函数，而是做三档分类：

```text
low-impact
medium-impact
high-impact
```

对应：

```text
low: predicted drop < 1%
medium: 1%–10%
high: >10%
```

这样比强行回归 accuracy drop 更稳。

---

## 我建议你把计划改成这个版本

### Step 1：构建 operator-aware bit severity table

从当前的全局表改成按 operator group 输出：

```text
severity_table_weight_fp32_attention_key.json
severity_table_weight_fp32_attention_value.json
severity_table_weight_fp32_attention_query.json
severity_table_weight_fp32_ffn_intermediate.json
severity_table_weight_fp32_ffn_output.json

severity_table_activation_input_fp32_*.json
severity_table_psum_output_fp32_*.json
```

如果文件太多，可以先做三类：

```text
attention / intermediate / output
```

---

### Step 2：构建 dataflow-aware position coverage

不要只记录 PE 位置，而要记录该位置在不同故障对象下的污染范围：

```text
weight coverage: mapped weight elements
input coverage: propagated MAC/output elements
psum coverage: affected accumulation chain length
```


---

### Step 3：构建 joint severity ranking

使用：

[
S_{\text{joint}}
================

\operatorname{Norm}
\left(
S_{\text{bit}}
\cdot
C_{\text{pos}}
\right)
]

如果加入层敏感度，可以写成：

[
S_{\text{joint}}
================

\operatorname{Norm}
\left(
S_{\text{bit}}
\cdot
C_{\text{pos}}
\cdot
S_{\text{layer}}
\right)
]

但 `S_layer` 不要一开始就加。先做 bit × position，看相关性是否提升；再加入 layer sensitivity，展示逐步增强效果。

---

### Step 4：用正式实验验证 ranking，而不是强行拟合数值

建议你验证：

```text
Spearman correlation
Kendall tau
Top-k overlap
High/medium/low classification accuracy
```

例如：

| Predictor              | Spearman | Top-10 overlap | High-risk recall |
| ---------------------- | -------: | -------------: | ---------------: |
| bit-only               |      low |            low |              low |
| position-only          |   medium |         medium |           medium |
| bit × position         |   higher |         higher |           higher |
| bit × position × layer |     best |           best |             best |

这张表会非常有说服力。

---

### Step 5：非线性校准作为 optional

最后再加：

[
\hat{D} = G_m(S_{\text{joint}})
]

这里 (G_m) 可以是 sigmoid、piecewise function，或者 isotonic regression。

