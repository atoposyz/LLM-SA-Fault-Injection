## 实验计划

### 关键实验一：多模型不同ber下的软硬件比较

#### 模型

BERT, Qwen3-0.6B, Qwen3-7B, (Qwen3-14B), Llama3.1-8B-instruct, Llama3.1-8B-instruct-int8量化, Qwen1.5-MoE-A2.7B, (Qwen3.5-9B)

#### 数据集

Squad 2

#### 样本量

200/次实验

#### 变量

错误比特率BER、模拟硬件注入or软件注入

由 $10^{-7}$ 到 $10^{-2}$

#### 指标

acc下降

### 关键实验二：固定BER下不同错误pattern的比较

#### 模型

Qwen3-7B, (可加Llama3.1-8B-instruct)

#### 数据集

Squad2

#### 样本量

200/次实验

#### 变量

数据流、错误reg的组合($3\times 3$)

#### 指标

acc下降

### 关键实验三：不同任务下不同指标的比较

#### 任务1：mmlu（选择题）

BERT, Qwen3-0.6B, Qwen3-7B, Llama3-8B-instruct

注错方式：随机

样本量：200

指标：acc下降, confidence

#### 任务2：xlsum（文本摘要）

Qwen3-7B, Llama3.1-8B-instruct

注错方式：随机

样本量：200

指标：token change rate, BERTscore, 崩溃率（全感叹号或乱码）

#### 任务3：gsm8k（数学推理）

Qwen3-7B, Llama3.1-8B（测试一下开启和不开启思考模式的区别）

注错方式：随机

样本量：200

指标：acc下降, 崩溃率（全感叹号或乱码）

### 关键实验四：Stablerank与其他文章的加速比与准确率下降比较

不同文章框架的推理时间与准确率下降

### 常规实验一：参数量大对比

#### 模型

Qwen3-0.6B, Qwen3-7B, Qwen3-14B

#### 数据集

Squad 2

#### 样本量

200/次实验

### 常规实验二：架构对比

#### 模型

Qwen3-7B, Qwen1.5-MoE-A2.7B, Qwen3.5-9B

#### 数据集

Squad 2

#### 样本量

200/次实验

### 常规实验三：量化对比

#### 模型

Llama3.1-8B-instruct, Llama3.1-8B-instruct-int8量化

#### 数据集

GSM8K

#### 样本量

200/次实验

### 组件实验一：算子

### 组件实验二：层

### 组件实验三：比特位

三组实验均使用硬件注错，保持另两个变量随机的情况下控制本变量

模型：Llama3.1-8B-instruct

任务：gsm8k
