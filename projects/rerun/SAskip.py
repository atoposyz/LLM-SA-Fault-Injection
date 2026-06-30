'''
SAskip

rank识别出数值异常后，直接跳过该样本，推理下一个样本

=============================================================================
硬编码配置（写死在代码中，修改需改源码）：
=============================================================================
  - 模型: Qwen/Qwen3-8B (bf16, device_map="auto", trust_remote_code=True)
  - 数据集: openai/gsm8k (main, split="train"), random.seed(37), 随机采样1000条
  - 脉动阵列尺寸: sa_rows=256, sa_cols=256
  - 数值精度: bf16
  - pos_mapping (kernel名 → 模块路径映射):
      q → self_attn.q_proj         k → self_attn.k_proj
      v → self_attn.v_proj         ffn-gate → mlp.gate_proj
      ffn-up → mlp.up_proj         ffn-down → mlp.down_proj
      attention-norm → post_attention_layernorm   input-norm → input_layernorm
  - sr_cfg_lower 默认值: threshold=1.2, cmp="<=", count_mode="consecutive",
    trigger_k=3, window_n=10, detect_layer=0
  - sr_cfg_upper 默认值: threshold=1.2, cmp=">=", count_mode="consecutive",
    trigger_k=3, window_n=10, detect_layer=0
  - shared_cfg: interval=50（两个检测器共用）
  - 重新生成阶段: N/A (skip模式，不重新生成)
  - 常量: MAX_float_16=65504, calculateSize=16, headSize=128
  - torch.no_grad() 下运行，不使用梯度

=============================================================================
可配置参数（通过命令行 --arg 指定）：
=============================================================================
  --layerList      注入层编号列表          (int, nargs='+', default=[0])
  --affect         是否影响后续所有层        (flag, default=False)
  --layerType      注入算子类型            (choices: q/k/v/ffn-gate/ffn-up/ffn-down/
                                           attention-norm/input-norm, default="q")
  --outputfile     输出目录               (str, default=".../rerun/result")
  --sampleid       指定样本ID列表          (int, nargs='*', default=[])
  --pe             手动指定PE位置(row col)  (int, nargs=2, default=[-1 -1]=随机)
  --reg            注入寄存器类型           (choices: input/weight/psum, default='weight')
  --dataflow       数据流类型              (choices: WS/OS/IS, default="WS")
  --injectConfig   注入错误类型字符串       (str, default="weight_bitflip_10")
                   格式: {mode}_{op}[_{pos}], mode=weight/input/psum,
                   op=bitflip/stuck_0/stuck_1, pos=bit位置
  --interval           StableRank计算间隔（两检测器共用）  (int, default=50)
  --detectLayerLower   下阈值检测层ID                        (int, default=0)
  --detectLayerUpper   上阈值检测层ID                        (int, default=0)
  --max_tokens         最大生成token数                       (int, default=5000)
  --sr_threshold_lower 下阈值                               (float, default=1.2)
  --sr_threshold_upper 上阈值                               (float, default=1.2)
  --sr_cmp_lower       下阈值比较运算符 (>=/<=/>/<)          (default="<=")
  --sr_cmp_upper       上阈值比较运算符 (>=/<=/>/<)          (default=">=")
  --sr_count_mode_lower  下阈值计数方式 (consecutive/cumulative/window)  (default="consecutive")
  --sr_count_mode_upper  上阈值计数方式                      (default="consecutive")
  --sr_trigger_k_lower   下阈值触发提前终止所需次数K        (int, default=3)
  --sr_trigger_k_upper   上阈值触发提前终止所需次数K        (int, default=3)
  --sr_window_n_lower    下阈值 window模式窗口大小N         (int, default=10)
  --sr_window_n_upper    上阈值 window模式窗口大小N         (int, default=10)

=============================================================================
程序行为：
=============================================================================
  1. 加载模型和数据集
  2. 在 detectLayerLower 和 detectLayerUpper 上分别注册 cache hook，缓存各层输出；
     在 max(detectLayerLower, detectLayerUpper) 上注册 check hook，每隔 interval 个
     token 计算两层的 StableRank (= Frobenius范数² / 谱范数²)
  3. 在目标层 target_layers 上注册 fault injector hook
  4. 逐 token 生成（max_tokens 步），每个 step 只输入最后一个 token + KV cache
  5. 当下阈值或上阈值检测器的异常次数达到各自的 sr_trigger_k 时，触发 StopForwardException
     → 保存被中断的结果（is_aborted=True），直接跳过该样本进入下一个
  6. 结果写入 JSONL，每条包含: sample_id, token_length, should_aborted,
     is_aborted, 注入参数, reference_answer, generated_answer, thinking_process

=============================================================================
执行命令示例：
=============================================================================

  # 1. 最简运行：默认配置，注入第0层q_proj的weight bitflip第10位
  uv run python projects/rerun/SAskip.py

  # 2. 在 layer 0,2,4 的 k_proj 注入 weight_stuck_1_15, OS数据流，
  #    层0检测下阈值(<=0.8)，层2检测上阈值(>=5.0)
  uv run python projects/rerun/SAskip.py \
      --layerList 0 2 4 \
      --layerType k \
      --injectConfig weight_stuck_1_15 \
      --dataflow OS \
      --detectLayerLower 0 \
      --detectLayerUpper 2 \
      --sr_threshold_lower 0.8 \
      --sr_threshold_upper 5.0

  # 3. 注入 input 寄存器 bitflip，指定PE位置 (12,34)，影响layer5及后续所有层
  uv run python projects/rerun/SAskip.py \
      --layerList 5 \
      --affect \
      --reg input \
      --injectConfig input_bitflip_7 \
      --pe 12 34 \
      --detectLayerLower 5 \
      --detectLayerUpper 7

  # 4. 调试：只跑指定样本0,5,10，max_tokens=500，两检测器均用累计计数
  uv run python projects/rerun/SAskip.py \
      --sampleid 0 5 10 \
      --max_tokens 500 \
      --sr_count_mode_lower cumulative \
      --sr_count_mode_upper cumulative \
      --sr_trigger_k_lower 5 \
      --sr_trigger_k_upper 5 \
      --sr_threshold_lower 0.5 \
      --sr_threshold_upper 10.0

  # 5. psum注入，注入ffn-down层，指定输出目录
  uv run python projects/rerun/SAskip.py \
      --layerList 1 \
      --layerType ffn-down \
      --reg psum \
      --injectConfig psum_stuck_0_3 \
      --dataflow WS \
      --outputfile ./my_results/

'''

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from struct import pack, unpack
import torch
from cmath import isinf, isnan
import random
import torch
import numpy as np
import operator
from collections import deque

from tool.single_bit_injector import SingleBit_Fast_SA_FaultInjector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.set_verbosity_error()
import os
import argparse
import sys
import json
import time

MAX_float_16 = 65504
calculateSize = 16
headSize = 128

inject_pos = {'value': 0}
selectHead = {'value': 0}
block = 8
blockX = 9
blockY = 9
flag = {'firstinit': 0, 'value': 0, 'normal': 0, 'is_rerun': 0, 'triggered_detector': ''}

# StableRank detect shared config
shared_cfg = {
    "interval": 50,              # 计算StableRank的间隔（两个检测器共用）
}

# Lower threshold detector config
sr_cfg_lower = {
    "threshold": 1.2,            # StableRank下阈值
    "cmp": "<=",                 # 比较符: >= <= > <
    "count_mode": "consecutive", # consecutive | cumulative | window
    "trigger_k": 3,              # 触发次数K
    "window_n": 10,              # window模式窗口大小N
    "detect_layer": 33,           # 计算stable rank的层id
}

# Upper threshold detector config
sr_cfg_upper = {
    "threshold": 1.8,            # StableRank上阈值
    "cmp": ">=",                 # 比较符: >= <= > <
    "count_mode": "consecutive", # consecutive | cumulative | window
    "trigger_k": 3,              # 触发次数K
    "window_n": 10,              # window模式窗口大小N
    "detect_layer": 0,           # 计算stable rank的层id
}

# Lower threshold detector state
sr_state_lower = {
    "value": 0,      # 计数器（连续/累计/窗口计数）
    "buf": None,     # window模式下deque
}

# Upper threshold detector state
sr_state_upper = {
    "value": 0,      # 计数器（连续/累计/窗口计数）
    "buf": None,     # window模式下deque
}


faultin = { 'value': 0}
set_fault = {"fx": 0, "fy": 0, "bx": 0, "by": 0}
layer_output_cache = {}

_CMP = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}

def sr_update_and_should_stop(stable_rank: float, sr_cfg: dict, sr_state: dict, is_rerun: int) -> bool:
    """
    根据 sr_cfg 更新 sr_state，并返回是否应该触发 early stop。
    - sr_cfg: threshold/cmp/count_mode/trigger_k/window_n
    - sr_state: value/buf
    - is_rerun: 1表示重跑阶段，不触发StopForwardException
    """
    cmp_op = sr_cfg.get("cmp", ">=")
    cmp_fn = _CMP.get(cmp_op)
    if cmp_fn is None:
        raise ValueError(f"Unsupported sr_cfg['cmp']: {cmp_op}")

    thr = float(sr_cfg.get("threshold", 0.0))
    is_abnormal = bool(cmp_fn(stable_rank, thr))

    mode = sr_cfg.get("count_mode", "consecutive")
    k = int(sr_cfg.get("trigger_k", 3))

    if mode == "consecutive":
        if is_abnormal:
            sr_state["value"] += 1
        else:
            sr_state["value"] = 0

    elif mode == "cumulative":
        if is_abnormal:
            sr_state["value"] += 1
        # 正常不清零

    elif mode == "window":
        n = int(sr_cfg.get("window_n", 10))
        buf = sr_state.get("buf")
        if buf is None or not isinstance(buf, deque) or buf.maxlen != n:
            buf = deque(maxlen=n)
            sr_state["buf"] = buf
        buf.append(1 if is_abnormal else 0)
        sr_state["value"] = sum(buf)

    else:
        raise ValueError(f"Unsupported sr_cfg['count_mode']: {mode}")

    return (sr_state["value"] >= k) and (is_rerun == 0)


class StopForwardException(Exception):
    """用于在hook中终止当前forward的异常类型"""
    pass


def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

faultpos = {'value':0}


def make_cache_hook(layer_id: int):
    """返回一个 forward hook，只缓存指定层的输出 token，不做任何检查。"""
    def hook(module, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output

        if isinstance(out_tensor, torch.Tensor):
            if out_tensor.dim() == 3:
                token_out = out_tensor[0, -1, :].detach().cpu().unsqueeze(0)  # [1, D]
            elif out_tensor.dim() == 2:
                token_out = out_tensor[-1, :].detach().cpu().unsqueeze(0)     # [1, D]
            else:
                return output

            if layer_id not in layer_output_cache:
                layer_output_cache[layer_id] = []
            layer_output_cache[layer_id].append(token_out)

        return output
    return hook


def make_check_hook():
    """返回一个 forward hook，递增 token 计数器并周期性执行双检测器 StableRank 检查。"""
    def hook(module, input, output):
        flag['normal'] = flag['normal'] + 1

        if flag['normal'] % shared_cfg["interval"] == 0 and flag['normal'] != 0:
            for layer_id in sorted(layer_output_cache.keys()):
                outputs = layer_output_cache[layer_id]  # List of [1, D]

                if len(outputs) == 0:
                    continue

                matrix = torch.cat(outputs, dim=0).to(torch.float32)  # [seq_len, dim]

                total_tokens = matrix.shape[0]
                start = 0
                end = min(start + shared_cfg["interval"], total_tokens)
                sub_matrix = matrix[start:end, :]

                if sub_matrix.numel() == 0:
                    continue

                if not torch.isfinite(sub_matrix).all():
                    stable_rank = 0.0
                    print(
                        f"[WARN] Non-finite value detected in StableRank input. "
                        f"Layer={layer_id}, Tokens={flag['normal'] - shared_cfg['interval']}-{flag['normal']}. "
                        f"Set stable_rank=0.0."
                    )
                else:
                    fro_norm_sq = torch.norm(sub_matrix, p='fro').pow(2).item()
                    sigma_max_sq = torch.linalg.norm(sub_matrix, ord=2).pow(2).item()
                    stable_rank = fro_norm_sq / sigma_max_sq if sigma_max_sq > 1e-12 else 0.0

                # 对两个检测器分别检查，任一触发则抛出异常
                for cfg, state, tag in [
                    (sr_cfg_lower, sr_state_lower, "LOWER"),
                    (sr_cfg_upper, sr_state_upper, "UPPER"),
                ]:
                    if cfg["detect_layer"] != layer_id:
                        continue
                    if sr_update_and_should_stop(stable_rank, cfg, state, flag.get("is_rerun", 0)):
                        flag["triggered_detector"] = tag.lower()
                        raise StopForwardException(
                            f"[ERROR] [{tag}] Layer {layer_id} 在 Tokens {flag['normal'] - shared_cfg['interval']}-{flag['normal']} 处 "
                            f"StableRank触发提前终止：sr={stable_rank:.4f}, thr={cfg.get('threshold')}, "
                            f"cmp={cfg.get('cmp')}, mode={cfg.get('count_mode')}, "
                            f"k={cfg.get('trigger_k')}, cnt={state.get('value')}."
                        )

            layer_output_cache.clear()

        return output
    return hook

def safe_decode(tokenizer, token_ids_or_tensor, name="generated"):
    if isinstance(token_ids_or_tensor, torch.Tensor):
        token_ids = token_ids_or_tensor.detach().cpu().tolist()
    else:
        token_ids = token_ids_or_tensor

    # 防止 generated[0] 之外传进二维 list
    if len(token_ids) > 0 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    vocab_size = len(tokenizer)
    clean_ids = []
    bad_ids = []

    for i, t in enumerate(token_ids):
        try:
            t_int = int(t)
        except Exception:
            bad_ids.append((i, t, "not_int"))
            continue

        if 0 <= t_int < vocab_size:
            clean_ids.append(t_int)
        else:
            bad_ids.append((i, t_int, "out_of_range"))

    if bad_ids:
        print(f"[WARN] {name}: removed {len(bad_ids)} invalid token ids.")
        print(f"[WARN] tokenizer vocab_size={vocab_size}")
        print(f"[WARN] first bad ids: {bad_ids[:20]}")

    decoded = tokenizer.decode(clean_ids, skip_special_tokens=True)
    return decoded, clean_ids, bad_ids

def resultoutput(token_length, is_aborted, decoded, idx, prompt, sample, layerid, layertype, posid, f, elapsed_time=0.0):
    sep = 'assistant\n'
    try:
        response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
    except IndexError:
        response = decoded.strip().replace("\n\n", "\n")

    result_data = {
        "sample_id": idx,
        "token_length": token_length,
        "should_aborted": True if (sr_state_lower['value'] >= sr_cfg_lower.get("trigger_k", 3) or
                                     sr_state_upper['value'] >= sr_cfg_upper.get("trigger_k", 3)) else False,
        "is_aborted": is_aborted,
        "elapsed_time": round(elapsed_time, 3),
        "fault_type": injector.fault_type_str,
        "inject_layer": layerid,
        "inject_kernel": layertype,
        "inject_pos": posid,
        "fault_pe_row": injector.fault_pe_row[0] if injector.fault_pe_row else -1,
        "fault_pe_col": injector.fault_pe_col[0] if injector.fault_pe_col else -1,
        "fault_pe_reg": injector.fault_reg[0] if injector.fault_reg else 0,
        "triggered_detector": flag.get("triggered_detector", ""),
        "dataflow": injector.dataflow,
        "reference_answer": sample['answer'],
        "generated_answer": response.split('</think>', 1)[1].strip() if '</think>' in decoded else "",
        "thinking_process": response.split('</think>', 1)[0].strip() if '</think>' in decoded else response,
        # "prompt": prompt,
        }
    json_line = json.dumps(result_data, ensure_ascii=False)
    f.write(json_line + "\n")
    f.flush()


pos_mapping = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "ffn-gate": "mlp.gate_proj",
    "ffn-up": "mlp.up_proj",
    "ffn-down": "mlp.down_proj",
    "attention-norm": "post_attention_layernorm",
    "input-norm": "input_layernorm",
}

if __name__ == "__main__":
    
    # 参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="输入错误注入层数，默认为0")
    parser.add_argument("--affect", action='store_true', help="是否开启注入后续层数，默认为False")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(), default="q", help="注入层选项")
    parser.add_argument("--outputfile", type=str, default="/workplace/home/mayongzhe/faultinject/projects/rerun/result", help="输出目录")
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="指定样本ID")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置(row col)，-1 -1表示全阵列随机选1个点")
    parser.add_argument("--reg", type=str, choices=['input', 'weight', 'psum'], default='weight', help="指定注入特定的寄存器")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"], help="数据流类型")
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10", help="注入错误类型")


    parser.add_argument("--interval", type=int, default=50, help="打印注入信息的间隔，默认为50")
    parser.add_argument("--detectLayerLower", type=int, default=0, help="下阈值检测层id，默认为0")
    parser.add_argument("--detectLayerUpper", type=int, default=0, help="上阈值检测层id，默认为0")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum number of new tokens to generate (default: 5000)")
    parser.add_argument("--sr_threshold_lower", type=float, default=1.2, help="StableRank下阈值")
    parser.add_argument("--sr_threshold_upper", type=float, default=1.2, help="StableRank上阈值")
    parser.add_argument("--sr_cmp_lower", type=str, choices=[">=", "<=", ">", "<"], default="<=",
                        help="下阈值比较运算符")
    parser.add_argument("--sr_cmp_upper", type=str, choices=[">=", "<=", ">", "<"], default=">=",
                        help="上阈值比较运算符")
    parser.add_argument("--sr_count_mode_lower", type=str, choices=["consecutive", "cumulative", "window"],
                        default="consecutive", help="下阈值计数方式：连续/累计/滑动窗口")
    parser.add_argument("--sr_count_mode_upper", type=str, choices=["consecutive", "cumulative", "window"],
                        default="consecutive", help="上阈值计数方式：连续/累计/滑动窗口")
    parser.add_argument("--sr_trigger_k_lower", type=int, default=3, help="下阈值触发提前终止所需次数K")
    parser.add_argument("--sr_trigger_k_upper", type=int, default=3, help="上阈值触发提前终止所需次数K")
    parser.add_argument("--sr_window_n_lower", type=int, default=10, help="下阈值 window模式窗口大小N")
    parser.add_argument("--sr_window_n_upper", type=int, default=10, help="上阈值 window模式窗口大小N")

    args = parser.parse_args()
    
    ifsample = len(args.sampleid) > 0
    shared_cfg["interval"] = args.interval
    sr_cfg_lower["detect_layer"] = args.detectLayerLower
    sr_cfg_upper["detect_layer"] = args.detectLayerUpper
    print(f"[INFO] 下阈值检测层: {args.detectLayerLower}, 上阈值检测层: {args.detectLayerUpper}")
    print(f"[INFO] 注入目标 Kernel 为：{args.layerType}, 层数列表：{args.layerList}")  
    max_new_tokens = args.max_tokens
    layerlist = args.layerList
    print(f"[INFO] 注入层列表为：{layerlist}")
    if not args.outputfile.endswith("/"):
        args.outputfile += "/"
    out_path = args.outputfile
    sample_index = args.sampleid
    print(f"[INFO] 输出文件路径为：{out_path}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"[INFO] 模型已加载")
    # 加载数据集
    ds = load_dataset("openai/gsm8k", "main", split="train")
    # ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    random.seed(37)
    samples = random.sample(list(ds), 1000)[0:1000]
    print("[INFO] 数据集已加载")

    if sample_index:
        # 指定了 sampleid，则从数据集中按索引取样
        print(f"[INFO] 指定样本ID：{sample_index}")
        try:
            samples = [samples[i] for i in sample_index]
        except IndexError as e:
            raise ValueError(f"指定的样本ID越界：{e}")
    else:
        print("[INFO] 未指定样本ID，使用随机选择的全部样本进行注错测试")
        sample_index = [str(i) for i in range(len(samples))]  # 生成样本索引列表
    injector = SingleBit_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow=args.dataflow, 
        fault_type=args.injectConfig, precision='bf16'
    )

    reg_map = {'input': 0, 'weight': 1, 'psum': 2}
    reg_target_id = reg_map.get(args.reg, 1)
    injector.fault_reg.append(reg_target_id) 
    if args.pe != [-1, -1]:
        injector.set_specific_fault(args.pe[0], args.pe[1])
    else:
        injector.init_random_fault()

    layertype = args.layerType
    posid = injector.fault_config.get('pos', 0)


    pe_tag = f"pe{args.pe[0]}_{args.pe[1]}" if args.pe != [-1, -1] else "pe_random"
    affect_tag = "affect1" if args.affect else "affect0"

    

    # Register cache hooks on unique detection layers
    detect_layers = list(set([sr_cfg_lower["detect_layer"], sr_cfg_upper["detect_layer"]]))
    check_layer = max(detect_layers)

    sr_cfg_lower["threshold"] = args.sr_threshold_lower
    sr_cfg_lower["cmp"] = args.sr_cmp_lower
    sr_cfg_lower["count_mode"] = args.sr_count_mode_lower
    sr_cfg_lower["trigger_k"] = args.sr_trigger_k_lower
    sr_cfg_lower["window_n"] = args.sr_window_n_lower

    sr_cfg_upper["threshold"] = args.sr_threshold_upper
    sr_cfg_upper["cmp"] = args.sr_cmp_upper
    sr_cfg_upper["count_mode"] = args.sr_count_mode_upper
    sr_cfg_upper["trigger_k"] = args.sr_trigger_k_upper
    sr_cfg_upper["window_n"] = args.sr_window_n_upper

    print(f"[INFO] Lower detector: layer={sr_cfg_lower['detect_layer']}, thr={sr_cfg_lower['threshold']}, "
        f"cmp={sr_cfg_lower['cmp']}, mode={sr_cfg_lower['count_mode']}, k={sr_cfg_lower['trigger_k']}, "
        f"window_n={sr_cfg_lower['window_n']}")
    print(f"[INFO] Upper detector: layer={sr_cfg_upper['detect_layer']}, thr={sr_cfg_upper['threshold']}, "
        f"cmp={sr_cfg_upper['cmp']}, mode={sr_cfg_upper['count_mode']}, k={sr_cfg_upper['trigger_k']}, "
        f"window_n={sr_cfg_upper['window_n']}")

    cache_handles = []
    for layer_id in detect_layers:
        h = model.model.layers[layer_id].register_forward_hook(make_cache_hook(layer_id))
        cache_handles.append(h)
    # Register check hook on the highest-indexed detection layer
    check_handle = model.model.layers[check_layer].register_forward_hook(make_check_hook())

    for layerid in args.layerList:
        print(f"[INFO] 注入算子为：{layertype}")   
        # print(f"[INFO] 当前注入配置：{args.injectConfig}, 注入寄存器：{args.reg}, 数据流：{args.dataflow}")
        print(f"[INFO] 当前注入层：{layerid} in {layerlist}")
        injector.print_config()
        def make_sr_tag(prefix, cfg):
            tag = f"{prefix}_sr{cfg['cmp']}{cfg['threshold']}"
            if cfg["count_mode"] == "consecutive":
                tag += f"_cons{cfg['trigger_k']}"
            elif cfg["count_mode"] == "cumulative":
                tag += f"_cum{cfg['trigger_k']}"
            elif cfg["count_mode"] == "window":
                tag += f"_win{cfg['trigger_k']}_w{cfg['window_n']}"
            else:
                tag += "_unk"
            tag += f"_L{cfg['detect_layer']}"
            return tag

        sr_tag = make_sr_tag("lower", sr_cfg_lower) + "_" + make_sr_tag("upper", sr_cfg_upper)
        output_filename = (
            f"SAskip_single_bit_"
            f"kernel-{layertype}_"
            f"layer-{layerid}_"
            f"reg-{args.reg}_"
            f"df-{args.dataflow}_"
            f"cfg-{args.injectConfig}_"
            f"{pe_tag}_"
            f"{sr_tag}"
            f"{affect_tag}.jsonl"
        )
        output_path = os.path.join(out_path, output_filename)

        target_layers = [layerid]
        if args.affect:
            target_layers = list(range(layerid, model.config.num_hidden_layers))
        handles = []
        for layernumber in target_layers:
            current_layertype = layertype
            if current_layertype in pos_mapping:
                hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[current_layertype])
                handles.append(hook.register_forward_hook(injector.hook_fn))
        # 生成文本并写入
        with open(output_path, "a", encoding="utf-8") as f:
            for idx, sample in enumerate(tqdm(samples)):
                if args.pe == [-1, -1]:
                    injector.init_random_fault()
                injector.enabled = True
                messages = [
                    {
                        'role': 'user',
                        'content':  "Solve the following math problem step by step.\n"
                        + sample["question"]
                        + "\nAnswer:"
                    }
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                tokenized = tokenizer(prompt, return_tensors="pt")
                num_tokens = tokenized["input_ids"].shape[-1]

                
                generated = inputs["input_ids"]          # [1, T_prompt]
                past_key_values = None

                t_start = time.time()
                try:
                    sr_state_lower["value"] = 0
                    sr_state_lower["buf"] = None
                    sr_state_upper["value"] = 0
                    sr_state_upper["buf"] = None
                    layer_output_cache.clear()
                    faultpos['value'] = 0
                    flag['normal'] = 0
                    flag['is_rerun'] = 0
                    flag['triggered_detector'] = ''
                    with torch.no_grad():
                        for step in range(max_new_tokens):
                            if past_key_values is None:
                                model_inputs = {"input_ids": generated}
                            else:
                                model_inputs = {
                                    "input_ids": generated[:, -1:],   # 只输入最后一个token
                                    "past_key_values": past_key_values,
                                    "use_cache": True,
                                }

                            outputs = model(**model_inputs)
                            past_key_values = outputs.past_key_values
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]
                            generated = torch.cat([generated, next_token], dim=-1)
                            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                                break

                    # 正常生成完成
                    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
                    token_length = generated.shape[1] - num_tokens
                    resultoutput(token_length=token_length, is_aborted=False, decoded=decoded, idx=sample_index[idx], prompt=prompt, sample=sample, layerid=layerid, layertype=layertype, posid=posid, f=f, elapsed_time=time.time() - t_start)

                except StopForwardException as e:
                    # 到达阈值，保存被中断结果，直接跳过该样本
                    decoded, clean_ids, bad_ids = safe_decode(tokenizer, generated[0], name="aborted_generated")
                    token_length = generated.shape[1] - num_tokens
                    resultoutput(token_length=token_length, is_aborted=True, decoded=decoded, idx=sample_index[idx], prompt=prompt, sample=sample, layerid=layerid, layertype=layertype, posid=posid, f=f, elapsed_time=time.time() - t_start)
                    # 直接 continue 到下一个样本，不重新运行
        print(f"已完成生成，结果保存在：{output_path}")
        for handle in handles:
            handle.remove()
    for h in cache_handles:
        h.remove()
    check_handle.remove()



