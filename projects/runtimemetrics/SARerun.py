'''
ReRun

rank识别出数值异常后，重新运行该样本

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

MAX_float_16 = 65504
calculateSize = 16
headSize = 128

inject_pos = {'value': 0}
selectHead = {'value': 0}
block = 8
blockX = 9
blockY = 9
flag = {'firstinit': 0, 'value': 0, 'normal': 0, 'is_rerun': 0}

# StableRank detect config
sr_cfg = {
    "interval": 50,              # 计算StableRank的间隔
    "threshold": 1.2,            # StableRank阈值
    "cmp": "<=",                 # 比较符: >= <= > <
    "count_mode": "consecutive", # consecutive | cumulative | window
    "trigger_k": 3,              # 触发次数K
    "window_n": 10,              # window模式窗口大小N
    "detect_layer": 0,           # 计算stable rank的层id
}

# StableRank detect state
sr_state = {
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

# def OutputStableRankTest(module, input, output):
#     flag['normal'] = flag['normal'] + 1
#     out_tensor = output[0] if isinstance(output, tuple) else output
#     if isinstance(out_tensor, torch.Tensor):
#         # out_tensor: [B, T, D] 或 [B, 1, D]
#         if out_tensor.dim() == 3:
#             token_out = out_tensor[0, -1, :].detach().cpu().unsqueeze(0)  # [1, D]
#         elif out_tensor.dim() == 2:
#             token_out = out_tensor[-1, :].detach().cpu().unsqueeze(0)     # [1, D]
#         else:
#             return output  # 其它形状直接跳过

#         if sr_cfg["detect_layer"] not in layer_output_cache:
#             layer_output_cache[sr_cfg["detect_layer"]] = []

#         layer_output_cache[sr_cfg["detect_layer"]].append(token_out)

#     if (flag['normal'] % sr_cfg["interval"] == 0 and flag['normal'] != 0):
#         for layer_id in sorted(layer_output_cache.keys()):
#             outputs = layer_output_cache[layer_id]  # List of [1, D]
#             matrix = torch.cat(outputs, dim=0).to(torch.float32)  # [seq_len, dim]

#             total_tokens = matrix.shape[0]
#             start = 0
#             end = min(start + sr_cfg["interval"], total_tokens)
#             sub_matrix = matrix[start:end, :]  # [interval, D] 或最后不足 interval

#             fro_norm_sq = torch.norm(sub_matrix, p='fro').pow(2).item()
#             sigma_max_sq = torch.linalg.norm(sub_matrix, ord=2).pow(2).item()
#             stable_rank = fro_norm_sq / sigma_max_sq if sigma_max_sq > 1e-12 else 0.0

#             layer_output_cache[layer_id] = []  # 清空缓存

#             # 触发逻辑
#             if sr_update_and_should_stop(stable_rank, sr_cfg, sr_state, flag.get("is_rerun", 0)):
#                 raise StopForwardException(
#                     f"[ERROR] Layer {layer_id} 在 Tokens {flag['normal'] - sr_cfg['interval']}-{flag['normal']} 处 "
#                     f"StableRank触发提前终止：sr={stable_rank:.4f}, thr={sr_cfg.get('threshold')}, "
#                     f"cmp={sr_cfg.get('cmp')}, mode={sr_cfg.get('count_mode')}, "
#                     f"k={sr_cfg.get('trigger_k')}, cnt={sr_state.get('value')}."
#                 )

#     return output

def OutputStableRankTest(module, input, output):
    flag['normal'] = flag['normal'] + 1
    out_tensor = output[0] if isinstance(output, tuple) else output

    if isinstance(out_tensor, torch.Tensor):
        # out_tensor: [B, T, D] 或 [B, 1, D]
        if out_tensor.dim() == 3:
            token_out = out_tensor[0, -1, :].detach().cpu().unsqueeze(0)  # [1, D]
        elif out_tensor.dim() == 2:
            token_out = out_tensor[-1, :].detach().cpu().unsqueeze(0)     # [1, D]
        else:
            return output  # 其它形状直接跳过

        if sr_cfg["detect_layer"] not in layer_output_cache:
            layer_output_cache[sr_cfg["detect_layer"]] = []

        layer_output_cache[sr_cfg["detect_layer"]].append(token_out)

    if (flag['normal'] % sr_cfg["interval"] == 0 and flag['normal'] != 0):
        for layer_id in sorted(layer_output_cache.keys()):
            outputs = layer_output_cache[layer_id]  # List of [1, D]

            if len(outputs) == 0:
                continue

            matrix = torch.cat(outputs, dim=0).to(torch.float32)  # [seq_len, dim]

            total_tokens = matrix.shape[0]
            start = 0
            end = min(start + sr_cfg["interval"], total_tokens)
            sub_matrix = matrix[start:end, :]  # [interval, D] 或最后不足 interval

            if sub_matrix.numel() == 0:
                layer_output_cache[layer_id] = []
                continue

            # 不捕获异常，直接判断是否全是有限值
            if not torch.isfinite(sub_matrix).all():
                stable_rank = 0.0
                print(
                    f"[WARN] Non-finite value detected in StableRank input. "
                    f"Layer={layer_id}, Tokens={flag['normal'] - sr_cfg['interval']}-{flag['normal']}. "
                    f"Set stable_rank=0.0."
                )
            else:
                fro_norm_sq = torch.norm(sub_matrix, p='fro').pow(2).item()
                sigma_max_sq = torch.linalg.norm(sub_matrix, ord=2).pow(2).item()
                stable_rank = fro_norm_sq / sigma_max_sq if sigma_max_sq > 1e-12 else 0.0

            layer_output_cache[layer_id] = []  # 清空缓存

            # 触发逻辑
            if sr_update_and_should_stop(stable_rank, sr_cfg, sr_state, flag.get("is_rerun", 0)):
                raise StopForwardException(
                    f"[ERROR] Layer {layer_id} 在 Tokens {flag['normal'] - sr_cfg['interval']}-{flag['normal']} 处 "
                    f"StableRank触发提前终止：sr={stable_rank:.4f}, thr={sr_cfg.get('threshold')}, "
                    f"cmp={sr_cfg.get('cmp')}, mode={sr_cfg.get('count_mode')}, "
                    f"k={sr_cfg.get('trigger_k')}, cnt={sr_state.get('value')}."
                )

    return output

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

def resultoutput(token_length, is_aborted, decoded, idx, prompt, sample, layerid, layertype, posid, f):
    sep = 'assistant\n'
    try:
        response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
    except IndexError:
        response = decoded.strip().replace("\n\n", "\n")

    result_data = {
        "sample_id": idx,
        "token_length": token_length,
        "should_aborted": True if sr_state['value'] >= sr_cfg.get("trigger_k", 3) else False,
        "is_aborted": is_aborted,
        "fault_type": injector.fault_type_str,
        "inject_layer": layerid,
        "inject_kernel": layertype,
        "inject_pos": posid,
        "fault_pe_row": injector.fault_pe_row[0] if injector.fault_pe_row else -1,
        "fault_pe_col": injector.fault_pe_col[0] if injector.fault_pe_col else -1,
        "fault_pe_reg": injector.fault_reg[0] if injector.fault_reg else 0,
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
    parser.add_argument("--outputfile", type=str, default="/workplace/home/mayongzhe/faultinject/projects/runtimemetrics/result", help="输出目录")
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="指定样本ID")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置(row col)，-1 -1表示全阵列随机选1个点")
    parser.add_argument("--reg", type=str, choices=['input', 'weight', 'psum'], default='weight', help="指定注入特定的寄存器")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"], help="数据流类型")
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10", help="注入错误类型")


    parser.add_argument("--interval", type=int, default=50, help="打印注入信息的间隔，默认为50")
    parser.add_argument("--detectLayer", type=int, default=0, help="计算stable rank的层id，默认为0")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum number of new tokens to generate (default: 5000)")
    parser.add_argument("--sr_threshold", type=float, default=1.2, help="StableRank阈值 threshold")
    parser.add_argument("--sr_cmp", type=str, choices=[">=", "<=", ">", "<"], default="<=",
                        help="StableRank异常判断比较符")
    parser.add_argument("--sr_count_mode", type=str, choices=["consecutive", "cumulative", "window"],
                        default="consecutive", help="计数方式：连续/累计/滑动窗口")
    parser.add_argument("--sr_trigger_k", type=int, default=3, help="触发提前终止所需次数K")
    parser.add_argument("--sr_window_n", type=int, default=10, help="window模式窗口大小N（仅window有效）")

    args = parser.parse_args()
    
    ifsample = len(args.sampleid) > 0
    sr_cfg["detect_layer"] = args.detectLayer
    sr_cfg["interval"] = args.interval
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
    samples = random.sample(list(ds), 1000)[0:100]
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

    

    hookoutput = model.model.layers[sr_cfg["detect_layer"]].register_forward_hook(OutputStableRankTest)
    # StableRank detect config
    sr_cfg["threshold"] = args.sr_threshold
    sr_cfg["cmp"] = args.sr_cmp
    sr_cfg["count_mode"] = args.sr_count_mode
    sr_cfg["trigger_k"] = args.sr_trigger_k
    sr_cfg["window_n"] = args.sr_window_n

    print(f"[INFO] StableRank detect policy: thr={sr_cfg['threshold']}, cmp={sr_cfg['cmp']}, "
        f"mode={sr_cfg['count_mode']}, k={sr_cfg['trigger_k']}, window_n={sr_cfg['window_n']}")

    for layerid in args.layerList:
        print(f"[INFO] 注入算子为：{layertype}")   
        # print(f"[INFO] 当前注入配置：{args.injectConfig}, 注入寄存器：{args.reg}, 数据流：{args.dataflow}")
        print(f"[INFO] 当前注入层：{layerid} in {layerlist}")
        injector.print_config()
        sr_tag = f"sr{sr_cfg['cmp']}{sr_cfg['threshold']}"

        if sr_cfg["count_mode"] == "consecutive":
            sr_tag += f"_cons{sr_cfg['trigger_k']}"
        elif sr_cfg["count_mode"] == "cumulative":
            sr_tag += f"_cum{sr_cfg['trigger_k']}"
        elif sr_cfg["count_mode"] == "window":
            sr_tag += f"_win{sr_cfg['trigger_k']}_w{sr_cfg['window_n']}"
        else:
            sr_tag += "_unk"
        output_filename = (
            f"single_bit_"
            f"kernel-{layertype}_"
            f"layer-{layerid}_"
            f"reg-{args.reg}_"
            f"df-{args.dataflow}_"
            f"cfg-{args.injectConfig}_"
            f"{pe_tag}_"
            f"{sr_tag}_detL{sr_cfg['detect_layer']}"
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

                try:
                    sr_state["value"] = 0
                    sr_state["buf"] = None
                    layer_output_cache.clear()
                    faultpos['value'] = 0
                    flag['normal'] = 0
                    flag['is_rerun'] = 0
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
                    resultoutput(token_length = token_length, is_aborted = False, decoded = decoded, idx = sample_index[idx], prompt = prompt, sample = sample, layerid = layerid, layertype = layertype, posid = posid, f = f)

                except StopForwardException as e:
                    # 到达阈值，先保存已有结果，再重新运行一次
                    decoded, clean_ids, bad_ids = safe_decode(tokenizer, generated[0], name="aborted_generated")
                    token_length = generated.shape[1] - num_tokens
                    resultoutput(token_length = token_length, is_aborted = True, decoded = decoded, idx = sample_index[idx], prompt = prompt, sample = sample, layerid = layerid, layertype = layertype, posid = posid, f = f)
                    # pass
                    # 重新运行，重置相关变量
                    sr_state["value"] = 0
                    sr_state["buf"] = None
                    layer_output_cache.clear()
                    faultpos['value'] = 0
                    flag['normal'] = 0
                    flag['is_rerun'] = 1
                    injector.enabled = False  # 先禁用注入，确保重跑阶段不受干扰
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            # top_p=0.95,
                            # temperature=0,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        token_length = outputs[0].shape[0] - num_tokens
                        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    resultoutput(token_length = token_length, is_aborted = False, decoded = decoded, idx = sample_index[idx], prompt = prompt, sample = sample, layerid = layerid, layertype = layertype, posid = posid, f = f)
                    injector.enabled = True  # 重新启用注入
        print(f"已完成生成，结果保存在：{output_path}")
        for handle in handles:
            handle.remove()
    hookoutput.remove()



