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

logging.set_verbosity_error()

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
    "threshold": 2.0,            # StableRank阈值
    "cmp": ">=",                 # 比较符: >= <= > <
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

pos_info = {
    'faultPosInCul_X': 0,
    'faultPosInCul_Y': 0,
    'blockPos_X': 0,
    'blockPos_Y': 0,
}
faultin = { 'value': 0}
set_fault = {"fx": 0, "fy": 0, "bx": 0, "by": 0}
layer_output_cache = {}

def Fp32Bitflip(data, pos):
    fs = pack('f', data)
    bval = list(unpack('BBBB', fs))
    q, r = divmod(pos, 8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew = unpack('f', fs)[0]
    if isnan(fnew) or isinf(fnew):
        fnew = 1.0 if data > 0 else 0.0
    if abs(fnew) > 1e4:
        fnew = 1.0 if data > 0 else -1.0
    if data >= 0 and fnew < 0:
        fnew = 0.0

    return fnew

def BFloat16Bitflip(data: torch.Tensor, pos: int) -> torch.Tensor:
    assert data.dtype == torch.bfloat16 and data.numel() == 1, "data must be a scalar tensor of dtype torch.bfloat16"
    int_repr = data.view(torch.uint16).item()
    int_repr ^= 1 << pos  # pos in [0, 15]
    flipped_tensor = torch.tensor(int_repr, dtype=torch.uint16).view(torch.bfloat16)

    return flipped_tensor

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


def Matrix_FaultIn(module, input, output):
    faultsquare = 1
    
    xSize = output[0].shape[0]
    ySize = output[0].shape[1]
    y_max = ySize / calculateSize
    x_max = xSize / calculateSize
    if flag['firstinit'] == 0:

        if faultin['value'] == 0:
            # 随机生成注入位置
            faultPosInCul_X = random.randint(0, calculateSize - faultsquare)
            faultPosInCul_Y = random.randint(0, calculateSize - faultsquare)
            blockPos_X = random.randint(0, max(int(x_max - blockX), 0))
            blockPos_Y = random.randint(0, int(y_max - blockY))
        else:
            faultPosInCul_X = set_fault['fx']
            faultPosInCul_Y = set_fault['fy'] 
            blockPos_X = set_fault['bx'] 
            blockPos_Y = set_fault['by'] 
        # print(f"faultpos['value']: {faultpos['value']}")
        if faultpos['value'] == 0:
            pos_info['faultPosInCul_X'] = faultPosInCul_X
            pos_info['faultPosInCul_Y'] = faultPosInCul_Y
            pos_info['blockPos_X'] = blockPos_X
            pos_info['blockPos_Y'] = blockPos_Y
            faultpos['value'] = 1
        # if y_max >= blockY and x_max >= blockX and flag['firstinit'] == 0:
        for i in range(blockX if x_max >= blockX else int(x_max)):
            for j in range(blockY if y_max >= blockY else int(y_max)):
                for fx in range(faultsquare):
                    for fy in range(faultsquare):
                        x = (blockPos_X + i) * calculateSize + faultPosInCul_X + fx
                        y = (blockPos_Y + j) * calculateSize + faultPosInCul_Y + fy

                        output[0][x][y] = BFloat16Bitflip(output[0][x][y], inject_pos['value']) 
        flag['firstinit'] = 1
    return output

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
            matrix = torch.cat(outputs, dim=0).to(torch.float32)  # [seq_len, dim]

            total_tokens = matrix.shape[0]
            start = 0
            end = min(start + sr_cfg["interval"], total_tokens)
            sub_matrix = matrix[start:end, :]  # [interval, D] 或最后不足 interval

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



def resultoutput(token_length, is_aborted, decoded, idx, prompt, sample, f):
    sep = 'assistant\n'
    try:
        response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
    except IndexError:
        response = decoded.strip().replace("\n\n", "\n")

    result_data = {
        "sample_id": idx,
        "block_pos": {
            "x": pos_info['blockPos_X'],
            "y": pos_info['blockPos_Y']
        },
        "fault_pos_in_blk": {
            "x": pos_info['faultPosInCul_X'],
            "y": pos_info['faultPosInCul_Y']
        },
        "token_length": token_length,
        "should_aborted": True if sr_state['value'] >= sr_cfg.get("trigger_k", 3) else False,
        "is_aborted": is_aborted,
        "reference_answer": sample['answer'],
        "generated_answer": response.split('</think>', 1)[1].strip() if '</think>' in decoded else "",
        "thinking_process": response.split('</think>', 1)[0].strip() if '</think>' in decoded else response,
        "prompt": prompt,                        
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

def main(argv=None):
    
    # 参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="输入错误注入层数[0-35]，默认为0")
    parser.add_argument("--pos", type=int, nargs='+', default=[11,12,13,14], help="输入错误注入位置，默认为空，位置之间用空格分隔")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(), default="q",
                        help="注入层选项：q, k, v, ffn-gate, ffn-up, ffn-down, attention-norm, input-norm")
    parser.add_argument("--outputfile", type=str, default="./output/", help="输入注错结果存放目录")
    parser.add_argument("--run", type=int, default=1, help="选择的样本运行次数，默认为1次")
    parser.add_argument("--faultin", type=int, nargs=4, default=[], help="输入4个注入位置参数，依次为 fx fy bx by，默认为全随机值" )
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="在随机选择的1000个样本中指定样本ID，默认为1000个全部进行注错测试")
    parser.add_argument("--interval", type=int, default=50, help="打印注入信息的间隔，默认为50")
    parser.add_argument("--detectLayer", type=int, default=0, help="计算stable rank的层id，默认为0")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum number of new tokens to generate (default: 5000)")
    parser.add_argument("--sr_threshold", type=float, default=1.3, help="StableRank阈值 threshold")
    parser.add_argument("--sr_cmp", type=str, choices=[">=", "<=", ">", "<"], default="<=",
                        help="StableRank异常判断比较符")
    parser.add_argument("--sr_count_mode", type=str, choices=["consecutive", "cumulative", "window"],
                        default="consecutive", help="计数方式：连续/累计/滑动窗口")
    parser.add_argument("--sr_trigger_k", type=int, default=3, help="触发提前终止所需次数K")
    parser.add_argument("--sr_window_n", type=int, default=10, help="window模式窗口大小N（仅window有效）")

    args = parser.parse_args(argv)
    faultin['value'] = 1 if len(args.faultin) == 4 else 0
    ifsample = len(args.sampleid) > 0
    sr_cfg["detect_layer"] = args.detectLayer
    sr_cfg["interval"] = args.interval
    layertype = args.layerType
    print(f"[INFO] 注入层为：{layertype}")    
    
    if faultin['value'] == 1:
        set_fault['fx'] = args.faultin[0] if args.faultin else 0
        set_fault['fy'] = args.faultin[1] if len(args.faultin) > 1 else 0
        set_fault['bx'] = args.faultin[2] if len(args.faultin) > 2 else 0
        set_fault['by'] = args.faultin[3] if len(args.faultin) > 3 else 0
        print(f"[INFO] 注入位置参数设置：fx={set_fault['fx']}, fy={set_fault['fy']}, bx={set_fault['bx']}, by={set_fault['by']}")
    else:
        print("[INFO] 注入位置参数设置为随机值")
        
    max_new_tokens = args.max_tokens
    poslist = args.pos
    layerlist = args.layerList
    print(f"[INFO] 注入位置列表为：{poslist}")
    print(f"[INFO] 注入层列表为：{layerlist}")
    if not args.outputfile.endswith("/"):
        args.outputfile += "/"
    out_path = args.outputfile
    runtimes = args.run
    sample_index = args.sampleid
    print(f"[INFO] 输出文件路径为：{out_path}")
    print(f"[INFO] 注错次数：{runtimes}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"[INFO] 模型已加载")
    # 加载数据集
    ds = load_dataset("TheFinAI/CONVFINQA_test_test", split="test")
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

    hookoutput = model.model.layers[sr_cfg["detect_layer"]].register_forward_hook(OutputStableRankTest)
    # StableRank detect config
    sr_cfg["threshold"] = args.sr_threshold
    sr_cfg["cmp"] = args.sr_cmp
    sr_cfg["count_mode"] = args.sr_count_mode
    sr_cfg["trigger_k"] = args.sr_trigger_k
    sr_cfg["window_n"] = args.sr_window_n

    print(f"[INFO] StableRank detect policy: thr={sr_cfg['threshold']}, cmp={sr_cfg['cmp']}, "
        f"mode={sr_cfg['count_mode']}, k={sr_cfg['trigger_k']}, window_n={sr_cfg['window_n']}")

    for posid in poslist:
        for layerid in layerlist:
            
            hook = get_module_by_path(model.model.layers[layerid], pos_mapping[layertype])
            for runid in range(runtimes):
                print(f"[INFO] 注入层为：{layertype}")   
                print(f"[INFO] 当前注入位置：{posid} in {poslist}")
                print(f"[INFO] 当前注入层：{layerid} in {layerlist}")
                print(f"[INFO] 运行次数：{runid + 1}/{runtimes}")
                inject_pos['value'] = posid
                sr_tag = f"sr{sr_cfg['cmp']}{sr_cfg['threshold']}"

                if sr_cfg["count_mode"] == "consecutive":
                    sr_tag += f"_cons{sr_cfg['trigger_k']}"
                elif sr_cfg["count_mode"] == "cumulative":
                    sr_tag += f"_cum{sr_cfg['trigger_k']}"
                elif sr_cfg["count_mode"] == "window":
                    sr_tag += f"_win{sr_cfg['trigger_k']}_w{sr_cfg['window_n']}"
                else:
                    sr_tag += "_unk"
                if faultin['value'] == 1:
                    output_path = out_path + str(layerid) + "_" + layertype + "_" + str(posid) + "_" + str(set_fault['fx']) + "_" + str(set_fault['fy']) + "_" + str(set_fault['bx']) + "_" + str(set_fault['by']) + ".jsonl"
                else:
                    output_path = f"{out_path}{str(layerid)}_{layertype}_{str(posid)}_{sr_tag}_detL{sr_cfg['detect_layer']}_rerun.jsonl"
                hookRegister = hook.register_forward_hook(Matrix_FaultIn)

                # 生成文本并写入
                with open(output_path, "a", encoding="utf-8") as f:
                    for idx, sample in enumerate(tqdm(samples)):
                        messages = [
                            {
                                'role': 'user',
                                'content': "You are a financial reasoning assistant. Read the following passage and answer the final question. \nExplain your reasoning step by step. \nThen you **MUST** give the final answer in the format: Answer: /box{your answer}\n" + sample['query'].replace("\n\n", "\n") + "Answer the last question."
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
                            flag['firstinit'] = 0
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
                            resultoutput(token_length = token_length, is_aborted = False, decoded = decoded, idx = sample_index[idx], prompt = prompt, sample = sample, f = f)

                        except StopForwardException as e:
                            # 到达阈值，先保存已有结果，再重新运行一次
                            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
                            token_length = len(generated[0]) - num_tokens
                            resultoutput(token_length = token_length, is_aborted = True, decoded = decoded, idx = sample_index[idx], prompt = prompt, sample = sample, f = f)
                            # 重新运行，重置相关变量
                            hookRegister.remove()
                            sr_state["value"] = 0
                            sr_state["buf"] = None
                            layer_output_cache.clear()
                            faultpos['value'] = 0
                            flag['normal'] = 0
                            flag['firstinit'] = 0
                            flag['is_rerun'] = 1
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
                            resultoutput(token_length = token_length, is_aborted = False, decoded = decoded, idx = sample_index[idx], prompt = prompt, sample = sample, f = f)
                            hookRegister = hook.register_forward_hook(Matrix_FaultIn)
                print(f"已完成生成，结果保存在：{output_path}")
                hookRegister.remove()
    hookoutput.remove()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

