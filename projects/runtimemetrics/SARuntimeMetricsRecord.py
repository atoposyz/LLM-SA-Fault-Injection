'''
Runtime Metrics Record

计算每interval个token的多个矩阵指标（StableRank, SVD Entropy, Participation Ratio,
Top-5 Energy Ratio, Frobenius Norm², Spectral Norm², Numerical Rank, Nuclear Norm,
Normalized Nuclear Rank）并记录到数组，不执行rerun，每个样本结束后将指标序列写入输出。
'''
'''
示例命令（对应输出文件 runtime_metrics_record_sid992_kernel-v_layer-35_cfg-input_bitflip_13_reg-input_df-WS_pe_random_intv50_detL0_3_6_9_12_15_18_21_24_27_30_33_affect0_inject.jsonl）：
uv run python SARuntimeMetricsRecord.py \
    --layerList 35 \
    --layerType v \
    --reg input \
    --dataflow WS \
    --injectConfig input_bitflip_13 \
    --interval 50 \
    --detectLayers 0 3 6 9 12 15 18 21 24 27 30 33
'''
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from struct import pack, unpack
import time
import torch
import random
import numpy as np
import os
import argparse
import sys
import json

from tool.single_bit_injector import SingleBit_Fast_SA_FaultInjector
from tool.runtime_metrics import compute_runtime_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.set_verbosity_error()

MAX_float_16 = 65504
calculateSize = 16
headSize = 128

inject_pos = {'value': 0}
selectHead = {'value': 0}
block = 8
blockX = 9
blockY = 9
flag = {'firstinit': 0, 'value': 0, 'normal': 0}

# Runtime metrics config
rm_cfg = {
    "interval": 50,
    "detect_layers": [0],
}

faultin = {'value': 0}
set_fault = {"fx": 0, "fy": 0, "bx": 0, "by": 0}
layer_output_cache = {}

faultpos = {'value': 0}


def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def make_rm_hook(layer_id: int):
    """为指定 layer_id 创建一个 hook，仅累加该层输出到缓存"""
    def hook(module, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output

        if isinstance(out_tensor, torch.Tensor):
            if out_tensor.dim() == 3:
                token_out = out_tensor[0, -1, :].detach().cpu().unsqueeze(0)
            elif out_tensor.dim() == 2:
                token_out = out_tensor[-1, :].detach().cpu().unsqueeze(0)
            else:
                return output

            if layer_id not in layer_output_cache:
                layer_output_cache[layer_id] = []

            layer_output_cache[layer_id].append(token_out)

        return output
    return hook


def safe_decode(tokenizer, token_ids_or_tensor, name="generated"):
    if isinstance(token_ids_or_tensor, torch.Tensor):
        token_ids = token_ids_or_tensor.detach().cpu().tolist()
    else:
        token_ids = token_ids_or_tensor

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


def resultoutput(token_length, decoded, idx, prompt, sample, layerid, layertype, posid, f, metrics_values, inference_time):
    sep = 'assistant\n'
    try:
        response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
    except IndexError:
        response = decoded.strip().replace("\n\n", "\n")

    result_data = {
        "sample_id": idx,
        "token_length": token_length,
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
        "runtime_metrics_values": metrics_values,
        "inference_time_s": round(inference_time, 2),
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

    parser.add_argument("--no_inject", action='store_true', help="关闭故障注入，仅记录无错误情况下的指标变化")
    parser.add_argument("--interval", type=int, default=50, help="计算Runtime Metrics的间隔，默认为50")
    parser.add_argument("--detectLayers", type=int, nargs='+', default=[0], help="计算runtime metrics的层id列表，默认为[0]")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum number of new tokens to generate (default: 5000)")

    args = parser.parse_args()

    ifsample = len(args.sampleid) > 0
    rm_cfg["detect_layers"] = args.detectLayers
    rm_cfg["interval"] = args.interval
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

    ds = load_dataset("openai/gsm8k", "main", split="train")
    random.seed(37)
    samples = random.sample(list(ds), 1000)[0:1000]
    print("[INFO] 数据集已加载")

    if sample_index:
        print(f"[INFO] 指定样本ID：{sample_index}")
        try:
            samples = [samples[i] for i in sample_index]
        except IndexError as e:
            raise ValueError(f"指定的样本ID越界：{e}")
    else:
        print("[INFO] 未指定样本ID，使用随机选择的全部样本进行注错测试")
        sample_index = [str(i) for i in range(len(samples))]

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
    inject_tag = "noinject" if args.no_inject else "inject"
    sample_tag = f"sid{'_'.join(str(s) for s in args.sampleid)}" if args.sampleid else "sidall"

    rm_hooks = []
    for dl in rm_cfg["detect_layers"]:
        rm_hooks.append(model.model.layers[dl].register_forward_hook(make_rm_hook(dl)))

    print(f"[INFO] Runtime Metrics record: interval={rm_cfg['interval']}, detect_layers={rm_cfg['detect_layers']}")

    for layerid in args.layerList:
        print(f"[INFO] 注入算子为：{layertype}")
        print(f"[INFO] 当前注入层：{layerid} in {layerlist}")
        injector.print_config()

        output_filename = (
            f"runtime_metrics_record_"
            f"{sample_tag}_"
            f"kernel-{layertype}_"
            f"layer-{layerid}_"
            f"cfg-{args.injectConfig}_"
            f"reg-{args.reg}_"
            f"df-{args.dataflow}_"
            f"{pe_tag}_"
            f"intv{rm_cfg['interval']}_detL{'_'.join(str(d) for d in rm_cfg['detect_layers'])}_"
            f"{affect_tag}_"
            f"{inject_tag}.jsonl"
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

        with open(output_path, "a", encoding="utf-8") as f:
            for idx, sample in enumerate(tqdm(samples)):
                if args.pe == [-1, -1]:
                    injector.init_random_fault()
                injector.enabled = not args.no_inject

                messages = [
                    {
                        'role': 'user',
                        'content':  "Solve the following math problem step by step.\n"
                        + sample["question"]
                        + '''
IMPORTANT: Do NOT just solve the problem. You are creating a comprehensive 
mathematics lesson plan based on this problem. Include:

1. Full solution (200 tokens)
2. 5 variations of the problem with solutions (500 tokens)
3. Pedagogical explanation for each step (600 tokens)
4. Common student errors and how to address them (500 tokens)
5. Real-world applications of this math concept (600 tokens)
6. Practice problems for students (400 tokens)
7. Grading rubric and answer key (300 tokens)
8. Extension activities for advanced students (400 tokens)
9. Visual representation description (300 tokens)
10. Cross-curricular connections (200 tokens)

Total must exceed 4000 tokens.
                        '''
                        + "\nAnswer:"
                    }
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                tokenized = tokenizer(prompt, return_tensors="pt")
                num_tokens = tokenized["input_ids"].shape[-1]

                snapshots = []
                layer_output_cache.clear()
                faultpos['value'] = 0

                t0 = time.time()
                generated = inputs["input_ids"]
                past_key_values = None

                with torch.no_grad():
                    for step in range(max_new_tokens):
                        if past_key_values is None:
                            model_inputs = {"input_ids": generated}
                        else:
                            model_inputs = {
                                "input_ids": generated[:, -1:],
                                "past_key_values": past_key_values,
                                "use_cache": True,
                            }

                        outputs = model(**model_inputs)
                        past_key_values = outputs.past_key_values
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=-1)

                        passes_completed = step + 1
                        if passes_completed % rm_cfg["interval"] == 0:
                            for layer_id in sorted(layer_output_cache.keys()):
                                outputs_list = layer_output_cache[layer_id]
                                if len(outputs_list) == 0:
                                    continue
                                matrix = torch.cat(outputs_list, dim=0).to(torch.float32)
                                total_tokens = matrix.shape[0]
                                end = min(rm_cfg["interval"], total_tokens)
                                sub_matrix = matrix[:end, :].clone()

                                snapshots.append({
                                    "token_start": passes_completed - rm_cfg["interval"],
                                    "token_end": passes_completed,
                                    "layer": layer_id,
                                    "matrix": sub_matrix,
                                })

                                layer_output_cache[layer_id] = []

                        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                            break

                # 保存最后一个不完整窗口
                passes_completed = step + 1
                for layer_id in sorted(layer_output_cache.keys()):
                    outputs_list = layer_output_cache[layer_id]
                    if len(outputs_list) == 0:
                        continue
                    matrix = torch.cat(outputs_list, dim=0).to(torch.float32).clone()
                    last_interval_end = (passes_completed // rm_cfg["interval"]) * rm_cfg["interval"]
                    snapshots.append({
                        "token_start": last_interval_end,
                        "token_end": passes_completed,
                        "layer": layer_id,
                        "matrix": matrix,
                    })
                    layer_output_cache[layer_id] = []

                # 推理结束后统一计算所有快照的 Runtime Metrics
                metrics_values = []
                for snap in snapshots:
                    metrics = compute_runtime_metrics(snap["matrix"])
                    del snap["matrix"]

                    if metrics is None:
                        metrics_values.append({
                            "token_start": snap["token_start"],
                            "token_end": snap["token_end"],
                            "layer": snap["layer"],
                            "stable_rank": 0.0,
                            "svd_entropy": 0.0,
                            "participation_ratio": 0.0,
                            "top5_energy_ratio": 0.0,
                            "frobenius_norm_sq": 0.0,
                            "spectral_norm_sq": 0.0,
                            "numerical_rank": 0,
                            "nuclear_norm": 0.0,
                            "normalized_nuclear_rank": 0.0,
                        })
                    else:
                        metrics_values.append({
                            "token_start": snap["token_start"],
                            "token_end": snap["token_end"],
                            "layer": snap["layer"],
                            "stable_rank": round(metrics["stable_rank"], 6),
                            "svd_entropy": round(metrics["svd_entropy"], 6),
                            "participation_ratio": round(metrics["participation_ratio"], 6),
                            "top5_energy_ratio": round(metrics["top5_energy_ratio"], 6),
                            "frobenius_norm_sq": round(metrics["frobenius_norm_sq"], 6),
                            "spectral_norm_sq": round(metrics["spectral_norm_sq"], 6),
                            "numerical_rank": metrics["numerical_rank"],
                            "nuclear_norm": round(metrics["nuclear_norm"], 6),
                            "normalized_nuclear_rank": round(metrics["normalized_nuclear_rank"], 6),
                        })

                # 累积 frobenius_norm_sq / nuclear_norm / numerical_rank（按层）
                cum_state = {}
                for m in metrics_values:
                    lid = m["layer"]
                    if lid not in cum_state:
                        cum_state[lid] = {"frobenius_norm_sq": 0.0, "nuclear_norm": 0.0, "numerical_rank": 0}
                    cum_state[lid]["frobenius_norm_sq"] += m["frobenius_norm_sq"]
                    cum_state[lid]["nuclear_norm"] += m["nuclear_norm"]
                    cum_state[lid]["numerical_rank"] = max(cum_state[lid]["numerical_rank"], m["numerical_rank"])
                    m["frobenius_norm_sq"] = round(cum_state[lid]["frobenius_norm_sq"], 1)
                    m["nuclear_norm"] = round(cum_state[lid]["nuclear_norm"], 1)
                    m["numerical_rank"] = cum_state[lid]["numerical_rank"]

                t1 = time.time()
                decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
                token_length = generated.shape[1] - num_tokens
                resultoutput(token_length=token_length, decoded=decoded, idx=sample_index[idx],
                             prompt=prompt, sample=sample, layerid=layerid, layertype=layertype,
                             posid=posid, f=f, metrics_values=metrics_values,
                             inference_time=t1 - t0)

                print(f"[RM] Sample {sample_index[idx]}: {len(metrics_values)} metric points recorded "
                      f"time={t1 - t0:.1f}s, "
                      f"across layers {rm_cfg['detect_layers']}, "
                      f"values={[(v['layer'], v['stable_rank'], v['svd_entropy'], v['numerical_rank']) for v in metrics_values]}")

        print(f"已完成生成，结果保存在：{output_path}")
        for handle in handles:
            handle.remove()
    for h in rm_hooks:
        h.remove()
