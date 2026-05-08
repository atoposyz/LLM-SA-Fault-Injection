import os
import sys
import json
import argparse
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 引入优化后的单比特注入器
from tool.single_bit_injector import SingleBit_Fast_SA_FaultInjector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

# 恢复 Qwen3.5 专属的混合注意力映射字典
pos_mapping = {
    "q": "self_attn.q_proj", 
    "k": "self_attn.k_proj", 
    "v": "self_attn.v_proj", 
    "o": "self_attn.o_proj",
    "mlp-gate": "mlp.gate_proj", 
    "mlp-up": "mlp.up_proj", 
    "mlp-down": "mlp.down_proj",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Bit Precise Fault Injection (Qwen3.5 Hybrid Arch)")
    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="输入错误注入层数，默认为0")
    parser.add_argument("--affect", action='store_true', help="是否开启注入后续层数，默认为False")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(), default="q", help="注入层选项")
    parser.add_argument("--outputfile", type=str, default="./result", help="输出目录")
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="指定样本ID")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置(row col)，-1 -1表示全阵列随机选1个点")
    parser.add_argument("--reg", type=str, choices=['input', 'weight', 'psum'], default='weight', help="指定注入特定的寄存器")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"], help="数据流类型")
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10", help="注入错误类型")
    args = parser.parse_args()

    print(f"[INFO] 注入目标 Kernel 为：{args.layerType}, 层数列表：{args.layerList}")
    
    if not args.outputfile.endswith("/"): args.outputfile += "/"
    out_path = args.outputfile
    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("[INFO] 模型已加载")

    ds = load_dataset("rajpurkar/squad", split="validation")
    random.seed(37)
    samples = random.sample(list(ds), 200)
    
    if args.sampleid:
        samples = [samples[i] for i in args.sampleid]
        sample_index = args.sampleid
    else:
        sample_index = [str(i) for i in range(len(samples))]

    injector = SingleBit_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow=args.dataflow, 
        fault_type=args.injectConfig, precision='bf16'
    )
    
    reg_map = {'input': 0, 'weight': 1, 'psum': 2}
    reg_target_id = reg_map.get(args.reg, 1)

    if args.pe != [-1, -1]:
        injector.set_specific_fault(args.pe[0], args.pe[1])
        injector.fault_reg[0] = reg_target_id 

    layertype = args.layerType
    posid = injector.fault_config.get('pos', 0)
    pe_tag = f"pe{args.pe[0]}_{args.pe[1]}" if args.pe != [-1, -1] else "pe_random"
    affect_tag = "affect1" if args.affect else "affect0"

    for layerid in args.layerList:
        output_filename = (
            f"single_bit_"
            f"kernel-{layertype}_"
            f"layer-{layerid}_"
            f"reg-{args.reg}_"
            f"df-{args.dataflow}_"
            f"cfg-{args.injectConfig}_"
            f"{pe_tag}_"
            f"{affect_tag}.jsonl"
        )
        output_path = os.path.join(out_path, output_filename)

        with open(output_path, "a", encoding="utf-8") as f:
            progress_bar = tqdm(
                samples,
                desc=f"single-bit inject L{layerid}",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            for idx, sample in enumerate(progress_bar):
                handles = []

                if args.pe == [-1, -1]:
                    injector.init_random_fault()
                    injector.fault_reg[0] = reg_target_id

                target_layers = [layerid]
                if args.affect:
                    target_layers = list(range(layerid, min(layerid + 4, model.config.num_hidden_layers)))

                for layernumber in target_layers:
                    current_layertype = layertype
                    if current_layertype in pos_mapping:
                        try:
                            hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[current_layertype])
                            handles.append(hook.register_forward_hook(injector.hook_fn))
                        except AttributeError as e:
                            progress_bar.write(
                                f"[Warning] Hook 挂载失败，层级 {layernumber}，路径 {pos_mapping[current_layertype]}。报错: {e}"
                            )
                            continue

                messages = [{"role": "user", "content": "Read the following passage and answer the question...\n" + sample['context'] + "\nQuestion: " + sample['question'] + "\nAnswer:"}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                num_tokens = inputs["input_ids"].shape[-1]

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, top_p=0.95, temperature=0)

                token_length = outputs[0].shape[0] - num_tokens
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = decoded.split('assistant\n', 1)[-1].strip() if 'assistant\n' in decoded else decoded.strip()

                result_data = {
                    "sample_id": sample_index[idx],
                    "token_length": token_length,
                    "reference_answer": sample['answers'],
                    "generated_answer": response,
                    "fault_type": injector.fault_type_str,
                    "inject_layer": layerid,
                    "inject_kernel": layertype,
                    "inject_pos": posid,
                    "fault_pe_row": injector.fault_pe_row[0] if injector.fault_pe_row else -1,
                    "fault_pe_col": injector.fault_pe_col[0] if injector.fault_pe_col else -1,
                    "fault_pe_reg": injector.fault_reg[0] if injector.fault_reg else 0,
                    "dataflow": injector.dataflow,
                }
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                f.flush()
                progress_bar.set_postfix(
                    layer=layerid,
                    kernel=layertype,
                    pe=f"{injector.fault_pe_row[0] if injector.fault_pe_row else -1},{injector.fault_pe_col[0] if injector.fault_pe_col else -1}",
                    refresh=False,
                )

                for handle in handles:
                    handle.remove()

        print(f"已完成 layer {layerid}，结果保存在：{output_path}")
