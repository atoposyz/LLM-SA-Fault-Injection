import os
import sys
import json
import argparse
import time
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from tool.single_bit_injector import SingleBit_Fast_SA_FaultInjector

device = torch.device("cuda:0")

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

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
    parser = argparse.ArgumentParser(description="Simplified Single-Bit Fault Injection (token + latency only)")
    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="注入层数列表")
    parser.add_argument("--affect", action='store_true', help="是否影响后续层")
    parser.add_argument("--layerType", type=str, choices=pos_mapping.keys(), default="q")
    parser.add_argument("--outputfile", type=str, default="./result_simple", help="输出目录")
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="指定样本ID")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定PE位置，-1 -1表示随机")
    parser.add_argument("--reg", type=str, choices=['input', 'weight', 'psum'], default='weight')
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"])
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10")
    args = parser.parse_args()

    print(f"[INFO] 注入目标: kernel={args.layerType}, layers={args.layerList}")
    print(f"[INFO] 使用 GPU: cuda:0")

    if not args.outputfile.endswith("/"):
        args.outputfile += "/"
    out_path = args.outputfile
    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        device_map={"": "cuda:1"},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("[INFO] 模型已加载")

    ds = load_dataset("openai/gsm8k", "main", split="train")
    random.seed(37)
    samples = random.sample(list(ds), 1000)[100:1000]

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
    injector.fault_reg.append(reg_target_id)
    if args.pe != [-1, -1]:
        injector.set_specific_fault(args.pe[0], args.pe[1])
    else:
        injector.init_random_fault()

    layertype = args.layerType
    pe_tag = f"pe{args.pe[0]}_{args.pe[1]}" if args.pe != [-1, -1] else "pe_random"
    affect_tag = "affect1" if args.affect else "affect0"

    for layerid in args.layerList:
        output_filename = (
            f"simple_"
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
                desc=f"simple-inject L{layerid}",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            for idx, sample in enumerate(progress_bar):
                handles = []

                if args.pe == [-1, -1]:
                    injector.init_random_fault()

                target_layers = [layerid]
                if args.affect:
                    target_layers = list(range(layerid, model.config.num_hidden_layers))

                for layernumber in target_layers:
                    current_layertype = layertype
                    if current_layertype in pos_mapping:
                        hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[current_layertype])
                        handles.append(hook.register_forward_hook(injector.hook_fn))

                messages = [{"role": "user", "content": "Solve the following math problem step by step.\n" + sample['question'] + "\nAnswer:"}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                num_tokens = inputs["input_ids"].shape[-1]

                t0 = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5000, do_sample=False, top_p=0.95, temperature=0)
                elapsed = time.time() - t0

                token_length = outputs[0].shape[0] - num_tokens

                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = decoded.split('assistant\n', 1)[-1].strip() if 'assistant\n' in decoded else decoded.strip()

                result_data = {
                    "sample_id": sample_index[idx],
                    "token_length": token_length,
                    "inference_time_s": round(elapsed, 4),
                    "generated_answer": response,
                }
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                f.flush()

                for handle in handles:
                    handle.remove()

        print(f"[DONE] layer {layerid} -> {output_path}")
