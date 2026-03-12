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

# 引入优化后的 BER 注入器
from tool.ber_injector import BER_Fast_SA_FaultInjector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

# 适配标准 Qwen3.5 架构的映射
pos_mapping = {
    "q": "self_attn.q_proj", 
    "k": "self_attn.k_proj", 
    "v": "self_attn.v_proj", 
    "o": "self_attn.o_proj",
    "qkv": "linear_attn.in_proj_qkv", # Linear Attention
    "mlp-gate": "mlp.gate_proj", 
    "mlp-up": "mlp.up_proj", 
    "mlp-down": "mlp.down_proj",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BER Massive Fault Injection")
    parser.add_argument("--outputfile", type=str, default="./result", help="输出目录")
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="指定样本ID")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"], help="默认数据流类型")
    parser.add_argument("--ber", type=int, default=1, help="注入的错误数量（对应原始逻辑中的绝对数量）")
    parser.add_argument("--fixReg", type=str, choices=['mixed', 'input', 'weight', 'psum'], default='mixed', help="混合注入寄存器")
    parser.add_argument("--fixDataflow", type=str, choices=['random', 'WS', 'OS', 'IS'], default='random', help="混合数据流")
    args = parser.parse_args()
    
    if not args.outputfile.endswith("/"): args.outputfile += "/"
    out_path = args.outputfile
    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", device_map="auto", trust_remote_code=True)
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

    # 初始化 BER Injector (使用 mixed 占位符)
    injector = BER_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow=args.dataflow, 
        fault_type="random_bitflip_mixed", precision='fp32'
    )
    reg_map = {'input': 0, 'weight': 1, 'psum': 2}

    output_path = f"{out_path}ber_layer_random_reg{args.fixReg}_df{args.fixDataflow}.jsonl"

    # 将文件打开操作移出循环，优化 IO 性能
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(samples)):
            handles = []
            
            # 1. 随机化故障类型 (bitflip/stuck_0/stuck_1)
            f_type = random.choice(["bitflip", "stuck_0", "stuck_1"])
            injector.fault_type_str = f"random_{f_type}_mixed"
            injector.parse_fault_type()
            
            # 2. 随机或固定数据流
            injector.dataflow = random.choice(["WS", "OS", "IS"]) if args.fixDataflow == 'random' else args.fixDataflow
            
            # 3. 大面积撒错逻辑 (BER)
            if args.fixReg == 'mixed':
                injector.init_multi_fault_positions(num_faults=args.ber, num_regs=3, num_bits=32)
            else:
                reg_id = reg_map.get(args.fixReg, 1)
                injector.init_multi_fault_positions(num_faults=args.ber, num_regs=1, num_bits=32)
                injector.fault_reg = [reg_id for _ in injector.fault_reg]

            # 4. Qwen3.5 专属 3:1 精准挂载钩子
            target_layers = list(range(model.config.num_hidden_layers))
            for layernumber in target_layers:
                # 首先挂载所有层通用的 MLP 模块
                for mlp_kernel in ["mlp-gate", "mlp-up", "mlp-down"]:
                    try:
                        hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[mlp_kernel])
                        handles.append(hook.register_forward_hook(injector.hook_fn))
                    except AttributeError:
                        continue
                
                # 然后根据 3:1 架构精准挂载 Attention 模块
                if layernumber % 4 == 3:
                    # Self Attention 层
                    attn_kernels = ["q", "k", "v", "o"]
                else:
                    # Linear Attention 层
                    attn_kernels = ["qkv", "o"] # 注意: 聚合了 qkv, 但通常 o_proj 依然独立
                
                for attn_kernel in attn_kernels:
                    try:
                        hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[attn_kernel])
                        handles.append(hook.register_forward_hook(injector.hook_fn))
                    except AttributeError:
                        continue

            # 5. 推理生成
            messages = [{"role": "user", "content": "Read the following passage and answer the question...\n" + sample['context'] + "\nQuestion: " + sample['question'] + "\nAnswer:"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            num_tokens = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, top_p=0.95, temperature=0)
            
            token_length = outputs[0].shape[0] - num_tokens
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 安全提取防崩溃
            response = decoded.split('assistant\n', 1)[-1].strip() if 'assistant\n' in decoded else decoded.strip()

            # 6. JSON 记录 (修复列表爆栈问题)
            result_data = {
                "sample_id": sample_index[idx],
                "token_length": token_length,
                "reference_answer": sample['answers'],
                "generated_answer": response,
                "fault_type": injector.fault_type_str,
                "inject_layer": "all",
                "inject_kernel": "all",
                "fault_count": args.ber, # 记录注入了多少个错误
                "fault_pe_reg": args.fixReg, # 用宏观策略名替代上万个元素的 list 记录
                "dataflow": injector.dataflow,
            }
            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            f.flush()
            
            for handle in handles:
                handle.remove()

    print(f"已完成大规模注入生成，结果保存在：{output_path}")