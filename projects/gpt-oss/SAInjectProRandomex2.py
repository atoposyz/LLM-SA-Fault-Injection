import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, logging

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 引入 vLLM-Hook
from vllm_hook_plugins import HookLLM

# 保留位置映射字典
pos_mapping = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "qkv": "linear_attn.in_proj_qkv",
    "mlp-gate": "mlp.gate_proj",
    "mlp-up": "mlp.up_proj",
    "mlp-down": "mlp.down_proj",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="输入错误注入层数，默认为0")
    parser.add_argument("--affect", action='store_true', help="是否开启注入后续层数，默认为False")
    parser.add_argument("--pos", type=int, nargs='+', default=[11,12,13,14], help="输入错误注入位置，默认为空，位置之间用空格分隔")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(), help="注入层选项")
    parser.add_argument("--outputfile", type=str, default="projects/gpt-oss/result", help="输入注错结果存放目录")
    parser.add_argument("--run", type=int, default=1, help="选择的样本运行次数，默认为1次")
    parser.add_argument("--faultin", type=int, nargs=4, default=[], help="输入4个注入位置参数，依次为 fx fy bx by，默认为全随机值" )
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="在随机选择的1000个样本中指定样本ID")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置，格式为 row col")
    parser.add_argument("--reg", type=str, choices=['input', 'weight', 'psum'], default='weight', help="指定注入特定的寄存器")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"], help="指定固定的数据流类型，默认为WS")
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10", help="注入错误类型")
    parser.add_argument("--random", action='store_true', help="开启随机模式")
    parser.add_argument("--ber", type=int, default=1, help="设置随机错误注入的错误率")
    parser.add_argument("--fixReg", type=str, choices=['mixed', 'input', 'weight', 'psum'], default='mixed')
    parser.add_argument("--fixDataflow", type=str, choices=['random', 'WS', 'OS', 'IS'], default='random')

    args = parser.parse_args()

    layertype = args.layerType if args.layerType else "q"
    print(f"[INFO] 注入层为：{layertype}")    
    
    if not args.outputfile.endswith("/"):
        args.outputfile += "/"
    out_path = args.outputfile

    # 1. 构建传递给 Worker 的配置字典
    fault_config = {
        "sa_rows": 256,
        "sa_cols": 256,
        "layerList": args.layerList,
        "affect": args.affect,
        "layerType": layertype,
        "pe": args.pe,
        "reg": args.reg,
        "dataflow": args.dataflow,
        "injectConfig": args.injectConfig,
        "random": args.random,
        "ber": args.ber,
        "fixReg": args.fixReg,
        "fixDataflow": args.fixDataflow,
        "faultin": args.faultin if args.faultin else []
    }
    config_file = "/tmp/vllm_fault_config.json"
    with open(config_file, "w") as f:
        json.dump(fault_config, f)
    
    # 将配置文件路径放入环境变量，供 Worker 读取
    os.environ["VLLM_FAULT_CONFIG"] = config_file

    model_id = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 2. 初始化 HookLLM (使用我们自定义的 fault_inject worker)
    print(f"[INFO] 正在通过 vLLM 加载模型...")
    llm = HookLLM(
        model=model_id,
        worker_name="fault_inject", 
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1
    )
    print(f"[INFO] 模型已加载")

    # 加载数据集
    ds = load_dataset("rajpurkar/squad", split="validation")
    random.seed(37)
    samples = random.sample(list(ds), 200)

    sample_index = args.sampleid
    if sample_index:
        samples = [samples[i] for i in sample_index]
    else:
        sample_index = [str(i) for i in range(len(samples))]

    # 确定输出文件名
    if args.random:
        suffix = f"random_reg{args.fixReg}_df{args.fixDataflow}"
    else:
        suffix = layertype
    output_path = f"{out_path}all_layer_{suffix}.jsonl"

    # 3. 运行推理循环
    for idx, sample in enumerate(tqdm(samples)):
        messages = [{
            'role': 'user',
            'content': "Read the following passage and answer the question with a text span directly from the passage.\n" + sample['context'] + "\nQuestion: " + sample['question'] + "\nAnswer:"
        }]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        tokenized = tokenizer(prompt, return_tensors="pt")
        num_tokens = tokenized["input_ids"].shape[-1]

        # 调用 vLLM 进行生成
        outputs = llm.generate(
            prompt, 
            use_hook=True, 
            max_tokens=200,
            temperature=0,
            top_p=0.95
        )
        
        # 解析 vLLM 输出
        generated_text = outputs[0].outputs[0].text
        output_token_ids = outputs[0].outputs[0].token_ids
        token_length = len(output_token_ids)

        sep = 'assistant\n'
        try:
            response = generated_text.split(sep, 1)[1].strip().replace("\n\n", "\n")
        except IndexError:
            response = generated_text.strip().replace("\n\n", "\n")

        # 记录结果 (注意：由于 injector 在 worker 进程中，这里的部分日志参数采用 args 配置替代)
        result_data = {
            "sample_id": sample_index[idx],
            "token_length": token_length,
            "reference_answer": sample['answers'],
            "generated_answer": response.split('</think>', 1)[1].strip() if '</think>' in generated_text else response,
            "fault_type": "random" if args.random else args.injectConfig,
            "inject_layer": "all" if args.random else args.layerList[0],
            "inject_kernel": "all" if args.random else layertype,
            "dataflow": "random" if args.fixDataflow == "random" else args.dataflow,
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

    print(f"已完成生成，结果保存在：{output_path}")