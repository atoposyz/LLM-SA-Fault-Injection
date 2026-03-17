'''

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

logging.set_verbosity_error()

import argparse
import sys
import json

import os
from tool.fault_injector_next import Fast_SA_FaultInjector

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)
# from tool.fault_injector_next import Fast_SA_FaultInjector

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_float_16 = 65504
calculateSize = 16
headSize = 128

inject_pos = {'value': 0}
selectHead = {'value': 0}
block = 8
blockX = 9
blockY = 9
flag = {'value': 0, 'normal': 0}
pos_info = {
    'faultPosInCul_X': 0,
    'faultPosInCul_Y': 0,
    'blockPos_X': 0,
    'blockPos_Y': 0,
}
faultin = { 'value': 0 }
set_fault = {"fx": 0, "fy": 0, "bx": 0, "by": 0}



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

    # 转成 uint16 表示，获取底层比特
    int_repr = data.view(torch.uint16).item()

    # 执行比特翻转
    int_repr ^= 1 << pos  # pos in [0, 15]

    if int_repr >> pos & 1 == 1:
        pos_info['z2o'].append(1)       # 比特翻转由0变为1
    else:
        pos_info['z2o'].append(0)

    # 再转回 bfloat16
    flipped_tensor = torch.tensor(int_repr, dtype=torch.uint16).view(torch.bfloat16)

    return flipped_tensor

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

faultpos = {'value':0}

LAYERNUMBER = 36
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
    
    # 参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="输入错误注入层数，默认为0")
    parser.add_argument("--affect", action='store_true', help="是否开启注入后续层数，默认为False")
    parser.add_argument("--pos", type=int, nargs='+', default=[11,12,13,14], help="输入错误注入位置，默认为空，位置之间用空格分隔")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(),
                        help="注入层选项：q, k, v, ffn-gate, ffn-up, ffn-down, attention-norm, input-norm")
    parser.add_argument("--outputfile", type=str, default="/workplace/home/mayongzhe/faultinject/projects/qwen/result", help="输入注错结果存放目录，会依据注入位置自动补充文件名")
    parser.add_argument("--run", type=int, default=1, help="选择的样本运行次数，默认为1次")
    parser.add_argument("--faultin", type=int, nargs=4, default=[], help="输入4个注入位置参数，依次为 fx fy bx by，默认为全随机值" )
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="在随机选择的1000个样本中指定样本ID，默认为1000个全部进行注错测试")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置，格式为 row col，默认为-1 -1表示随机选择")
    parser.add_argument("--reg", type=str, choices=['input', 'weight', 'psum'], default='weight', help="如果是非随机模式，可以指定注入特定的寄存器(input, weight, psum)，默认为 weight")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"], help="指定固定的数据流类型，默认为WS")
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10", help="注入错误类型，默认为 weight_bitflip_10")
    parser.add_argument("--random", action='store_true', help="开启随机模式，默认混合全随机。可用--fixReg/--fixDataflow限制部分条件")
    parser.add_argument("--ber", type=int, default=1, help="设置随机错误注入的错误率，默认为1，即每个PE阵列中注入1个错误")
    parser.add_argument("--fixReg", type=str, choices=['mixed', 'input', 'weight', 'psum'], default='mixed', help="在随机模式下固定注入的寄存器类型，mixed代表真正的混合随机")
    parser.add_argument("--fixDataflow", type=str, choices=['random', 'WS', 'OS', 'IS'], default='random', help="在随机模式下固定数据流类型，random代表每次生成新随机数据流")

    faultin['value'] = 1 if "--faultin" in sys.argv else 0
    ifsample = True if "--sampleid" in sys.argv else False
    
    args = parser.parse_args()

    
    layertype = args.layerType if args.layerType else "q"
    print(f"[INFO] 注入层为：{layertype}")    
    
    if faultin['value'] == 1:
        set_fault['fx'] = args.faultin[0] if args.faultin else 0
        set_fault['fy'] = args.faultin[1] if len(args.faultin) > 1 else 0
        set_fault['bx'] = args.faultin[2] if len(args.faultin) > 2 else 0
        set_fault['by'] = args.faultin[3] if len(args.faultin) > 3 else 0
        print(f"[INFO] 注入位置参数设置：fx={set_fault['fx']}, fy={set_fault['fy']}, bx={set_fault['bx']}, by={set_fault['by']}")
    else:
        print("[INFO] 注入位置参数设置为随机值")

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

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"[INFO] 模型已加载")
    # 加载数据集
    # ds = load_dataset("TheFinAI/CONVFINQA_test_test", split="test")
    ds = load_dataset("rajpurkar/squad", split="validation")
    # ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # ds = load_dataset("rajpurkar/squad", split="validation")
    random.seed(37)
    samples = random.sample(list(ds), 200)
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

    Fault_Type = args.injectConfig  
    injector = Fast_SA_FaultInjector(
        sa_rows=256, 
        sa_cols=256, 
        dataflow=args.dataflow, 
        fault_type=Fault_Type, 
        precision='fp32'
    )
    print(f"[INFO] 注错模块已初始化, 大小为 256x256, 故障类型为 {Fault_Type}")
    # injector.enabled = False
    # 预先进行判断逻辑，避免在内部重复
    reg_map = {'input': 0, 'weight': 1, 'psum': 2}
    reg_target_id = reg_map.get(args.reg, 1)

    if not args.random:
        # 非随机模式(精准模式)，如果指定了PE，那么全过程固定
        if args.pe != [-1, -1]:
            injector.set_fault_position(args.pe[0], args.pe[1])
            injector.fault_reg[0] = reg_target_id # 手动修改指定的reg
            
    for idx, sample in enumerate(tqdm(samples)):
        handles = []
        
        if args.random:
            # === 随机模式 (全随机或部分固定) ===
            f_type = random.choice(["bitflip", "stuck_0", "stuck_1"])
            # 使用 0-31 作为 fp32 的比位位置
            posid = random.randint(0, 31) 
            # mode name mainly to make parsing safe, runtime behavior uses reg_masks!
            Fault_Type = f"random_{f_type}_{posid}"
            
            injector.fault_type_str = Fault_Type
            injector.parse_fault_type()
            
            # 数据流随机或固定
            if args.fixDataflow == 'random':
                injector.dataflow = random.choice(["WS", "OS", "IS"])
            else:
                injector.dataflow = args.fixDataflow
            
            # 逻辑 reg 类型配置
            if args.fixReg == 'mixed':
                # 真正的全混合，分给3个虚拟地址空间
                injector.init_multi_fault_positions(
                    num_faults=args.ber,
                    num_regs=3,
                    num_bits=32
                )
            else:
                # 仅注入特定的固定寄存器
                reg_id = reg_map.get(args.fixReg, 1)
                injector.init_multi_fault_positions(
                    num_faults=args.ber,
                    num_regs=1,
                    num_bits=32
                )
                # 将随机赋予的都是0的reg_idx强行转换为指定的reg_id
                injector.fault_reg = [reg_id for _ in injector.fault_reg]
                
        else:
            # === 精准模式 (非随机) ===
            # Fault_Type 和 dataflow 保持初始化时的值，不需要重新 parse
            # 但如果 PE 是随机 (-1 -1)，则为每一个 sample 刷新一次随机单点
            if args.pe == [-1, -1]:
                injector.init_fault_position()
                injector.fault_reg[0] = reg_target_id # 覆盖随机默认的reg为指定的reg
                
            # 我们假定 `--injectConfig` 里包含了 pos 信息，所以从 config 里提取
            posid = injector.fault_config.get('pos', 0)

        if args.random:
            layerid = "all"
            layertype = "all"
        else:
            layerid = layerlist[0]
            layertype = args.layerType if args.layerType else "q"
                
        inject_pos['value'] = posid

        if injector.enabled == False:
            output_path = f"{out_path}origin_{len(samples)}.jsonl"
        else:
            if args.random:
                suffix = f"random_reg{args.fixReg}_df{args.fixDataflow}"
            else:
                suffix = layertype
            output_path = f"{out_path}all_layer_{suffix}.jsonl"
            
        target_layers = list(range(model.config.num_hidden_layers)) if args.random else layerlist
        if args.affect and not args.random:
            target_layers = list(range(layerlist[0], min(layerlist[0] + 4, model.config.num_hidden_layers)))
            
        for layernumber in target_layers:
            if args.random or layertype == "all" or pos_mapping.get(layertype) == "all":
                # 挂所有可以挂的钩子
                for kernal in pos_mapping.values():
                    if kernal == "all":
                        continue
                    try:
                        hook = get_module_by_path(model.model.layers[layernumber], kernal)
                        hookRegister = hook.register_forward_hook(injector.hook_fn)
                        handles.append(hookRegister)
                    except AttributeError:
                        continue
            else:
                current_layertype = layertype
                # 根据层数结构动态调整不同的钩子
                if current_layertype in ["q", "k", "v", "o", "qkv"]:
                    if layernumber % 4 == 3:
                        if current_layertype == "qkv":
                            current_layertype = "q"
                    else:
                        if current_layertype in ["q", "k", "v", "o"]:
                            current_layertype = "qkv"

                if current_layertype in pos_mapping:
                    try:
                        hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[current_layertype])
                        hookRegister = hook.register_forward_hook(injector.hook_fn)
                        handles.append(hookRegister)
                    except AttributeError:
                        continue
        # 生成文本并写入
        with open(output_path, "a", encoding="utf-8") as f:
            faultpos['value'] = 0
            flag['normal'] = 0
            messages = [
                {
                    'role': 'user',
                    # 'content': "You are a financial reasoning assistant. Read the following passage and answer the final question. \nExplain your reasoning step by step. \nThen you **MUST** give the final answer in the format: Answer: /box{your answer}\n" + sample['query'].replace("\n\n", "\n") + "Answer the last question."
                    # 'content': "Read the following passage and answer the question. \nExplain your reasoning step by step. \nThen you **MUST** give the final answer in the format: Answer: /box{your answer}\n" + sample['problem']
                    'content': "Read the following passage and answer the question with a text span directly from the passage.\n" + sample['context'] + "\nQuestion: " + sample['question'] + "\nAnswer:"
                }
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            tokenized = tokenizer(prompt, return_tensors="pt")
            num_tokens = tokenized["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    top_p=0.95,
                    temperature=0,
                    eos_token_id=tokenizer.eos_token_id,
                )
                            
                token_length = outputs[0].shape[0] - num_tokens

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
            sep = 'assistant\n'
            try:
                response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
            except IndexError:
                response = decoded.strip().replace("\n\n", "\n")

            result_data = {
                            "sample_id": sample_index[idx],
                            "token_length": token_length,
                            "reference_answer": sample['answers'],
                            # "thinking_process": response.split('</think>', 1)[0].strip() if '</think>' in decoded else response,
                            "generated_answer": response.split('</think>', 1)[1].strip() if '</think>' in decoded else "",
                            # "prompt": prompt,
                            "fault_type":injector.fault_type_str,
                            "inject_layer":layerid,
                            "inject_kernel": layertype,
                            "inject_pos": inject_pos['value'],
                            # "fault_pe_row": str(injector.fault_pe_row) if len(injector.fault_pe_row)>1 else (injector.fault_pe_row[0] if injector.fault_pe_row else -1),
                            # "fault_pe_col": str(injector.fault_pe_col) if len(injector.fault_pe_col)>1 else (injector.fault_pe_col[0] if injector.fault_pe_col else -1),
                            "fault_pe_reg": str(injector.fault_reg) if len(injector.fault_reg)>1 else (injector.fault_reg[0] if injector.fault_reg else 0),
                            "dataflow": injector.dataflow,
                        }

    
    
            json_line = json.dumps(result_data, ensure_ascii=False)
            f.write(json_line + "\n")
            f.flush()
                        
            
            for handle in handles:
                handle.remove()
    print(f"已完成生成，结果保存在：{output_path}")
