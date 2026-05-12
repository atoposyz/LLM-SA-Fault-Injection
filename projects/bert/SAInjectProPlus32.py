import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from datasets import load_dataset
from struct import pack, unpack
from cmath import isinf, isnan
import os

import torch.nn.functional as F

logging.set_verbosity_error()

import argparse
import sys
import json

# 获取当前脚本的父目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到Python的搜索路径中
sys.path.append(project_root)
from tool.fault_injector_next import Fast_SA_FaultInjector as SA_FaultInjector

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
    'blockPos_Y': 0
}
faultin = { 'value': 0 }
set_fault = {"fx": 0, "fy": 0, "bx": 0, "by": 0}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_mapping = {
    "q": "attention.self.query",
    "k": "attention.self.key",
    "v": "attention.self.value",
    "o": "attention.output.dense",
    "im": "intermediate.dense",
    "op": "output.dense",
    "all": "all"
}

    

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

faultpos = {'value':0}


'''
python /workplace/home/mayongzhe/code/bert-emotion/SAInjectProPlus.py --layerType v --pos 30 --layerList 3 --outputfile /workplace/home/mayongzhe/code/bert-emotion/new_res

'''


if __name__ == "__main__":
    
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="输入错误注入层数[0-35]，默认为0")
    parser.add_argument("--affect", action='store_true', help="是否开启注入后续层数，默认为False")
    parser.add_argument("--pos", type=int, nargs='+', default=[11,12,13,14], help="输入错误注入位置，默认为空，位置之间用空格分隔")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(),
                        help="注入层选项：q, k, v, ffn-gate, ffn-up, ffn-down, attention-norm, input-norm")
    parser.add_argument("--outputfile", type=str, default="/workplace/home/aistation/qwen3/bf16bit/rank/", help="输入注错结果存放目录，默认为/home/aistation/python-code/qwen3/bf16bit/normlayer/，会依据注入位置自动补充文件名，文件类型为txt")
    parser.add_argument("--run", type=int, default=1, help="选择的样本运行次数，默认为1次")
    parser.add_argument("--faultin", type=int, nargs=4, default=[], help="输入4个注入位置参数，依次为 fx fy bx by，默认为全随机值" )
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="在随机选择的1000个样本中指定样本ID，默认为1000个全部进行注错测试")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置，格式为 row col，默认为-1 -1表示随机选择")
    parser.add_argument("--injectConfig", type=str, default="input_bitflip", help="注入错误类型，默认为 input_bitflip")

    faultin['value'] = 1 if "--faultin" in sys.argv else 0
    ifsample = True if "--sampleid" in sys.argv else False
    
    args = parser.parse_args()

    if args.layerType is None:
        print("[ERROR] 启动注错模式必须指定 --layerType 参数！")
        sys.exit(1)

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

    tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion").to(device)
    model.eval()
    print(f"[INFO] 模型已加载")
    # 加载数据集
    ds = load_dataset("boltuix/emotions-dataset", split="train")
    random.seed(37)
    samples = random.sample(list(ds), 1000)
    print("[INFO] 数据集已加载")

    id2label = model.config.id2label
    label2id = model.config.label2id

    if sample_index:
        # 指定了 sampleid，则从数据集中按索引取样
        try:
            samples = [samples[i] for i in sample_index]
        except IndexError as e:
            raise ValueError(f"指定的样本ID越界：{e}")
    else:
        print("[INFO] 未指定样本ID，使用随机选择的全部样本进行注错测试")
        sample_index = [str(i) for i in range(len(samples))]  # 生成样本索引列表
    

    Fault_Type = args.injectConfig  

    injector = SA_FaultInjector(sa_rows=32, sa_cols=32, fault_type=Fault_Type)
    print(f"[INFO] 注错模块已初始化, 大小为 32x32, 故障类型为 {Fault_Type}")

    if args.pe != [-1, -1]:
        injector.set_fault_position(args.pe[0], args.pe[1])
    else:
        injector.init_fault_position()  # 随机选择注入位置


    handles =[]

    for posid in poslist:
        injector.set_fault_config_injpos(posid)
        # injector.print_config()
        # exit(1)
        for layerid in layerlist:
            # hook = get_module_by_path(model.text_decoder.bert.encoder.layer[layerid], pos_mapping[layertype])
            
            for runid in range(runtimes):
                print(f"[INFO] 注入层为：{layertype}")
                print(f"[INFO] 当前注入位置：{posid} in {poslist}")
                print(f"[INFO] 当前注入层：{layerid} in {layerlist}")
                print(f"[INFO] 运行次数：{runid + 1}/{runtimes}")
                inject_pos['value'] = posid

                if faultin['value'] == 1:
                    output_path = out_path + str(layerid) + "_" + layertype + "_" + str(posid) + "_" + str(set_fault['fx']) + "_" + str(set_fault['fy']) + "_" + str(set_fault['bx']) + "_" + str(set_fault['by']) + ".jsonl"
                else:
                    output_path = f"{out_path}{injector.fault_config['mode']}_{injector.fault_config['type']}{injector.fault_config['stuck'] if injector.fault_config['type'] == 'stuck' else ''}_{layertype}_L{layerid}+_P{posid}_FPE{args.pe[0]},{args.pe[1]}.jsonl"
                    # output_path = out_path + str(layerid) + "_" + layertype + "_" + str(posid) + "_ram_fault_1000.jsonl"
                if args.affect:
                    print(f"[INFO] 开启后续层注入模式")
                    lastLayer = 4
                else:
                    print(f"[INFO] 仅注入指定层模式")
                    lastLayer = layerid + 1
                for layernumber in range(lastLayer):
                    if layernumber < layerid:
                        continue
                    if pos_mapping[layertype] != "all":
                        hook = get_module_by_path(model.bert.encoder.layer[layernumber], pos_mapping[layertype])
                        hookRegister = hook.register_forward_hook(injector.hook_fn)
                        handles.append(hookRegister)
                    else:
                        for kernal in pos_mapping.values():
                            if kernal == "all":
                                continue
                            hook = get_module_by_path(model.bert.encoder.layer[layernumber], kernal)
                            hookRegister = hook.register_forward_hook(injector.hook_fn)
                            handles.append(hookRegister)
                # hookRegister = hook.register_forward_hook(injector.hook_fn)

                # 生成文本并写入
                with open(output_path, "a", encoding="utf-8") as f:
                    correct_top1 = 0
                    correct_top3 = 0
                    total = len(samples)
                    for idx, sample in enumerate(tqdm(samples)):
                        faultpos['value'] = 0
                        flag['normal'] = 0

                        inputs = tokenizer(sample['Sentence'], return_tensors="pt", truncation=True).to(device)
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            # logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                            probs = F.softmax(logits, dim=-1)
                            # probs = torch.nan_to_num(probs, nan=1e-7, posinf=1.0, neginf=0.0)
                        pred = torch.argmax(probs, dim=-1).item()
                        pred_label = id2label[pred]
                        true_label = sample['Label']

                        if pred == label2id[true_label]:
                            correct_top1 += 1

                        # 获取 top-3
                        topk_probs, topk_indices = torch.topk(probs, k=3, dim=-1)
                        topk_probs = topk_probs[0].tolist()
                        topk_indices = topk_indices[0].tolist()
                        top3_labels = [id2label[idx] for idx in topk_indices]

                        top3_str = ", ".join(
                            f"{label}:{prob:.4f}" for label, prob in zip(top3_labels, topk_probs)
                        )

                        top3_hit = true_label in top3_labels
                        if top3_hit:
                            correct_top3 += 1
                        result_data = {
                            "sample_id": sample_index[idx],
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "top3": top3_str,
                            "top1_correct": pred == label2id[true_label],
                            "top3_correct": top3_hit,   
                            "text": sample['Sentence'],
                        }

                        json_line = json.dumps(result_data, ensure_ascii=False)
                        f.write(json_line + "\n")
                        
                        f.flush()
                        # print(flag['normal'])
                        # flag['normal'] = 0
                for handle in handles:
                    handle.remove()
                print(f"已完成生成，结果保存在：{output_path}")
                # hookRegister.remove()
'''

python /workplace/home/mayongzhe/code/bert-emotion/SAInjectProPlus.py --outputfile /workplace/home/mayongzhe/code/bert-emotion/new_res/L0allbit --layerType all --pos 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --layerList 0 --injectConfig weight_stuck_at_1 --pe 0 0

'''

