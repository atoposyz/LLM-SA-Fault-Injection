"""
注错运行脚本，使用 fault_injector_next 执行注错，使用 runtime_metrics 采集指标。
"""

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from tool.fault_injector_next import Fast_SA_FaultInjector
from tool.runtime_metrics import RuntimeMetricsWriter, compute_runtime_metrics, extract_tensor

logging.set_verbosity_error()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


pos_mapping = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "mlp-gate": "mlp.gate_proj",
    "mlp-up": "mlp.up_proj",
    "mlp-down": "mlp.down_proj",
}


def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def ensure_dir_with_suffix(path_value):
    if not path_value.endswith("/"):
        path_value += "/"
    os.makedirs(path_value, exist_ok=True)
    return path_value


def resolve_injection_targets(layernumber, layertype, layer_module, random_mode):
    targets = []
    if random_mode or layertype == "all":
        for kernel_name, kernel_path in pos_mapping.items():
            try:
                targets.append((f"layers.{layernumber}.{kernel_name}", get_module_by_path(layer_module, kernel_path)))
            except AttributeError:
                continue
        return targets

    current_layertype = layertype
    if current_layertype in pos_mapping:
        try:
            targets.append(
                (
                    f"layers.{layernumber}.{current_layertype}",
                    get_module_by_path(layer_module, pos_mapping[current_layertype]),
                )
            )
        except AttributeError:
            pass
    return targets


class RuntimeMatrixCollector:
    def __init__(self):
        self.records = []

    def clear(self):
        self.records = []

    def capture(self, record, tensor):
        if tensor is None or tensor.numel() == 0:
            return
        self.records.append((record, tensor.detach().cpu().contiguous()))

    def write_metrics(self, writer):
        for record, matrix in self.records:
            metrics = compute_runtime_metrics(matrix)
            if metrics is None:
                continue
            output_record = dict(record)
            output_record.update(metrics)
            writer.write_record(output_record)
        self.clear()
        writer.flush()


def build_metrics_hook(writer, collector, run_meta_getter, injector):
    def hook(module, _input_tuple, output_tuple):
        module_name = getattr(module, "_runtime_metrics_name", f"{module.__class__.__name__}_{id(module)}")
        should_capture, module_step, global_step = writer.next_step(module_name)
        if not should_capture:
            return output_tuple

        output_tensor = extract_tensor(output_tuple)
        if output_tensor is None:
            return output_tuple

        record = {
            "run_tag": writer.run_tag,
            "layer_name": module_name,
            "module_type": module.__class__.__name__,
            "stage": "output",
            "module_step": int(module_step),
            "global_step": int(global_step),
            "metric_interval": int(writer.interval),
            "fault_enabled": True,
            "fault_type": injector.fault_type_str,
            "dataflow": injector.dataflow,
            "fault_pe_row": list(injector.fault_pe_row),
            "fault_pe_col": list(injector.fault_pe_col),
            "fault_reg": list(injector.fault_reg),
            "fault_bit": list(injector.fault_bit),
        }
        record.update(writer.extra_meta)
        record.update(run_meta_getter())
        collector.capture(record, output_tensor)
        return output_tuple

    return hook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layerList", type=int, nargs="+", default=[0], help="输入错误注入层数，默认为0")
    parser.add_argument("--detectLayerList", type=int, nargs="+", default=[0], help="采集指标的 detect layer 列表")
    parser.add_argument("--affect", action="store_true", help="是否开启注入后续层数，默认为False")
    parser.add_argument("--layerType", type=str, choices=list(pos_mapping.keys()) + ["all"], help="注入层类型")
    parser.add_argument("--outputfile", type=str, default="/workplace/home/mayongzhe/faultinject/projects/qwen/result")
    parser.add_argument("--sampleid", type=int, nargs="*", default=[], help="指定样本ID")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="手动指定注入PE位置")
    parser.add_argument("--reg", type=str, choices=["input", "weight", "psum"], default="weight")
    parser.add_argument("--dataflow", type=str, default="WS", choices=["WS", "OS", "IS"])
    parser.add_argument("--injectConfig", type=str, default="weight_bitflip_10")
    parser.add_argument("--random", action="store_true", help="开启随机注错模式")
    parser.add_argument("--ber", type=int, default=1, help="随机注错错误率")
    parser.add_argument("--fixReg", type=str, choices=["mixed", "input", "weight", "psum"], default="mixed")
    parser.add_argument("--fixDataflow", type=str, choices=["random", "WS", "OS", "IS"], default="random")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"], help="注错精度，默认bf16")
    parser.add_argument("--metricInterval", type=int, default=20, help="运行时指标采样间隔，默认20")
    parser.add_argument("--metricOutput", type=str, default="", help="运行时指标输出 jsonl 路径")
    args = parser.parse_args()

    layertype = args.layerType if args.layerType else "q"
    layerlist = args.layerList
    detect_layer_list = args.detectLayerList
    out_path = ensure_dir_with_suffix(args.outputfile)
    sample_index = args.sampleid

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto", trust_remote_code=True)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    random.seed(37)
    samples = random.sample(list(ds), 200)

    if sample_index:
        samples = [samples[i] for i in sample_index]
    else:
        sample_index = [str(i) for i in range(len(samples))]

    injector = Fast_SA_FaultInjector(
        sa_rows=256,
        sa_cols=256,
        dataflow=args.dataflow,
        fault_type=args.injectConfig,
        precision=args.precision,
    )
    injector.enabled = True

    metric_output_path = args.metricOutput or f"{out_path}runtime_metrics_inject.jsonl"
    writer = RuntimeMetricsWriter(
        output_path=metric_output_path,
        interval=args.metricInterval,
        run_tag="inject",
        extra_meta={"script": "SAInjectRuntime"},
        flush_every=1,
    )

    reg_map = {"input": 0, "weight": 1, "psum": 2}
    reg_target_id = reg_map.get(args.reg, 1)
    if not args.random and args.pe != [-1, -1]:
        injector.set_fault_position(args.pe[0], args.pe[1])
        injector.fault_reg[0] = reg_target_id

    current_meta = {}
    matrix_collector = RuntimeMatrixCollector()
    detect_metric_hook = build_metrics_hook(writer, matrix_collector, lambda: current_meta, injector)

    for idx, sample in enumerate(tqdm(samples)):
        handles = []

        if args.random:
            f_type = random.choice(["bitflip", "stuck_0", "stuck_1"])
            posid = random.randint(0, 31)
            injector.fault_type_str = f"random_{f_type}_{posid}"
            injector.parse_fault_type()
            injector.dataflow = random.choice(["WS", "OS", "IS"]) if args.fixDataflow == "random" else args.fixDataflow

            if args.fixReg == "mixed":
                injector.init_multi_fault_positions(num_faults=args.ber, num_regs=3, num_bits=32)
            else:
                reg_id = reg_map.get(args.fixReg, 1)
                injector.init_multi_fault_positions(num_faults=args.ber, num_regs=1, num_bits=32)
                injector.fault_reg = [reg_id for _ in injector.fault_reg]

            layerid = "all"
            current_layertype = "all"
        else:
            if args.pe == [-1, -1]:
                injector.init_fault_position()
                injector.fault_reg[0] = reg_target_id
            posid = injector.fault_config.get("pos", 0)
            layerid = layerlist[0]
            current_layertype = layertype

        writer.set_run_tag("inject", extra_meta={"script": "SAInjectRuntime"}, reset_counters=True)
        matrix_collector.clear()
        current_meta = {
            "sample_id": sample_index[idx],
            "inject_layer": layerid,
            "inject_kernel": current_layertype,
            "inject_pos": posid,
            "detect_layers": list(detect_layer_list),
        }

        if args.random:
            result_output_path = f"{out_path}all_layer_random_reg{args.fixReg}_df{args.fixDataflow}.jsonl"
        else:
            result_output_path = f"{out_path}all_layer_{current_layertype}.jsonl"

        target_layers = list(range(model.config.num_hidden_layers)) if args.random else layerlist
        if args.affect and not args.random:
            target_layers = list(range(layerlist[0], model.config.num_hidden_layers))

        for layernumber in target_layers:
            layer_module = model.model.layers[layernumber]
            for hook_name, hook_module in resolve_injection_targets(layernumber, current_layertype, layer_module, args.random):
                setattr(hook_module, "_runtime_metrics_name", hook_name)
                handles.append(hook_module.register_forward_hook(injector.hook_fn))

        for detect_layer in detect_layer_list:
            detect_module = model.model.layers[detect_layer]
            setattr(detect_module, "_runtime_metrics_name", f"layers.{detect_layer}")
            handles.append(detect_module.register_forward_hook(detect_metric_hook))

        with open(result_output_path, "a", encoding="utf-8") as handle:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem step by step.\n"
                        + sample["question"]
                        + "\nAnswer:"
                    ),
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            tokenized = tokenizer(prompt, return_tensors="pt")
            num_tokens = tokenized["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=False,
                    top_p=0.95,
                    temperature=0,
                    eos_token_id=tokenizer.eos_token_id,
                )
                token_length = outputs[0].shape[0] - num_tokens

            matrix_collector.write_metrics(writer)

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            sep = "assistant\n"
            try:
                response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
            except IndexError:
                response = decoded.strip().replace("\n\n", "\n")

            result_data = {
                "sample_id": sample_index[idx],
                "token_length": token_length,
                "reference_answer": sample["answer"],
                "generated_answer": response.split("</think>", 1)[1].strip() if "</think>" in response else response,
                "fault_type": injector.fault_type_str,
                "inject_layer": layerid,
                "inject_kernel": current_layertype,
                "inject_pos": posid,
                "detect_layers": detect_layer_list,
                "fault_pe_reg": str(injector.fault_reg) if len(injector.fault_reg) > 1 else (injector.fault_reg[0] if injector.fault_reg else 0),
                "dataflow": injector.dataflow,
                "metric_output": metric_output_path,
                "metric_interval": args.metricInterval,
            }
            handle.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            handle.flush()

        for hook_handle in handles:
            hook_handle.remove()

    print(f"已完成生成，结果保存在：{result_output_path}")
    print(f"运行时指标保存在：{metric_output_path}")
