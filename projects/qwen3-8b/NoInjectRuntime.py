"""
无注错运行脚本，只采集运行时指标。
适配 Qwen/Qwen3-8B，优先使用本地缓存。
"""

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from tool.runtime_metrics import RuntimeMetricsWriter, compute_runtime_metrics, extract_tensor

logging.set_verbosity_error()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen3-8B"


def load_model_and_tokenizer(model_id, use_cache_first=True):
    """加载模型和分词器，优先使用本地缓存。"""
    if use_cache_first:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, local_files_only=True
            )
            print(f"[cache] 使用本地缓存加载 {model_id}")
            return tokenizer, model
        except (OSError, EnvironmentError):
            print("[cache] 本地缓存未就绪，尝试从远程下载...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True
    )
    return tokenizer, model


def load_dataset_with_cache(*args, use_cache_first=True, **kwargs):
    """加载数据集，优先使用本地缓存。"""
    if use_cache_first:
        try:
            ds = load_dataset(*args, **kwargs, download_mode="reuse_dataset_if_exists")
            print(f"[cache] 使用本地缓存加载数据集")
            return ds
        except Exception:
            print("[cache] 本地数据集缓存未就绪，尝试从远程下载...")
    return load_dataset(*args, **kwargs)


def ensure_dir_with_suffix(path_value):
    if not path_value.endswith("/"):
        path_value += "/"
    os.makedirs(path_value, exist_ok=True)
    return path_value


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


def build_metrics_hook(writer, collector, run_meta_getter):
    def hook(module, _input_tuple, output_tuple):
        module_name = getattr(
            module, "_runtime_metrics_name", f"{module.__class__.__name__}_{id(module)}"
        )
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
            "fault_enabled": False,
            "fault_type": "none",
            "dataflow": "none",
        }
        record.update(writer.extra_meta)
        record.update(run_meta_getter())
        collector.capture(record, output_tensor)
        return output_tuple

    return hook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detectLayerList", type=int, nargs="+", default=[0],
        help="采集指标的 detect layer 列表"
    )
    parser.add_argument(
        "--outputfile", type=str,
        default="/workplace/home/mayongzhe/faultinject/projects/qwen3-8b/result"
    )
    parser.add_argument(
        "--sampleid", type=int, nargs="*", default=[],
        help="指定采样样本ID"
    )
    parser.add_argument(
        "--metricInterval", type=int, default=20,
        help="运行时指标采样间隔，默认20"
    )
    parser.add_argument(
        "--metricOutput", type=str,
        default="",
        help="运行时指标输出 jsonl 路径"
    )
    parser.add_argument(
        "--no-cache-priority", action="store_true",
        help="禁用本地缓存优先，直接从远程下载"
    )
    args = parser.parse_args()

    out_path = ensure_dir_with_suffix(args.outputfile)
    detect_layer_list = args.detectLayerList
    sample_index = args.sampleid
    use_cache = not args.no_cache_priority

    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_cache_first=use_cache)
    model.eval()

    ds = load_dataset_with_cache(
        "openai/gsm8k", "main", split="train", use_cache_first=use_cache
    )
    random.seed(37)
    samples = random.sample(list(ds), 200)

    if sample_index:
        samples = [samples[i] for i in sample_index]
    else:
        sample_index = [str(i) for i in range(len(samples))]

    result_output_path = f"{out_path}origin_{len(samples)}.jsonl"
    metric_output_path = args.metricOutput or f"{out_path}runtime_metrics_noinject.jsonl"
    open(result_output_path, "w", encoding="utf-8").close()
    open(metric_output_path, "w", encoding="utf-8").close()
    writer = RuntimeMetricsWriter(
        output_path=metric_output_path,
        interval=args.metricInterval,
        run_tag="noinject",
        extra_meta={"script": "NoInjectRuntime", "model": MODEL_ID},
        flush_every=1,
    )

    current_meta = {}
    matrix_collector = RuntimeMatrixCollector()
    metric_hook = build_metrics_hook(writer, matrix_collector, lambda: current_meta)

    for idx, sample in enumerate(tqdm(samples)):
        handles = []
        writer.set_run_tag(
            "noinject",
            extra_meta={"script": "NoInjectRuntime", "model": MODEL_ID},
            reset_counters=True,
        )
        matrix_collector.clear()

        current_meta = {
            "sample_id": sample_index[idx],
            "inject_layer": "none",
            "inject_kernel": "none",
            "inject_pos": -1,
            "detect_layers": list(detect_layer_list),
        }

        for layernumber in detect_layer_list:
            layer_module = model.model.layers[layernumber]
            setattr(layer_module, "_runtime_metrics_name", f"layers.{layernumber}")
            handles.append(layer_module.register_forward_hook(metric_hook))

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
            num_tokens = inputs["input_ids"].shape[-1]

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

            decoded = tokenizer.decode(outputs[0][num_tokens:], skip_special_tokens=True)
            response = decoded.strip().replace("\n\n", "\n")

            result_data = {
                "sample_id": sample_index[idx],
                "token_length": token_length,
                "reference_answer": sample["answer"],
                "generated_answer": (
                    response.split("</think>", 1)[1].strip()
                    if "</think>" in response
                    else response
                ),
                "fault_type": "none",
                "inject_layer": "none",
                "inject_kernel": "none",
                "inject_pos": -1,
                "detect_layers": detect_layer_list,
                "fault_pe_reg": "none",
                "dataflow": "none",
                "metric_output": metric_output_path,
                "metric_interval": args.metricInterval,
            }
            handle.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            handle.flush()

        for hook_handle in handles:
            hook_handle.remove()

    print(f"已完成生成，结果保存在：{result_output_path}")
    print(f"运行时指标保存在：{metric_output_path}")
