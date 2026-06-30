"""
Propagation degree study — how far does a single PE fault propagate in WS dataflow.

Varies propagation_degree ∈ {0, 1, 4, 16, 64, 256} for input stuck-at faults
at bits 24/25/26. Compares accuracy against a clean (no-injection) baseline on
GSM8K.

Usage:
    uv run python projects/qwen3-8b/run_propagation_study.py -o result_propagation/ -g 0
    uv run python projects/qwen3-8b/run_propagation_study.py -n 100 --no-cache-priority
    uv run python projects/qwen3-8b/run_propagation_study.py -n 20 \
        --filter-layer 0 --filter-bit 25 26 27 --filter-stuck 1 \
        --filter-p 0 16 32 64 96 128 160 192 224 256 \
        --kernel-groups v k k,v q,k,v,o
"""

import argparse
import csv
import json
import os
import random
import re
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tool", "src"))
from tool.propagation_injector import Propagation_SA_FaultInjector

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen3-8B"

LAYER_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "mlp-gate": "mlp.gate_proj",
    "mlp-up": "mlp.up_proj",
    "mlp-down": "mlp.down_proj",
}


def _parse_kernel_groups(raw_groups: list[str] | None, fallback: list[str]) -> list[list[str]]:
    if not raw_groups:
        return [fallback]

    groups = []
    for raw_group in raw_groups:
        kernels = [item.strip() for item in raw_group.split(",") if item.strip()]
        if not kernels:
            continue
        unknown = sorted(set(kernels) - set(LAYER_PATHS))
        if unknown:
            raise ValueError(f"Unknown kernel(s) in --kernel-groups: {unknown}")
        groups.append(kernels)

    if not groups:
        raise ValueError("--kernel-groups did not contain any valid kernel names")
    return groups


# ---------------------------------------------------------------------------
# helpers (from run_direct_inject.py)
# ---------------------------------------------------------------------------

def ensure_dir(path_value: str) -> str:
    if not path_value.endswith("/"):
        path_value += "/"
    os.makedirs(path_value, exist_ok=True)
    return path_value


def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def load_model_and_tokenizer(model_id: str, use_cache_first: bool = True):
    if use_cache_first:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, local_files_only=True)
            print(f"[cache] using local cache for {model_id}")
            return tokenizer, model
        except (OSError, EnvironmentError):
            print("[cache] local cache miss, downloading...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True)
    return tokenizer, model


def load_dataset_with_cache(*args, use_cache_first=True, **kwargs):
    if use_cache_first:
        try:
            return load_dataset(*args, download_mode="reuse_dataset_if_exists", **kwargs)
        except (FileNotFoundError, OSError):
            print("[cache] dataset cache miss, downloading...")
    return load_dataset(*args, **kwargs)


# ---------------------------------------------------------------------------
# generation (from run_direct_inject.py)
# ---------------------------------------------------------------------------

def _generate(sample, tokenizer, model) -> dict:
    messages = [{
        "role": "user",
        "content": ("Solve the following math problem step by step.\n"
                     + sample["question"] + "\nAnswer:")
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=2000, do_sample=False,
            eos_token_id=tokenizer.eos_token_id)
    token_length = outputs[0].shape[0] - num_tokens

    decoded = tokenizer.decode(outputs[0][num_tokens:], skip_special_tokens=True)
    response = decoded.strip().replace("\n\n", "\n")
    return {
        "token_length": token_length,
        "generated_answer": (
            response.split("</think>", 1)[1].strip()
            if "</think>" in response else response),
    }


# ---------------------------------------------------------------------------
# accuracy computation (from result/evaluate.py)
# ---------------------------------------------------------------------------

def clean_num(s: str) -> str:
    s = s.replace(",", "")
    s = s.rstrip(".")
    return s


def extract_final_number(text: str) -> str | None:
    # 1. \boxed{...}
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        last = boxed_matches[-1].strip()
        cleaned = re.sub(r"\\text\{[^}]*\}", "", last)
        cleaned = re.sub(r"\\frac\{[^}]*\}\{[^}]*\}", "", cleaned)
        cleaned = cleaned.replace("$", "").replace("%", "").replace("\\", "").strip()
        num = re.search(r"(\d[\d,.]*)", cleaned)
        if num:
            return clean_num(num.group(1))

    # 2. "#### 123"
    gsm = re.findall(r"####\s*\$?([\d.,]+)", text)
    if gsm:
        return clean_num(gsm[-1])

    # 3. ✅/Answer line (cross-line tolerant)
    tail = text[-400:]
    tail_flat = tail.replace("\n", " ")
    all_m = list(
        re.finditer(
            r"(?:✅|Final Answer|final answer|Answer|answer)[:\s]+(?:.*?)(\d[\d,.]*)\s*",
            tail_flat,
        )
    )
    if all_m:
        return clean_num(all_m[-1].group(1))

    # 4. **bold** number
    bold_nums = re.findall(r"\*\*[^*]*?(\d[\d,.]*)[^*]*?\*\*", tail)
    if bold_nums:
        return clean_num(bold_nums[-1])

    # 5. Fallback: last number in final 200 chars
    all_nums = re.findall(r"\b(\d[\d,.]*)\b", text[-200:])
    if all_nums:
        candidates = [
            clean_num(n)
            for n in all_nums
            if len(n.replace(",", "").replace(".", "")) <= 10
        ]
        if candidates:
            return candidates[-1]

    return None


def _compute_accuracy(jsonl_path: str) -> float:
    """Read a JSONL results file and return accuracy as percentage."""
    if not os.path.exists(jsonl_path):
        return 0.0
    data = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    correct = 0
    comparable = 0
    for d in data:
        ref_ans = extract_final_number(d["reference_answer"])
        gen_ans = extract_final_number(d["generated_answer"])
        if ref_ans is None or gen_ans is None:
            continue
        try:
            ref_num = float(ref_ans)
            gen_num = float(gen_ans)
        except ValueError:
            continue
        comparable += 1
        if abs(ref_num - gen_num) < 0.01:
            correct += 1
    return correct / comparable * 100 if comparable > 0 else 0.0


# ---------------------------------------------------------------------------
# baseline runner
# ---------------------------------------------------------------------------

def _run_baseline(samples, tokenizer, model, out_dir: str) -> float:
    """Generate outputs with no fault injection. Returns accuracy."""
    filepath = os.path.join(out_dir, "baseline.jsonl")
    if os.path.exists(filepath):
        print(f"[SKIP] baseline.jsonl exists — computing accuracy from cache")
        return _compute_accuracy(filepath)

    random.seed(42)
    torch.manual_seed(42)

    print(f"[RUN] baseline ({len(samples)} samples, no hooks)")
    with open(filepath, "a", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(samples, desc="baseline")):
            gen = _generate(sample, tokenizer, model)
            result = {
                "sample_id": str(idx),
                "reference_answer": sample["answer"],
                **gen,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    acc = _compute_accuracy(filepath)
    print(f"[DONE] baseline accuracy: {acc:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# combo runner
# ---------------------------------------------------------------------------

def _run_combo(bit: int, stuck_val: str, p: int, target_layers: list,
               target_kernels: list, out_dir: str, samples, tokenizer, model) -> float:
    """Run one combo. target_layers=[0] = single layer, target_layers=[0..31] = all layers."""
    stuck_label = f"stuck_{stuck_val}"
    layer_tag = "Lall" if len(target_layers) > 1 else f"L{target_layers[0]}"
    kernel_tag = "_".join(sorted(target_kernels)) if len(target_kernels) < 4 else f"{len(target_kernels)}k"
    fname = f"{layer_tag}_{kernel_tag}_bit{bit}_{stuck_label}_p{p}.jsonl"
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    filepath = os.path.join(samples_dir, fname)

    if os.path.exists(filepath):
        print(f"[SKIP] {fname} exists — computing accuracy from cache")
        return _compute_accuracy(filepath)

    random.seed(42)
    torch.manual_seed(42)

    fault_type = f"input_stuck_{stuck_val}_{bit}"
    injector = Propagation_SA_FaultInjector(
        propagation_degree=p,
        sa_rows=256, sa_cols=256,
        dataflow="WS",
        fault_type=fault_type,
        precision="fp32",
    )
    injector.enabled = True
    injector.set_fault_position(0, 0)

    n_hooks = len(target_layers) * len(target_kernels)
    print(f"[RUN] layers={len(target_layers)}  kernels={target_kernels}  hooks={n_hooks}  bit={bit}  stuck={stuck_label}  p={p}")

    handles = []
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        for k in target_kernels:
            kernel_path = LAYER_PATHS[k]
            try:
                mod = get_module_by_path(layer, kernel_path)
                handles.append(mod.register_forward_hook(injector.hook_fn))
            except AttributeError:
                pass

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            for idx, sample in enumerate(tqdm(samples, desc=fname)):
                gen = _generate(sample, tokenizer, model)
                result = {
                    "sample_id": str(idx),
                    "reference_answer": sample["answer"],
                    **gen,
                    "fault_type": fault_type,
                    "dataflow": "WS",
                    "precision": "fp32",
                    "bit": bit,
                    "stuck": stuck_label,
                    "propagation_degree": p,
                    "inject_layer": layer_tag,
                    "inject_kernel": kernel_tag,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        for h in handles:
            h.remove()

    acc = _compute_accuracy(filepath)
    print(f"[DONE] {fname}  accuracy={acc:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Propagation Degree Study — Qwen3-8B input stuck-at faults")
    parser.add_argument("--outputfile", "-o", type=str, default="./result_propagation")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--num-samples", "-n", type=int, default=200)
    parser.add_argument("--filter-bit", type=int, nargs="*",
                        default=[24, 25, 26], help="Filter bit positions to run")
    parser.add_argument("--filter-stuck", type=str, nargs="*",
                        default=["0", "1"], help='Filter stuck directions ("0" or "1")')
    parser.add_argument("--filter-layer", type=int, nargs="*",
                        default=[0], help="Filter layer indices to inject")
    parser.add_argument("--filter-p", type=int, nargs="*",
                        default=[0, 1, 4, 16, 64, 256], help="Filter propagation degrees")
    parser.add_argument("--all-layers", action="store_true",
                        help="Inject into ALL layers×kernels simultaneously (224 hooks)")
    parser.add_argument("--filter-kernel", type=str, nargs="*",
                        default=list(LAYER_PATHS.keys()),
                        help="Filter kernels to inject")
    parser.add_argument("--kernel-groups", type=str, nargs="*", default=None,
                        help=("Run multiple comma-separated kernel groups, e.g. "
                              "--kernel-groups v k k,v q,k,v,o"))
    parser.add_argument("--no-cache-priority", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_path = ensure_dir(args.outputfile)
    use_cache = not args.no_cache_priority

    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_cache_first=use_cache)
    model.eval()
    num_all_layers = model.config.num_hidden_layers
    print(f"[INFO] model loaded: {MODEL_ID}, layers={num_all_layers}")

    ds = load_dataset_with_cache(
        "openai/gsm8k", "main", split="train", use_cache_first=use_cache)
    random.seed(37)
    samples = random.sample(list(ds), args.num_samples)
    print(f"[INFO] dataset loaded: {len(samples)} samples")

    # ── baseline ──────────────────────────────────────────────────────────
    baseline_acc = _run_baseline(samples, tokenizer, model, out_path)
    print(f"[INFO] baseline accuracy: {baseline_acc:.2f}%")

    # ── combos ────────────────────────────────────────────────────────────
    bits = args.filter_bit
    stuck_vals = args.filter_stuck
    ps = args.filter_p
    kernel_groups = _parse_kernel_groups(args.kernel_groups, args.filter_kernel)

    if args.all_layers:
        layer_groups = [list(range(num_all_layers))]
    else:
        layer_groups = [[l] for l in args.filter_layer]

    csv_path = os.path.join(out_path, "accuracy_vs_p.csv")
    csv_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["layer", "kernel", "bit", "stuck", "p", "baseline_accuracy",
                             "faulty_accuracy", "acc_drop"])

        for target_layers in layer_groups:
            layer_label = "all" if len(target_layers) > 1 else str(target_layers[0])
            for kernels in kernel_groups:
                kernel_label = "_".join(sorted(kernels)) if len(kernels) < 4 else f"{len(kernels)}k"
                for bit in bits:
                    for stuck_val in stuck_vals:
                        for p in ps:
                            acc = _run_combo(bit, stuck_val, p, target_layers, kernels,
                                             out_path, samples, tokenizer, model)
                            acc_drop = baseline_acc - acc
                            writer.writerow([
                                layer_label, kernel_label, bit, f"stuck_{stuck_val}", p,
                                f"{baseline_acc:.2f}", f"{acc:.2f}", f"{acc_drop:.2f}",
                            ])
                            csvfile.flush()

    print(f"[INFO] accuracy_vs_p.csv written to {csv_path}")


if __name__ == "__main__":
    main()
