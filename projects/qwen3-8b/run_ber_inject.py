"""
BER 多点故障注入脚本 — Qwen3-8B。
遍历 dataflow x reg x BER 组合，每组合输出独立 JSONL。
支持 --filter-* 参数拆分工作到多 GPU/多机器。
"""

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from tool.ber_injector import BER_Fast_SA_FaultInjector
from tool.grouped_injector import GroupedExact_SA_FaultInjector, GlobalCoverage_SA_FaultInjector

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen3-8B"

pos_mapping = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "mlp-gate": "mlp.gate_proj",
    "mlp-up": "mlp.up_proj",
    "mlp-down": "mlp.down_proj",
}

BER_VALUES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

MODE_GROUPS = [
    ("random", "mixed"),
    ("IS", "input"), ("IS", "weight"), ("IS", "psum"),
    ("OS", "input"), ("OS", "weight"), ("OS", "psum"),
    ("WS", "input"), ("WS", "weight"), ("WS", "psum"),
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def ensure_dir(path_value: str) -> str:
    if not path_value.endswith("/"):
        path_value += "/"
    os.makedirs(path_value, exist_ok=True)
    return path_value


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
# sample runner
# ---------------------------------------------------------------------------


def _run_sample(df, reg, ber, expected_faults, num_regs, reg_map, idx, sample,
                injector, tokenizer, model, file_handle):
    handles = []

    # fault_type: stuck_0 or stuck_1 only
    # mode must be one of input/weight/psum — the injector dispatches on this
    f_type = random.choice(["stuck_0", "stuck_1"])
    mode = random.choice(["input", "weight", "psum"]) if reg == "mixed" else reg
    injector.fault_type_str = f"{mode}_{f_type}_0"
    injector.parse_fault_type()

    # dataflow: random per sample in random mode
    if df == "random":
        injector.dataflow = random.choice(["WS", "OS", "IS"])

    # BER injection
    if expected_faults >= 1.0:
        injector.init_faults_by_ber(ber, num_regs=num_regs, num_bits=32)
    else:
        # Always consume 1 random position (keep sequence aligned)
        injector.init_multi_fault_positions(1, num_regs=num_regs, num_bits=32)
        if not (random.random() < expected_faults):
            injector.reset_fault_pe()

    # fix reg for single-reg modes
    if reg != "mixed":
        reg_id = reg_map[reg]
        injector.fault_reg = [reg_id for _ in injector.fault_reg]

    # mount hooks on all layers
    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for kernel_name, kernel_path in pos_mapping.items():
            mod = get_module_by_path(layer, kernel_path)
            handles.append(mod.register_forward_hook(injector.hook_fn))

    # generate
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

    # record
    result = {
        "sample_id": str(idx),
        "token_length": token_length,
        "reference_answer": sample["answer"],
        "generated_answer": (
            response.split("</think>", 1)[1].strip()
            if "</think>" in response else response),
        "fault_type": injector.fault_type_str,
        "dataflow": injector.dataflow,
        "ber": ber,
        "fix_reg": reg,
        "inject_layer": "all",
        "inject_kernel": "all",
    }
    file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    file_handle.flush()

    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# combo runner
# ---------------------------------------------------------------------------


def _run_combo(df, reg, ber, out_path, samples, tokenizer, model, reg_map,
               inject_mode="exact"):
    ber_str = f"{ber:.0e}".replace("e-0", "e-").replace("e+0", "e")
    fname = f"ber_{ber_str}_df{df}_reg{reg}.jsonl"
    filepath = out_path + fname

    num_regs = 3 if reg == "mixed" else 1
    total_bits = 256 * 256 * num_regs * 32
    expected_faults = total_bits * ber

    # seed reset per combo
    random.seed(42)
    torch.manual_seed(42)

    # resume: count already-completed samples from existing file
    resume_from = 0
    if os.path.exists(filepath):
        with open(filepath) as fp:
            resume_from = sum(1 for _ in fp)
        if resume_from >= len(samples):
            print(f"[SKIP] {fname} already complete ({resume_from}/{len(samples)})")
            return
        print(f"[RESUME] {fname} from sample {resume_from}/{len(samples)}")

        # fast-forward random state to match completed samples
        _consume_random(df, reg, ber, expected_faults, num_regs, resume_from)

    print(f"[RUN] {fname}  expected_faults={expected_faults:.3f}")

    # injector — use "WS" as placeholder for random mode (overridden per sample)
    initial_df = "WS" if df == "random" else df
    inj_cls = {
        "exact": BER_Fast_SA_FaultInjector,
        "grouped_exact": GroupedExact_SA_FaultInjector,
        "coverage": GlobalCoverage_SA_FaultInjector,
    }[inject_mode]
    injector = inj_cls(
        sa_rows=256, sa_cols=256, dataflow=initial_df,
        fault_type="weight_stuck_0_0", precision="bf16")
    injector.enabled = True
    if inject_mode != "exact":
        print(f"[INFO] using {inject_mode} injector")

    # sample loop
    with open(filepath, "a", encoding="utf-8") as f:
        for idx in range(resume_from, len(samples)):
            _run_sample(df, reg, ber, expected_faults, num_regs, reg_map,
                        idx, samples[idx], injector, tokenizer, model, f)
    print(f"[DONE] {fname}")


def _consume_random(df, reg, ber, expected_faults, num_regs, n):
    """Fast-forward random state for n already-completed samples."""
    max_positions = 256 * 256 * num_regs * 32
    for _ in range(n):
        random.choice(["stuck_0", "stuck_1"])
        if reg == "mixed":
            random.choice(["input", "weight", "psum"])
        if df == "random":
            random.choice(["WS", "OS", "IS"])
        if expected_faults >= 1.0:
            num_faults = int(max_positions * ber)
            random.sample(range(max_positions), num_faults)
        else:
            random.sample(range(max_positions), 1)
            random.random()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B BER Fault Injection")
    parser.add_argument("--outputfile", "-o", type=str, default="./result")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--filter-dataflow", type=str, nargs="*",
                        choices=["random", "IS", "OS", "WS"],
                        default=["random", "IS", "OS", "WS"])
    parser.add_argument("--filter-reg", type=str, nargs="*",
                        choices=["mixed", "input", "weight", "psum"],
                        default=["mixed", "input", "weight", "psum"])
    parser.add_argument("--filter-ber", type=float, nargs="*",
                        default=BER_VALUES)
    parser.add_argument("--num-samples", "-n", type=int, default=200)
    parser.add_argument("--mode", type=str, choices=["exact", "grouped_exact", "coverage"],
                        default="exact")
    parser.add_argument("--no-cache-priority", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_path = ensure_dir(args.outputfile)
    use_cache = not args.no_cache_priority

    # model & dataset
    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_cache_first=use_cache)
    model.eval()
    print(f"[INFO] model loaded: {MODEL_ID}, layers={model.config.num_hidden_layers}")

    ds = load_dataset_with_cache("openai/gsm8k", "main", split="train", use_cache_first=use_cache)
    random.seed(37)
    samples = random.sample(list(ds), args.num_samples)
    print(f"[INFO] dataset loaded: {len(samples)} samples")

    # build combo list (filtered)
    combos = []
    for df, reg in MODE_GROUPS:
        if df not in args.filter_dataflow:
            continue
        if reg not in args.filter_reg:
            continue
        for ber_ in args.filter_ber:
            if ber_ not in BER_VALUES:
                continue
            combos.append((df, reg, ber_))
    print(f"[INFO] {len(combos)} combinations to run")

    reg_map = {"input": 0, "weight": 1, "psum": 2}

    for df, reg, ber in combos:
        _run_combo(df, reg, ber, out_path, samples, tokenizer, model, reg_map,
                   inject_mode=args.mode)


if __name__ == "__main__":
    main()
