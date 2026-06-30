"""
Direct bit injection baseline — corrupt linear layer outputs.

Injects bit errors directly into each linear layer's output tensor
(equivalent to psum injection in SA terms), with no PE grid mapping.
Contrast with run_ber_inject.py which simulates SA hardware topology.

Usage:
    uv run python projects/qwen3-8b/run_direct_inject.py -o result/ -g 0
    uv run python projects/qwen3-8b/run_direct_inject.py --filter-ber 1e-8 1e-9 -n 10
"""

import argparse
import json
import os
import random
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tool", "src"))
from tool.direct_injector import DirectBitInjector

logging.set_verbosity_error()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen3-8B"
BER_VALUES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

LAYER_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "mlp-gate": "mlp.gate_proj",
    "mlp-up": "mlp.up_proj",
    "mlp-down": "mlp.down_proj",
}


# ---------------------------------------------------------------------------
# helpers
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
# sample runner
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
# combo runner
# ---------------------------------------------------------------------------

def _run_combo(ber: float, out_path: str, samples, tokenizer, model):
    ber_str = f"{ber:.0e}".replace("e-0", "e-").replace("e+0", "e")
    fname = f"ber_{ber_str}_direct_output.jsonl"
    filepath = out_path + fname

    if os.path.exists(filepath):
        print(f"[SKIP] {fname} exists")
        return

    random.seed(42)
    torch.manual_seed(42)

    injector = DirectBitInjector(op="random", precision="bf16")
    injector.enabled = True

    print(f"[RUN] {fname}  ber={ber}")

    # Register hooks on all linear layers — each operator gets a unique
    # op_id so faults are sampled independently across the global bit space.
    num_layers = model.config.num_hidden_layers
    handles = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for kernel_name, kernel_path in LAYER_PATHS.items():
            try:
                mod = get_module_by_path(layer, kernel_path)
                op_id = f"L{layer_idx}_{kernel_name}"
                handles.append(mod.register_forward_hook(injector.make_hook(op_id)))
            except AttributeError:
                continue

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            for idx, sample in enumerate(tqdm(samples, desc=fname)):
                injector.reset_for_sample(ber, seed=42 + idx)
                gen = _generate(sample, tokenizer, model)

                result = {
                    "sample_id": str(idx),
                    "reference_answer": sample["answer"],
                    **gen,
                    "fault_type": "output_random_0",
                    "dataflow": "direct",
                    "ber": ber,
                    "fix_reg": "output",
                    "inject_layer": "all",
                    "inject_kernel": "all",
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        for h in handles:
            h.remove()

    print(f"[DONE] {fname}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Direct Bit Injection — Qwen3-8B output corruption")
    parser.add_argument("--outputfile", "-o", type=str, default="./result")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--filter-ber", type=float, nargs="*", default=BER_VALUES)
    parser.add_argument("--num-samples", "-n", type=int, default=200)
    parser.add_argument("--no-cache-priority", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_path = ensure_dir(args.outputfile)
    use_cache = not args.no_cache_priority

    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_cache_first=use_cache)
    model.eval()
    print(f"[INFO] model loaded: {MODEL_ID}, layers={model.config.num_hidden_layers}")

    ds = load_dataset_with_cache("openai/gsm8k", "main", split="train", use_cache_first=use_cache)
    random.seed(37)
    samples = random.sample(list(ds), args.num_samples)
    print(f"[INFO] dataset loaded: {len(samples)} samples")

    ber_list = [b for b in args.filter_ber if b in BER_VALUES]
    print(f"[INFO] {len(ber_list)} BER levels to run")

    for ber in ber_list:
        _run_combo(ber, out_path, samples, tokenizer, model)


if __name__ == "__main__":
    main()
