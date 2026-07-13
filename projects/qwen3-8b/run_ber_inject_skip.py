"""
BER 多点故障注入脚本 + StableRank skip — Qwen3-8B。

与 run_ber_inject.py 相同的 BER 遍历逻辑，但将 model.generate() 替换为
逐 token 生成 + StableRank 异常检测。当检测到数值异常时，终止当前样本的
生成，保存 is_aborted=True 的结果，跳到下一个样本。

StableRank 检测逻辑来自 projects/rerun/SAskip.py。
"""

import argparse
import json
import operator
import os
import random
import time
from collections import deque

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
# StableRank skip infrastructure (from SAskip.py)
# ---------------------------------------------------------------------------

_CMP = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


class StopForwardException(Exception):
    """用于在 hook 中终止当前 forward 的异常。"""


def _sr_update_and_should_stop(stable_rank, cfg, state, is_rerun):
    cmp_fn = _CMP[cfg.get("cmp", ">=")]
    is_abnormal = bool(cmp_fn(stable_rank, cfg.get("threshold", 0.0)))
    mode = cfg.get("count_mode", "consecutive")
    k = cfg.get("trigger_k", 3)

    if mode == "consecutive":
        state["value"] = state["value"] + 1 if is_abnormal else 0
    elif mode == "cumulative":
        if is_abnormal:
            state["value"] += 1
    elif mode == "window":
        n = cfg.get("window_n", 10)
        buf = state.get("buf")
        if buf is None or buf.maxlen != n:
            buf = deque(maxlen=n)
            state["buf"] = buf
        buf.append(1 if is_abnormal else 0)
        state["value"] = sum(buf)

    return state["value"] >= k and not is_rerun


# Default SR config (overridable via CLI)
sr_cfg_lower = {
    "threshold": 1.2, "cmp": "<=", "count_mode": "consecutive",
    "trigger_k": 3, "window_n": 10, "detect_layer": 0,
}
sr_cfg_upper = {
    "threshold": 1.2, "cmp": ">=", "count_mode": "consecutive",
    "trigger_k": 3, "window_n": 10, "detect_layer": 0,
}
SR_INTERVAL = 50


def _make_cache_hook(layer_id, layer_output_cache):
    def hook(module, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(out_tensor, torch.Tensor):
            if out_tensor.dim() == 3:
                token_out = out_tensor[0, -1, :].detach().cpu().unsqueeze(0)
            elif out_tensor.dim() == 2:
                token_out = out_tensor[-1, :].detach().cpu().unsqueeze(0)
            else:
                return output
            layer_output_cache.setdefault(layer_id, []).append(token_out)
        return output
    return hook


def _make_check_hook(layer_output_cache, counter, cfg_lower, cfg_upper,
                    state_lower, state_upper, is_rerun_flag, triggered):
    def hook(module, input, output):
        counter[0] += 1
        if counter[0] % SR_INTERVAL != 0:
            return output

        for layer_id in sorted(layer_output_cache.keys()):
            outputs = layer_output_cache[layer_id]
            if not outputs:
                continue
            matrix = torch.cat(outputs, dim=0).float()
            total = matrix.shape[0]
            start = 0
            end = min(start + SR_INTERVAL, total)
            sub = matrix[start:end, :]
            if sub.numel() == 0:
                continue

            if not torch.isfinite(sub).all():
                sr = 0.0
            else:
                fro = torch.norm(sub, p='fro').pow(2).item()
                sig = torch.linalg.norm(sub, ord=2).pow(2).item()
                sr = fro / sig if sig > 1e-12 else 0.0

            for cfg, st, tag in [(cfg_lower, state_lower, "lower"),
                                  (cfg_upper, state_upper, "upper")]:
                if cfg["detect_layer"] != layer_id:
                    continue
                if _sr_update_and_should_stop(sr, cfg, st, is_rerun_flag[0]):
                    triggered[0] = tag
                    raise StopForwardException(
                        f"[{tag.upper()}] L{layer_id} tokens~{counter[0]} "
                        f"sr={sr:.4f} thr={cfg['threshold']}")
        layer_output_cache.clear()
        return output
    return hook


def _safe_decode(tokenizer, token_ids):
    ids = token_ids.detach().cpu().tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    clean = [int(t) for t in ids if 0 <= int(t) < len(tokenizer)]
    return tokenizer.decode(clean, skip_special_tokens=True), clean

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def get_module_by_path(module, path):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def ensure_dir(path_value):
    if not path_value.endswith("/"):
        path_value += "/"
    os.makedirs(path_value, exist_ok=True)
    return path_value


def load_model_and_tokenizer(model_id, use_cache_first=True):
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
# sample runner (with StableRank skip)
# ---------------------------------------------------------------------------


def _run_sample_skip(df, reg, ber, expected_faults, num_regs, reg_map, idx, sample,
                     injector, tokenizer, model, file_handle,
                     sr_lower_cfg, sr_upper_cfg, max_new_tokens,
                     cache_hooks, check_hook_handle,
                     layer_output_cache, token_counter, sr_state_lower,
                     sr_state_upper, is_rerun_flag, triggered):
    # --- fault setup (same as original) ---
    f_type = random.choice(["stuck_0", "stuck_1"])
    mode = random.choice(["input", "weight", "psum"]) if reg == "mixed" else reg
    injector.fault_type_str = f"{mode}_{f_type}_0"
    injector.parse_fault_type()

    if df == "random":
        injector.dataflow = random.choice(["WS", "OS", "IS"])

    if expected_faults >= 1.0:
        injector.init_faults_by_ber(ber, num_regs=num_regs, num_bits=32)
    else:
        injector.init_multi_fault_positions(1, num_regs=num_regs, num_bits=32)
        if not (random.random() < expected_faults):
            injector.reset_fault_pe()

    if reg != "mixed":
        r_id = reg_map[reg]
        injector.fault_reg = [r_id for _ in injector.fault_reg]

    # mount injection hooks on all layers
    handles = []
    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for kernel_name, kernel_path in pos_mapping.items():
            mod = get_module_by_path(layer, kernel_path)
            handles.append(mod.register_forward_hook(injector.hook_fn))

    # --- prompt ---
    messages = [{
        "role": "user",
        "content": ("Solve the following math problem step by step.\n"
                     + sample["question"] + "\nAnswer:")
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs["input_ids"].shape[-1]

    # --- reset SR state ---
    sr_state_lower["value"] = 0
    sr_state_lower["buf"] = None
    sr_state_upper["value"] = 0
    sr_state_upper["buf"] = None
    layer_output_cache.clear()
    token_counter[0] = 0
    is_rerun_flag[0] = 0
    triggered[0] = ""

    # --- per-token generation ---
    generated = inputs["input_ids"]
    past_key_values = None
    is_aborted = False
    t_start = time.time()

    try:
        with torch.no_grad():
            for _step in range(max_new_tokens):
                if past_key_values is None:
                    model_inputs = {"input_ids": generated}
                else:
                    model_inputs = {
                        "input_ids": generated[:, -1:],
                        "past_key_values": past_key_values,
                        "use_cache": True,
                    }
                outputs = model(**model_inputs)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                    break
    except StopForwardException:
        is_aborted = True

    elapsed = time.time() - t_start
    token_length = generated.shape[1] - num_tokens
    decoded, _ = _safe_decode(tokenizer, generated[0][num_tokens:])
    response = decoded.strip().replace("\n\n", "\n")

    result = {
        "sample_id": str(idx),
        "token_length": token_length,
        "is_aborted": is_aborted,
        "triggered_detector": triggered[0],
        "elapsed_time": round(elapsed, 3),
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


def _run_combo_skip(df, reg, ber, out_path, samples, tokenizer, model, reg_map,
                    inject_mode, sr_lower_cfg, sr_upper_cfg, max_new_tokens):
    ber_str = f"{ber:.0e}".replace("e-0", "e-").replace("e+0", "e")
    fname = f"ber_{ber_str}_df{df}_reg{reg}.jsonl"
    filepath = out_path + fname

    num_regs = 3 if reg == "mixed" else 1
    total_bits = 256 * 256 * num_regs * 32
    expected_faults = total_bits * ber

    random.seed(42)
    torch.manual_seed(42)

    resume_from = 0
    if os.path.exists(filepath):
        with open(filepath) as fp:
            resume_from = sum(1 for _ in fp)
        if resume_from >= len(samples):
            print(f"[SKIP] {fname} already complete ({resume_from}/{len(samples)})")
            return
        print(f"[RESUME] {fname} from sample {resume_from}/{len(samples)}")
        _consume_random(df, reg, ber, expected_faults, num_regs, resume_from)

    print(f"[RUN] {fname}  expected_faults={expected_faults:.3f}")

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

    # --- StableRank detection setup ---
    detect_layers = list(set([sr_lower_cfg["detect_layer"], sr_upper_cfg["detect_layer"]]))
    check_layer = max(detect_layers)

    layer_output_cache = {}
    token_counter = [0]
    sr_state_lower = {"value": 0, "buf": None}
    sr_state_upper = {"value": 0, "buf": None}
    is_rerun_flag = [0]
    triggered = [""]

    cache_handles = []
    for lid in detect_layers:
        h = model.model.layers[lid].register_forward_hook(
            _make_cache_hook(lid, layer_output_cache))
        cache_handles.append(h)
    check_h = model.model.layers[check_layer].register_forward_hook(
        _make_check_hook(layer_output_cache, token_counter,
                         sr_lower_cfg, sr_upper_cfg,
                         sr_state_lower, sr_state_upper,
                         is_rerun_flag, triggered))

    with open(filepath, "a", encoding="utf-8") as f:
        for idx in range(resume_from, len(samples)):
            _run_sample_skip(
                df, reg, ber, expected_faults, num_regs, reg_map,
                idx, samples[idx], injector, tokenizer, model, f,
                sr_lower_cfg, sr_upper_cfg, max_new_tokens,
                cache_handles, check_h,
                layer_output_cache, token_counter,
                sr_state_lower, sr_state_upper,
                is_rerun_flag, triggered)

    for h in cache_handles:
        h.remove()
    check_h.remove()
    print(f"[DONE] {fname}")


def _consume_random(df, reg, ber, expected_faults, num_regs, n):
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
    parser = argparse.ArgumentParser(description="Qwen3-8B BER Fault Injection + StableRank skip")
    parser.add_argument("--outputfile", "-o", type=str, default="./result")
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--filter-dataflow", type=str, nargs="*",
                        choices=["random", "IS", "OS", "WS"],
                        default=["random", "IS", "OS", "WS"])
    parser.add_argument("--filter-reg", type=str, nargs="*",
                        choices=["mixed", "input", "weight", "psum"],
                        default=["mixed", "input", "weight", "psum"])
    parser.add_argument("--filter-ber", type=float, nargs="*", default=BER_VALUES)
    parser.add_argument("--num-samples", "-n", type=int, default=200)
    parser.add_argument("--mode", type=str, choices=["exact", "grouped_exact", "coverage"],
                        default="exact")
    parser.add_argument("--no-cache-priority", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2000)

    # SR lower detector
    parser.add_argument("--sr-lower-layer", type=int, default=0)
    parser.add_argument("--sr-lower-threshold", type=float, default=1.2)
    parser.add_argument("--sr-lower-cmp", type=str, choices=[">=", "<=", ">", "<"], default="<=")
    parser.add_argument("--sr-lower-count-mode", type=str,
                        choices=["consecutive", "cumulative", "window"], default="consecutive")
    parser.add_argument("--sr-lower-trigger-k", type=int, default=3)
    parser.add_argument("--sr-lower-window-n", type=int, default=10)

    # SR upper detector
    parser.add_argument("--sr-upper-layer", type=int, default=0)
    parser.add_argument("--sr-upper-threshold", type=float, default=1.2)
    parser.add_argument("--sr-upper-cmp", type=str, choices=[">=", "<=", ">", "<"], default=">=")
    parser.add_argument("--sr-upper-count-mode", type=str,
                        choices=["consecutive", "cumulative", "window"], default="consecutive")
    parser.add_argument("--sr-upper-trigger-k", type=int, default=3)
    parser.add_argument("--sr-upper-window-n", type=int, default=10)

    args = parser.parse_args()

    sr_lower = {
        "detect_layer": args.sr_lower_layer,
        "threshold": args.sr_lower_threshold,
        "cmp": args.sr_lower_cmp,
        "count_mode": args.sr_lower_count_mode,
        "trigger_k": args.sr_lower_trigger_k,
        "window_n": args.sr_lower_window_n,
    }
    sr_upper = {
        "detect_layer": args.sr_upper_layer,
        "threshold": args.sr_upper_threshold,
        "cmp": args.sr_upper_cmp,
        "count_mode": args.sr_upper_count_mode,
        "trigger_k": args.sr_upper_trigger_k,
        "window_n": args.sr_upper_window_n,
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_path = ensure_dir(args.outputfile)
    use_cache = not args.no_cache_priority

    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_cache_first=use_cache)
    model.eval()
    print(f"[INFO] model loaded: {MODEL_ID}, layers={model.config.num_hidden_layers}")
    print(f"[INFO] SR lower: L{sr_lower['detect_layer']} thr={sr_lower['threshold']} "
          f"cmp={sr_lower['cmp']} {sr_lower['count_mode']} k={sr_lower['trigger_k']}")
    print(f"[INFO] SR upper: L{sr_upper['detect_layer']} thr={sr_upper['threshold']} "
          f"cmp={sr_upper['cmp']} {sr_upper['count_mode']} k={sr_upper['trigger_k']}")

    ds = load_dataset_with_cache("openai/gsm8k", "main", split="train", use_cache_first=use_cache)
    random.seed(37)
    samples = random.sample(list(ds), args.num_samples)
    print(f"[INFO] dataset loaded: {len(samples)} samples")

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
        _run_combo_skip(df, reg, ber, out_path, samples, tokenizer, model, reg_map,
                        inject_mode=args.mode,
                        sr_lower_cfg=sr_lower, sr_upper_cfg=sr_upper,
                        max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
