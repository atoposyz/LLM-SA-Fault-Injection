import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
tool_src_path = os.path.join(project_root, "tool", "src")
if tool_src_path not in sys.path:
    sys.path.append(tool_src_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HiddenCaptureHook:
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach().cpu()
        else:
            self.output = output.detach().cpu()


def compare_outputs():
    import random
    random.seed(42)
    torch.manual_seed(42)

    model_id = "Qwen/Qwen3-0.6B"
    print(f"[INFO] Initializing model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    model.eval()

    layer_0 = model.model.layers[0]
    injection_target = layer_0.self_attn.v_proj

    # --- WS 8x8 PE(2,3), add rand [1,10] at affected positions ---
    PE_R, PE_C = 8, 8
    pe_row, pe_col = 2, 3
    ADD_MIN, ADD_MAX = 1.0, 10.0
    print(f"[CONFIG] PE {PE_R}x{PE_C} | fault PE({pe_row},{pe_col}) | add rand [{ADD_MIN}, {ADD_MAX}]")

    prompt = """
Role: You are an expert senior researcher specializing in Computer Architecture, specifically focusing on the intersection of Large Language Models (LLMs) and hardware reliability. Your task is to provide a comprehensive, 5-stage critical review of a hypothetical research paper titled "Characterizing Soft Errors in Systolic Array-based LLM Accelerators: A Layer-wise Fault Injection Study."

Context for the Task:
The paper claims that by performing fault injection on the output matrices (hidden states) of specific Transformer layers, one can predict the overall system failure rate without full-model simulation. As a reviewer, you need to be extremely pedantic and rigorous.

Stage 1: Technical Summary (Approx. 200 words)
Summarize the core methodology of injecting bit-flips into the hidden state tensors. Address how the authors handle the [Batch, Sequence, Hidden] dimension. Discuss the implications of injecting faults into the linear attention layers of a Qwen-like architecture, specifically considering the hybrid structure where every 4 layers consist of 3 linear attention layers and 1 self-attention layer. Explain if the authors' claim about systolic array efficiency under fault conditions holds architectural merit.

Stage 2: Methodology Critique (Approx. 300 words)
Analyze the fault model. Is a simple bit-flip in the FP16 exponent enough? Or should they consider permanent hardware faults in the PE (Processing Element) arrays of the GPU? Evaluate their use of PyTorch hooks for fault injection. Does capturing the output_tuple at the layer level accurately reflect transient errors in the MAC (Multiply-Accumulate) units within a systolic array? You must provide at least three specific scenarios where their layer-wise abstraction fails to capture the low-level hardware behavior of a Tensor Core or a specialized AI accelerator.

Stage 3: Data Integrity and Statistical Significance (Approx. 200 words)
The authors used a dataset of 10,000 prompts but only injected faults into the first 10 tokens of each sequence. Critique this choice heavily. Given that the output matrix's vertical axis (Sequence Length) represents the token index, how does error propagation differ between the 'prefill' stage and the 'decoding' stage? Demand a more detailed breakdown of the Error Propagation Probability (EPP) across different hidden dimensions and positional encodings.

Stage 4: Architectural Impact (Approx. 200 words)
Discuss the hardware-software co-design implications. If we identify that specific layers (e.g., the 1/4 self-attention layers) are more "vulnerable" to faults than the linear attention layers, what hardware-level protection mechanisms (like Selective ECC or Modular Redundancy) should be implemented in the next generation of AI chips? Compare this to standard ISO 26262 automotive safety standards for ASIL-D compliance in autonomous driving LLM applications.

Stage 5: Final Verdict and Revision Requests (Approx. 150 words)
Provide a structured list of 10 mandatory revisions. Each revision must be technically dense, referencing specific concepts like "quantization-aware fault tolerance," "RoPE (Rotary Positional Embeddings) sensitivity to bit-flips," and "gradient-based vulnerability ranking."

Constraints for the Output:
- Use professional, academic English.
- Use LaTeX for any mathematical notations or formal logic.
- Ensure the tone is critical yet constructive.
- Do not provide a summary; go straight into the Stage 1 review.
- Elaborate on each point to ensure the output length is substantial.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # --- 1. Baseline ---
    hook_v = HiddenCaptureHook()
    print("[INFO] Running Baseline...")
    h = injection_target.register_forward_hook(hook_v)
    with torch.no_grad():
        model(**inputs)
    baseline = hook_v.output.clone()
    h.remove()

    # --- 2. Faulty Passes: OS 3 reg types, add +5 at PE-mapped positions ---
    MODES = [
        ('os_input',  'OS Input Register'),
        ('os_weight', 'OS Weight Register'),
        ('os_psum',   'OS Psum Register'),
        ('ws_input',  'WS Input Register'),
        ('ws_weight', 'WS Weight Register'),
        ('ws_psum',   'WS Psum Register'),
    ]
    results_dir = os.path.join(os.path.dirname(__file__), "comparison_results", "comparison")
    os.makedirs(results_dir, exist_ok=True)

    for mode, label in MODES:
        print(f"\n[INFO] Faulty Pass: {label} (rand [{ADD_MIN}, {ADD_MAX}])...")

        def make_add_hook(mode, r, c):
            def add_hook(module, inp, out):
                Y = out[0].clone() if isinstance(out, tuple) else out.clone()
                M, N = Y.shape[1], Y.shape[2]
                device = Y.device
                i_mod = torch.arange(M, device=device) % PE_R
                j_mod = torch.arange(N, device=device) % PE_C
                df, reg = mode.split('_')
                if df == 'os':
                    if reg == 'input':
                        mask = (i_mod.unsqueeze(1) == r) & (j_mod.unsqueeze(0) >= c)
                    elif reg == 'weight':
                        mask = (i_mod.unsqueeze(1) >= r) & (j_mod.unsqueeze(0) == c)
                    elif reg == 'psum':
                        mask = (i_mod.unsqueeze(1) == r) & (j_mod.unsqueeze(0) == c)
                elif df == 'ws':
                    if reg == 'input':
                        mask = j_mod.unsqueeze(0) >= c
                    elif reg == 'weight':
                        mask = j_mod.unsqueeze(0) == c
                    elif reg == 'psum':
                        mask = j_mod.unsqueeze(0) == c
                rand_vals = ADD_MIN + (ADD_MAX - ADD_MIN) * torch.rand(M, N, device=device)
                Y = Y + (mask * rand_vals).to(Y.dtype)
                return Y
            return add_hook

        handle_inj = injection_target.register_forward_hook(make_add_hook(mode, pe_row, pe_col))
        handle_cap = injection_target.register_forward_hook(hook_v)

        with torch.no_grad():
            model(**inputs)

        faulty = hook_v.output.clone()
        handle_inj.remove()
        handle_cap.remove()

        diff = torch.abs(baseline - faulty)
        print(f"  Max Diff: {diff.max().item():.2f}  Mean Diff: {diff.mean().item():.4f}")

        # Top-left 32x32 raw diff
        raw = diff[0, :32, :32]
        print(f"  Top-left 32x32 ({label}):")
        print("    " + "".join(f"{j:4d}" for j in range(32)))
        for r in range(32):
            row_vals = " ".join(f"{raw[r,j].item():4.0f}" for j in range(32))
            print(f"  {r:2d}: {row_vals}")

        # PE schematic heatmap (log scale)
        crop = torch.log1p(raw)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        sns.heatmap(crop.float().numpy(), ax=ax, cmap="Reds",
                    linewidths=0.5, linecolor='#f0f0f0', square=True,
                    cbar=False, xticklabels=False, yticklabels=False)
        plt.tight_layout(pad=0)
        img_png = os.path.join(results_dir, f"{mode}_add1-10.png")
        img_pdf = os.path.join(results_dir, f"{mode}_add1-10.pdf")
        plt.savefig(img_png, dpi=600)
        plt.savefig(img_pdf)
        plt.close()
        print(f"  [SAVED] {img_png}")
        print(f"  [SAVED] {img_pdf}")


if __name__ == "__main__":
    compare_outputs()
