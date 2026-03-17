import os
import sys
import torch
import numpy as np  # [修复]: 移至顶部，确保全局可用
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Environment Setup ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
tool_src_path = os.path.join(project_root, "tool", "src")
if tool_src_path not in sys.path:
    sys.path.append(tool_src_path)

from tool.single_bit_injector import SingleBit_Fast_SA_FaultInjector

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

class HiddenCaptureHook:
    def __init__(self, name):
        self.name = name
        self.output = None

    def __call__(self, module, input, output):
        # K-proj output is usually just a tensor
        if isinstance(output, tuple):
            self.output = output[0].detach().cpu()
        else:
            self.output = output.detach().cpu()

class SoftwareFaultHook:
    def __init__(self, target_seq, target_hidden, bit_pos=28):
        self.target_seq = target_seq
        self.target_hidden = target_hidden
        self.bit_pos = bit_pos

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output
        
        orig_dtype = out_tensor.dtype
        out_tensor_f32 = out_tensor.float()
        t_int = out_tensor_f32.view(torch.int32)
        
        mask = 1 << self.bit_pos
        # Inject at batch 0, target_seq, target_hidden
        s = min(self.target_seq, out_tensor.shape[1] - 1)
        h = min(self.target_hidden, out_tensor.shape[2] - 1)
        t_int[0, s, h] ^= mask
        
        modified_out = t_int.view(torch.float32).to(orig_dtype)
        if isinstance(output, tuple):
            return (modified_out,) + output[1:]
        else:
            return modified_out

def run_k_matrix_comparison():
    import random
    random.seed(42)
    torch.manual_seed(42)

    model_id = "Qwen/Qwen3-0.6B"
    print(f"[INFO] Initializing model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    model.eval()

    # Configuration for Layers
    layer_0 = model.model.layers[0]
    last_layer_idx = model.config.num_hidden_layers - 1
    last_layer = model.model.layers[last_layer_idx]
    
    # Target: K projection modules
    k_proj_0 = layer_0.self_attn.k_proj
    k_proj_last = last_layer.self_attn.k_proj

    # --- INPUT PROMPT ---
    prompt = """Role: You are an expert senior researcher specializing in Computer Architecture, specifically focusing on the intersection of Large Language Models (LLMs) and hardware reliability. Your task is to provide a comprehensive, 5-stage critical review of a hypothetical research paper titled "Characterizing Soft Errors in Systolic Array-based LLM Accelerators: A Layer-wise Fault Injection Study."

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
- Elaborate on each point to ensure the output length is substantial."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Hooks for capturing K-matrices
    hook_k0 = HiddenCaptureHook("K Matrix Layer 0")
    hook_kl = HiddenCaptureHook(f"K Matrix Layer {last_layer_idx}")
    
    # ---------------------------------------------------------
    # 1. BASELINE RUN
    # ---------------------------------------------------------
    print("[INFO] Running Baseline...")
    h0 = k_proj_0.register_forward_hook(hook_k0)
    hl = k_proj_last.register_forward_hook(hook_kl)
    with torch.no_grad():
        model(**inputs)
    baseline_k0 = hook_k0.output.clone()
    baseline_kl = hook_kl.output.clone()
    h0.remove(); hl.remove()

    # ---------------------------------------------------------
    # 2. HARDWARE FAULT (Weight Inject at Layer 0 K-proj)
    # ---------------------------------------------------------
    print("[INFO] Running Hardware Fault Injection (Layer 0 K-proj weight)...")
    
    # [核心修复 1]: 备份 Layer 0 K-proj 的原始权重
    orig_k_proj_0_weight = k_proj_0.weight.data.clone()

    hw_injector = SingleBit_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow="WS", 
        fault_type="weight_bitflip_28", precision='fp32'
    )
    hw_injector.init_random_fault()
    hw_injector.fault_reg[0] = 1 # Weight
    
    h_hw = k_proj_0.register_forward_hook(hw_injector.hook_fn)
    h0 = k_proj_0.register_forward_hook(hook_k0)
    hl = k_proj_last.register_forward_hook(hook_kl)
    with torch.no_grad():
        model(**inputs)
    hw_fault_k0 = hook_k0.output.clone()
    hw_fault_kl = hook_kl.output.clone()
    h_hw.remove(); h0.remove(); hl.remove()

    # [核心修复 2]: 强制将权重恢复至未污染状态，防止状态泄漏到下一个实验
    k_proj_0.weight.data.copy_(orig_k_proj_0_weight)

    # ---------------------------------------------------------
    # 3. SOFTWARE FAULT (Output Inject at Layer 0 K-proj)
    # ---------------------------------------------------------
    print("[INFO] Running Software Fault Injection (Layer 0 K-proj output at [0, 10, 128])...")
    sw_hook = SoftwareFaultHook(target_seq=10, target_hidden=128, bit_pos=28)
    
    h_sw = k_proj_0.register_forward_hook(sw_hook)
    h0 = k_proj_0.register_forward_hook(hook_k0)
    hl = k_proj_last.register_forward_hook(hook_kl)
    with torch.no_grad():
        model(**inputs)
    sw_fault_k0 = hook_k0.output.clone()
    sw_fault_kl = hook_kl.output.clone()
    h_sw.remove(); h0.remove(); hl.remove()

    # ---------------------------------------------------------
    # 4. ANALYSIS & PLOTTING
    # ---------------------------------------------------------
    os.makedirs("results/k_matrix_comparison", exist_ok=True)
    
    hw_diff0 = torch.abs(baseline_k0 - hw_fault_k0)
    hw_diffl = torch.abs(baseline_kl - hw_fault_kl)
    sw_diff0 = torch.abs(baseline_k0 - sw_fault_k0)
    sw_diffl = torch.abs(baseline_kl - sw_fault_kl)

    print("\n--- K-Matrix Variation Summary ---")
    print(f"HW Fault  | L0 Max Diff: {hw_diff0.max():.4f} | Last Max Diff: {hw_diffl.max():.4f}")
    print(f"SW Fault  | L0 Max Diff: {sw_diff0.max():.4f} | Last Max Diff: {sw_diffl.max():.4f}")

    # ---------------------------------------------------------
    # 5. HEATMAP VISUALIZATION (2x2 Grid)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    sns.heatmap(hw_diff0[0].float().numpy(), ax=axes[0,0], cmap="YlOrRd")
    axes[0,0].set_title("Hardware Fault: Layer 0 K Matrix Diff")
    
    sns.heatmap(hw_diffl[0].float().numpy(), ax=axes[0,1], cmap="YlOrRd")
    axes[0,1].set_title("Hardware Fault: Last Layer K Matrix Diff")

    sns.heatmap(sw_diff0[0].float().numpy(), ax=axes[1,0], cmap="YlOrRd")
    axes[1,0].set_title("Software Fault: Layer 0 K Matrix Diff")
    
    sns.heatmap(sw_diffl[0].float().numpy(), ax=axes[1,1], cmap="YlOrRd")
    axes[1,1].set_title("Software Fault: Last Layer K Matrix Diff")

    plt.tight_layout()
    plt.savefig("results/k_matrix_comparison/k_matrix_comparison_heatmaps.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 6. PROJECTION ANALYSIS (Strict 2x2 Layout)
    # ---------------------------------------------------------
    sw_matrix = sw_diff0[0].float().numpy()
    hw_matrix = hw_diff0[0].float().numpy()
    
    # 1D Projections
    sw_token_proj = np.max(np.abs(sw_matrix), axis=1)
    sw_dim_proj = np.max(np.abs(sw_matrix), axis=0)
    hw_token_proj = np.max(np.abs(hw_matrix), axis=1)
    hw_dim_proj = np.max(np.abs(hw_matrix), axis=0)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Row 1: Software Injection Analysis
    axes[0, 0].plot(sw_token_proj, color='red', linewidth=1.5)
    axes[0, 0].set_title("SW Fault: Error vs. Token Index (Expect Single Spike)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Token Index")
    axes[0, 0].set_ylabel("Max Abs Error")
    
    axes[0, 1].plot(sw_dim_proj, color='red', linewidth=1.5)
    axes[0, 1].set_title("SW Fault: Error vs. Hidden Dim (Expect Single Spike)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Hidden Dimension")
    axes[0, 1].set_ylabel("Max Abs Error")
    
    # Row 2: Hardware Injection Analysis
    axes[1, 0].plot(hw_token_proj, color='midnightblue', linewidth=1.5)
    axes[1, 0].set_title("HW Fault: Error vs. Token Index (Expect Plateau/Step)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Token Index")
    axes[1, 0].set_ylabel("Max Abs Error")
    
    axes[1, 1].plot(hw_dim_proj, color='midnightblue', linewidth=1.5)
    axes[1, 1].set_title("HW Fault: Error vs. Hidden Dim (Expect Single Spike)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Hidden Dimension")
    axes[1, 1].set_ylabel("Max Abs Error")
    
    for ax in axes.flat:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    proj_output = "results/k_matrix_comparison/fault_projections_analysis.png"
    plt.savefig(proj_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Heatmaps saved to: results/k_matrix_comparison/k_matrix_comparison_heatmaps.png")
    print(f"[SUCCESS] Projection analysis plot saved to: {proj_output}")

    # ---------------------------------------------------------
    # 7. SLIDING WINDOW SUM ANALYSIS (20x20)
    # ---------------------------------------------------------
    print("[INFO] Calculating 20x20 sliding window sums...")
    
    def slide_sum_2d(matrix, window_size=20):
        t = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, window_size, window_size))
        summed = torch.nn.functional.conv2d(t, kernel, stride=1, padding=0)
        return summed.squeeze().numpy()

    sw_sum0 = slide_sum_2d(sw_diff0[0].float().numpy())
    sw_suml = slide_sum_2d(sw_diffl[0].float().numpy())
    hw_sum0 = slide_sum_2d(hw_diff0[0].float().numpy())
    hw_suml = slide_sum_2d(hw_diffl[0].float().numpy())

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    sns.heatmap(hw_sum0, ax=axes[0,0], cmap="YlOrRd")
    axes[0,0].set_title("Hardware Fault (20x20 Sum): Layer 0")
    
    sns.heatmap(hw_suml, ax=axes[0,1], cmap="YlOrRd")
    axes[0,1].set_title("Hardware Fault (20x20 Sum): Last Layer")

    sns.heatmap(sw_sum0, ax=axes[1,0], cmap="YlOrRd")
    axes[1,0].set_title("Software Fault (20x20 Sum): Layer 0")
    
    sns.heatmap(sw_suml, ax=axes[1,1], cmap="YlOrRd")
    axes[1,1].set_title("Software Fault (20x20 Sum): Last Layer")

    plt.tight_layout()
    sum_output = "results/k_matrix_comparison/sliding_window_sum_heatmaps.png"
    plt.savefig(sum_output)
    plt.close()
    
    print(f"[SUCCESS] Sliding window sum heatmaps saved to: {sum_output}")

if __name__ == "__main__":
    run_k_matrix_comparison()