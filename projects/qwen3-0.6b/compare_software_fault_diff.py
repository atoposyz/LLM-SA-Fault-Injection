import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model ID
MODEL_ID = "Qwen/Qwen3-0.6B"

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

class HiddenCaptureHook:
    def __init__(self, name):
        self.name = name
        self.output = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach().cpu()
        else:
            self.output = output.detach().cpu()

class SoftwareFaultHook:
    """
    Directly injects a fault into the output tensor.
    """
    def __init__(self, target_seq, target_hidden, bit_pos=28):
        self.target_seq = target_seq
        self.target_hidden = target_hidden
        self.bit_pos = bit_pos

    def __call__(self, module, input, output):
        # Determine the tensor to modify
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output

        # Ensure we're injecting at a valid location
        batch, seq_len, hidden_dim = out_tensor.shape
        s = min(self.target_seq, seq_len - 1)
        h = min(self.target_hidden, hidden_dim - 1)

        # In-place modification to simulate the fault
        # We need to treat it as int32 for bit manipulation
        orig_dtype = out_tensor.dtype
        # print(f"[DEBUG] Injecting software fault at seq {s}, hidden {h}, bit {self.bit_pos}")

        # Clone to avoid affecting the original baseline run if needed, but here we run separate passes
        # Note: We must be careful with BFloat16 vs Float32
        out_tensor_f32 = out_tensor.float()
        t_int = out_tensor_f32.view(torch.int32)
        
        # Apply bit flip
        mask = 1 << self.bit_pos
        t_int[0, s, h] ^= mask
        
        # Convert back
        modified_out = t_int.view(torch.float32).to(orig_dtype)
        
        # To avoid side effects on other parts of the graph if needed, 
        # but here we just return the modified tensor.
        if isinstance(output, tuple):
            return (modified_out,) + output[1:]
        else:
            return modified_out

def run_software_test():
    import random
    random.seed(42)
    torch.manual_seed(42)

    print(f"[INFO] Initializing model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", trust_remote_code=True)
    model.eval()

    layer_0 = model.model.layers[0]
    last_layer_idx = model.config.num_hidden_layers - 1
    last_layer = model.model.layers[last_layer_idx]
    
    # Target for software injection: Layer 0 K-projection output
    injection_target = layer_0.self_attn.k_proj

    # Input prompt (same as the hardware test for comparison)
    prompt = """
Role: You are an expert senior researcher specializing in Computer Architecture...
""" # Shortened for brevity in the script, but will use the full one from before
    
    # We'll use the user provided longer prompt for more sequence indices
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

    hook_0 = HiddenCaptureHook("Layer 0")
    hook_last = HiddenCaptureHook(f"Layer {last_layer_idx}")
    
    # --- 1. Baseline ---
    print("[INFO] Running Baseline...")
    h0 = layer_0.register_forward_hook(hook_0)
    hl = last_layer.register_forward_hook(hook_last)
    
    with torch.no_grad():
        model(**inputs)
    
    baseline_0 = hook_0.output.clone()
    baseline_last = hook_last.output.clone()
    
    h0.remove()
    hl.remove()

    # --- 2. Software Fault Injection ---
    # Target: seq 10, hidden 128 (arbitrary but fixed)
    print("[INFO] Running Software Fault Injection at output [0, 10, 128]...")
    fault_hook = SoftwareFaultHook(target_seq=10, target_hidden=128, bit_pos=28)
    
    h_fault = injection_target.register_forward_hook(fault_hook)
    h0 = layer_0.register_forward_hook(hook_0)
    hl = last_layer.register_forward_hook(hook_last)
    
    with torch.no_grad():
        model(**inputs)
    
    faulty_0 = hook_0.output.clone()
    faulty_last = hook_last.output.clone()
    
    h_fault.remove()
    h0.remove()
    hl.remove()

    # --- 3. Analysis ---
    diff_0 = torch.abs(baseline_0 - faulty_0)
    diff_last = torch.abs(baseline_last - faulty_last)

    print("\n--- Software Fault Comparison ---")
    print(f"Layer 0   | Max Diff: {diff_0.max().item():.6f}")
    print(f"Last Layer| Max Diff: {diff_last.max().item():.6f}")

    # --- 4. Plotting ---
    os.makedirs("results/software_fault", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(diff_0[0].float().numpy(), ax=axes[0], cmap="YlOrRd")
    axes[0].set_title("Layer 0 Output Difference (Direct Software Injection)")
    axes[0].set_xlabel("Hidden Dimension")
    axes[0].set_ylabel("Token Index")

    sns.heatmap(diff_last[0].float().numpy(), ax=axes[1], cmap="YlOrRd")
    axes[1].set_title(f"Last Layer Output Difference\n(Propagation from Layer 0 Software Fault)")
    axes[1].set_xlabel("Hidden Dimension")
    axes[1].set_ylabel("Token Index")

    plt.tight_layout()
    output_img = "results/software_fault/software_fault_comparison.png"
    plt.savefig(output_img)
    plt.close()
    
    print(f"\n[SUCCESS] Heatmap saved to: {output_img}")

if __name__ == "__main__":
    run_software_test()
