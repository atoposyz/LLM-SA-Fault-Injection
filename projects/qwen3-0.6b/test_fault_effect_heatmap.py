import os
import sys
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add tool/src to sys.path to allow importing tool package
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
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        # Hidden states are typically the first element of the output tuple
        if isinstance(output, tuple):
            self.output = output[0].detach().cpu()
        else:
            self.output = output.detach().cpu()

def run_test():
    import random
    random.seed(42)
    torch.manual_seed(42)

    model_id = "Qwen/Qwen3-0.6B"
    print(f"[INFO] Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    model.eval()

    # Define layers to capture
    layer_0 = model.model.layers[0]
    last_layer_idx = model.config.num_hidden_layers - 1
    last_layer = model.model.layers[last_layer_idx]
    
    # Injection target: Layer 0, K projection
    injection_target_path = "self_attn.k_proj"
    injection_target = get_module_by_path(layer_0, injection_target_path)

    # Initialize Injector
    injector = SingleBit_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow="WS", 
        fault_type="weight_bitflip_28", precision='fp32'
    )
    injector.init_random_fault()
    injector.fault_reg[0] = 1 # Weight
    print(f"[INFO] Injecting at Row: {injector.fault_pe_row[0]}, Col: {injector.fault_pe_col[0]}, Bit: {injector.fault_bit[0]}")

    # Prepare input
    # prompt = "Hello, how are you today?"
    prompt ="""
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

    # Hooks for capturing hidden states
    hook_layer_0 = HiddenCaptureHook()
    hook_last_layer = HiddenCaptureHook()
    
    h0 = layer_0.register_forward_hook(hook_layer_0)
    hl = last_layer.register_forward_hook(hook_last_layer)

    # --- Baseline Run ---
    print("[INFO] Running baseline...")
    with torch.no_grad():
        model(**inputs)
    
    baseline_hidden_0 = hook_layer_0.output
    baseline_hidden_last = hook_layer_0.output # Wait, I need to capture both
    # Re-assigning to avoid confusion if the hook updates
    baseline_hidden_0 = hook_layer_0.output.clone()
    baseline_hidden_last = hook_last_layer.output.clone()

    # --- Faulty Run ---
    print("[INFO] Running faulty run (Injecting into Layer 0 K)...")
    h_inj = injection_target.register_forward_hook(injector.hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    
    faulty_hidden_0 = hook_layer_0.output.clone()
    faulty_hidden_last = hook_last_layer.output.clone()
    
    h_inj.remove()
    h0.remove()
    hl.remove()

    # --- Comparison ---
    diff_0 = torch.abs(baseline_hidden_0 - faulty_hidden_0)
    diff_last = torch.abs(baseline_hidden_last - faulty_hidden_last)

    print(f"Layer 0 Max Diff: {diff_0.max().item()}")
    print(f"Last Layer Max Diff: {diff_last.max().item()}")

    # --- Heatmaps ---
    os.makedirs("results/heatmaps", exist_ok=True)
    
    def plot_heatmap(diff_tensor, title, filename):
        # diff_tensor shape: [batch, seq_len, hidden_size]
        # We take the first sample in batch and squeeze seq_len x hidden_size
        data = diff_tensor[0].float().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(data, cmap="YlOrRd")
        plt.title(title)
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Sequence Index")
        plt.savefig(filename)
        plt.close()
        print(f"[INFO] Saved heatmap to {filename}")

    plot_heatmap(diff_0, f"Layer 0 Output Difference (Injected K at Layer 0)", "results/heatmaps/diff_layer_0.png")
    plot_heatmap(diff_last, f"Last Layer Output Difference (Injected K at Layer 0)", "results/heatmaps/diff_last_layer.png")

if __name__ == "__main__":
    run_test()
