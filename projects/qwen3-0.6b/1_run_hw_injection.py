import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Environment Setup ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
tool_src_path = os.path.join(project_root, "tool", "src")
if tool_src_path not in sys.path:
    sys.path.append(tool_src_path)

from tool.single_bit_injector import SingleBit_Fast_SA_FaultInjector

class HiddenCaptureHook:
    def __init__(self, name):
        self.name = name
        self.output = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach().cpu()
        else:
            self.output = output.detach().cpu()

def main():
    torch.manual_seed(42)
    os.makedirs("results/tensors", exist_ok=True)

    model_id = "Qwen/Qwen3-0.6B"
    print(f"[INFO] Initializing model {model_id} for Hardware Injection...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    model.eval()

    layer_0 = model.model.layers[0]
    last_layer_idx = model.config.num_hidden_layers - 1
    last_layer = model.model.layers[last_layer_idx]
    k_proj_0 = layer_0.self_attn.k_proj
    k_proj_last = last_layer.self_attn.k_proj

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

    hook_k0 = HiddenCaptureHook("K Matrix Layer 0")
    hook_kl = HiddenCaptureHook(f"K Matrix Layer {last_layer_idx}")

    # 1. BASELINE RUN
    print("[INFO] Running Baseline...")
    h0 = k_proj_0.register_forward_hook(hook_k0)
    hl = k_proj_last.register_forward_hook(hook_kl)
    with torch.no_grad():
        model(**inputs)
    baseline_k0 = hook_k0.output.clone()
    baseline_kl = hook_kl.output.clone()
    h0.remove(); hl.remove()

    # Save Baseline Tensors
    torch.save(baseline_k0, "results/tensors/baseline_k0.pt")
    torch.save(baseline_kl, "results/tensors/baseline_kl.pt")
    print("[SUCCESS] Baseline tensors saved.")

    # 2. HARDWARE FAULT RUN
    print("[INFO] Running Hardware Fault Injection...")
    orig_k_proj_0_weight = k_proj_0.weight.data.clone() # Backup

    hw_injector = SingleBit_Fast_SA_FaultInjector(
        sa_rows=256, sa_cols=256, dataflow="WS", 
        fault_type="weight_bitflip_28", precision='fp32'
    )
    hw_injector.set_specific_fault(5, 67)
    hw_injector.fault_reg[0] = 1 
    
    h_hw = k_proj_0.register_forward_hook(hw_injector.hook_fn)
    h0 = k_proj_0.register_forward_hook(hook_k0)
    hl = k_proj_last.register_forward_hook(hook_kl)
    with torch.no_grad():
        model(**inputs)
    hw_fault_k0 = hook_k0.output.clone()
    hw_fault_kl = hook_kl.output.clone()
    h_hw.remove(); h0.remove(); hl.remove()

    k_proj_0.weight.data.copy_(orig_k_proj_0_weight) # Restore

    # Save HW Fault Tensors
    torch.save(hw_fault_k0, "results/tensors/hw_fault_k0.pt")
    torch.save(hw_fault_kl, "results/tensors/hw_fault_kl.pt")
    print("[SUCCESS] Hardware fault tensors saved.")

if __name__ == "__main__":
    main()