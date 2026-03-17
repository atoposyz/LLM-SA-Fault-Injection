import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def slide_sum_2d(matrix, window_size=20):
    t = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, window_size, window_size))
    summed = torch.nn.functional.conv2d(t, kernel, stride=1, padding=0)
    return summed.squeeze().numpy()

def main():
    tensor_dir = "results/tensors"
    plot_dir = "results/k_matrix_comparison"
    os.makedirs(plot_dir, exist_ok=True)

    print("[INFO] Loading tensors...")
    try:
        baseline_k0 = torch.load(f"{tensor_dir}/baseline_k0.pt")
        baseline_kl = torch.load(f"{tensor_dir}/baseline_kl.pt")
        hw_fault_k0 = torch.load(f"{tensor_dir}/hw_fault_k0.pt")
        hw_fault_kl = torch.load(f"{tensor_dir}/hw_fault_kl.pt")
        sw_fault_k0 = torch.load(f"{tensor_dir}/sw_fault_k0.pt")
        sw_fault_kl = torch.load(f"{tensor_dir}/sw_fault_kl.pt")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find saved tensors. Please run the injection scripts first. Details: {e}")
        return

    print("[INFO] Calculating differences...")
    hw_diff0 = torch.abs(baseline_k0 - hw_fault_k0)
    hw_diffl = torch.abs(baseline_kl - hw_fault_kl)
    sw_diff0 = torch.abs(baseline_k0 - sw_fault_k0)
    sw_diffl = torch.abs(baseline_kl - sw_fault_kl)

    print("\n--- K-Matrix Variation Summary ---")
    print(f"HW Fault  | L0 Max Diff: {hw_diff0.max():.4f} | Last Max Diff: {hw_diffl.max():.4f}")
    print(f"SW Fault  | L0 Max Diff: {sw_diff0.max():.4f} | Last Max Diff: {sw_diffl.max():.4f}")

    # --- 1. HEATMAP VISUALIZATION ---
    print("[INFO] Generating basic heatmaps...")
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
    plt.savefig(f"{plot_dir}/k_matrix_comparison_heatmaps.png")
    plt.close()

    # --- 2. PROJECTION ANALYSIS ---
    print("[INFO] Generating 1D projection plots...")
    sw_matrix = sw_diff0[0].float().numpy()
    hw_matrix = hw_diff0[0].float().numpy()
    
    sw_token_proj = np.max(np.abs(sw_matrix), axis=1)
    sw_dim_proj = np.max(np.abs(sw_matrix), axis=0)
    hw_token_proj = np.max(np.abs(hw_matrix), axis=1)
    hw_dim_proj = np.max(np.abs(hw_matrix), axis=0)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    axes[0, 0].plot(sw_token_proj, color='red', linewidth=1.5)
    axes[0, 0].set_title("SW Fault: Error vs. Token Index", fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel("Max Abs Error")
    
    axes[0, 1].plot(sw_dim_proj, color='red', linewidth=1.5)
    axes[0, 1].set_title("SW Fault: Error vs. Hidden Dim", fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel("Max Abs Error")
    
    axes[1, 0].plot(hw_token_proj, color='midnightblue', linewidth=1.5)
    axes[1, 0].set_title("HW Fault: Error vs. Token Index", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Max Abs Error")
    
    axes[1, 1].plot(hw_dim_proj, color='midnightblue', linewidth=1.5)
    axes[1, 1].set_title("HW Fault: Error vs. Hidden Dim", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Max Abs Error")
    
    for ax in axes.flat:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/fault_projections_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. SLIDING WINDOW SUM ANALYSIS ---
    print("[INFO] Generating 20x20 sliding window heatmaps...")
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
    plt.savefig(f"{plot_dir}/sliding_window_sum_heatmaps.png")
    plt.close()
    
    print("[SUCCESS] All plots generated successfully!")

if __name__ == "__main__":
    main()