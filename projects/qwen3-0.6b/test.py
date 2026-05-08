import matplotlib.pyplot as plt
import numpy as np

# 1. 基础数据设定：用户指定的参数规模 (Billion)
parameters_billion = [0.6, 1.7, 4, 8, 14, 32]
x_labels = [f"{p}B" for p in parameters_billion]

# 统一 SFI 抽样数、生成Token数和框架损耗
n_samples = 3000
tokens_gen = 128
framework_overhead = 1.5
pcie_bw = 25  # PCIe 4.0 x16 实际有效带宽 (GB/s)，这是内存卸载的绝对瓶颈

# 2. 显卡规格设定 (显存大小 GB, 理论高带宽显存 HBM 有效带宽 GB/s)
gpus = {
    'H100 (80GB)':      {'vram': 80,  'bw': 3000, 'color': '#2ca02c', 'marker': 'D', 'ls': '-'},
    'A100 (80GB)':      {'vram': 80,  'bw': 1500, 'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
    'A30 (24GB)':       {'vram': 24,  'bw': 933,  'color': '#ff7f0e', 'marker': 's', 'ls': '--'},
    'RTX 4060 Ti (16GB)':{'vram': 16,  'bw': 288,  'color': '#d62728', 'marker': '^', 'ls': '-.'}
}

# 3. 科学计算算法核心
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

# 我们使用离散的X轴索引以保证图表美观，避免小模型挤在一起
x_indices = np.arange(len(parameters_billion))


def compute_times_hours(specs):
    times_hours = []

    for p in parameters_billion:
        # 估算模型运行所需的显存峰值：2 bytes (FP16) + 约 20% 的KV Cache与框架开销
        req_vram_gb = p * 2 * 1.2

        # 核心算法：检查是否撞击“显存墙 (Memory Wall)”触发卸载
        if req_vram_gb <= specs['vram']:
            # 不卸载：所有数据从极快的 GPU HBM 读取
            t_token = req_vram_gb / specs['bw']
        else:
            # 触发卸载 (Offloading)：显存里的部分用高带宽，放不下的部分走龟速 PCIe
            vram_part = specs['vram']
            offload_part = req_vram_gb - specs['vram']
            t_token = (vram_part / specs['bw']) + (offload_part / pcie_bw)
        
        # 加上常数级算力开销 (极其微小，重点是带宽)
        t_token += 0.005 
        
        # 计算 3000 次实验的总耗时 (单位：小时)
        t_inference = t_token * tokens_gen
        t_total_hours = (t_inference * n_samples * framework_overhead) / 3600
        times_hours.append(t_total_hours)

    return times_hours


def draw_figure(y_scale, ylabel, output_name):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for name, specs in gpus.items():
        times_hours = compute_times_hours(specs)
        ax.plot(x_indices, times_hours,
                marker=specs['marker'], linestyle=specs['ls'], markersize=8, linewidth=2.5,
                color=specs['color'], label=name)

    # 4. 图表美化与设置
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontweight='bold')

    ax.set_xlabel('Model Size (Parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('Scaling of Fault Injection Runtime Across Model Sizes (n=3000 samples)', fontsize=15, pad=15)

    # 添加极具说服力的“人类时间感知”辅助线
    time_references = {
        24: '1 Day',
        24 * 7: '1 Week',
        24 * 30: '1 Month',
        # 24 * 365: '1 Year'
    }
    for hrs, label in time_references.items():
        ax.axhline(y=hrs, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax.text(x_indices[-1] + 0.1, hrs, label, va='center', ha='left',
                fontsize=10, fontweight='bold', color='dimgray')

    # 调整X轴范围以容纳右侧文字
    ax.set_xlim(-0.2, len(x_indices) - 0.4)

    ax.grid(True, which="major", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)

    # 把图例放在左上角
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')

    # # 在图中关键点添加“显存墙”越界标记 (以A100为例)
    # ax.annotate('A100 Memory Wall Crossed\n(Triggering PCIe Offload)',
    #             xy=(3, 10), xytext=(2.5, 100),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
    #             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig(f'result/{output_name}.pdf', format='pdf', dpi=300)
    plt.savefig(f'result/{output_name}.png', format='png', dpi=300)
    plt.close(fig)


draw_figure('log', 'Total Fault Injection Runtime (hours, log scale)', 'fi_hardware_constraints_v3')
draw_figure('linear', 'Total Fault Injection Runtime (hours, linear scale)', 'fi_hardware_constraints_v3_linear')
