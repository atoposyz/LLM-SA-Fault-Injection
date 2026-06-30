import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Model configurations from the paper table
# ============================================================

models = [
    {"name": "0.6B",      "params_b": 0.6,   "layers": 28, "hidden": 1024, "ops": 426},
    {"name": "1.7B",      "params_b": 1.7,   "layers": 28, "hidden": 2048, "ops": 426},
    {"name": "4B",        "params_b": 4.0,   "layers": 36, "hidden": 2560, "ops": 546},
    {"name": "8B",        "params_b": 8.0,   "layers": 36, "hidden": 4096, "ops": 546},
    {"name": "14B",       "params_b": 14.0,  "layers": 40, "hidden": 5120, "ops": 606},
    {"name": "30B-A3B",   "params_b": 30.0,  "layers": 48, "hidden": 2048, "ops": 678},
    {"name": "32B",       "params_b": 32.0,  "layers": 64, "hidden": 5120, "ops": 966},
    {"name": "235B-A22B", "params_b": 235.0, "layers": 94, "hidden": 4096, "ops": 1322},
]


# ============================================================
# 2. Fault-injection settings
# ============================================================

samples_per_component = 1000
tokens_gen = 128
framework_overhead = 1.5

bytes_per_param = 2.0
kv_cache_factor = 0.20

compute_latency_floor = 0.005


# ============================================================
# 3. GPU bandwidth specifications
#    No memory-wall assumption: only HBM bandwidth is used.
# ============================================================

gpus = {
    "H100": {
        "bw": 3000,
        "color": "#2ca02c",
        "marker": "D",
        "ls": "-"
    },
    "A100": {
        "bw": 1500,
        "color": "#1f77b4",
        "marker": "o",
        "ls": "-"
    },
    "A30": {
        "bw": 933,
        "color": "#ff7f0e",
        "marker": "s",
        "ls": "--"
    },
    "RTX 4060 Ti": {
        "bw": 288,
        "color": "#d62728",
        "marker": "^",
        "ls": "-."
    },
}


# ============================================================
# 4. Runtime estimation
# ============================================================

def estimate_required_memory_gb(params_b):
    """
    Estimate model memory footprint under BF16/FP16.
    """
    weight_mem_gb = params_b * bytes_per_param
    total_mem_gb = weight_mem_gb * (1.0 + kv_cache_factor)
    return total_mem_gb


def estimate_token_latency_seconds(model, gpu_specs):
    """
    Estimate per-token latency without memory-wall/offloading.

    All data are assumed to be served by GPU HBM.
    """
    required_memory_gb = estimate_required_memory_gb(model["params_b"])

    memory_time = required_memory_gb / gpu_specs["bw"]

    layer_factor = model["layers"] / 28.0
    hidden_factor = model["hidden"] / 1024.0
    structural_factor = 0.15 * layer_factor * np.sqrt(hidden_factor)

    return memory_time + compute_latency_floor * (1.0 + structural_factor)


def estimate_single_inference_seconds(model, gpu_specs):
    """
    Runtime of one fault-injected inference execution.
    """
    t_token = estimate_token_latency_seconds(model, gpu_specs)
    return t_token * tokens_gen * framework_overhead


def estimate_total_fi_hours(model, gpu_specs, mode="operator"):
    """
    Total FI runtime.

    mode:
        model    : one global model-level evaluation
        layer    : evaluate each layer separately
        operator : evaluate each operator separately
    """
    if mode == "model":
        components = 1
    elif mode == "layer":
        components = model["layers"]
    elif mode == "operator":
        components = model["ops"]
    else:
        raise ValueError("mode must be one of: model, layer, operator")

    single_inference_seconds = estimate_single_inference_seconds(model, gpu_specs)

    total_seconds = (
        components
        * samples_per_component
        * single_inference_seconds
    )

    return total_seconds / 3600.0


def build_runtime_table(mode="operator"):
    rows = []

    for model in models:
        row = {
            "Model": model["name"],
            "Params(B)": model["params_b"],
            "Layers": model["layers"],
            "Hidden": model["hidden"],
            "Ops": model["ops"],
        }

        for gpu_name, gpu_specs in gpus.items():
            row[gpu_name] = estimate_total_fi_hours(
                model,
                gpu_specs,
                mode=mode
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 5. Plotting
# ============================================================

def setup_matplotlib():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def add_time_reference_lines(ax, x_indices):
    refs = {
        24: "1 Day",
        24 * 7: "1 Week",
        24 * 30: "1 Month",
        24 * 365: "1 Year",
    }

    for hours, label in refs.items():
        ax.axhline(
            y=hours,
            color="gray",
            linestyle=":",
            linewidth=1.3,
            alpha=0.8
        )
        ax.text(
            x_indices[-1] + 0.15,
            hours,
            label,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="dimgray"
        )


def draw_figure(mode="operator", y_scale="log", output_name="fi_runtime_no_memory_wall"):
    setup_matplotlib()
    os.makedirs("result", exist_ok=True)

    x_labels = [m["name"] for m in models]
    x_indices = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for gpu_name, gpu_specs in gpus.items():
        y = [
            estimate_total_fi_hours(model, gpu_specs, mode=mode)
            for model in models
        ]

        ax.plot(
            x_indices,
            y,
            marker=gpu_specs["marker"],
            linestyle=gpu_specs["ls"],
            markersize=8,
            linewidth=2.5,
            color=gpu_specs["color"],
            label=gpu_name
        )

    if y_scale == "log":
        ax.set_yscale("log")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontweight="bold")

    if mode == "operator":
        granularity = "Operator-wise"
    elif mode == "layer":
        granularity = "Layer-wise"
    else:
        granularity = "Model-level"

    ax.set_xlabel("Model Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Fault Injection Runtime (hours)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{granularity} Fault Injection Runtime"
        f"(n={samples_per_component} samples)",
        fontsize=15,
        pad=15
    )

    add_time_reference_lines(ax, x_indices)

    ax.set_xlim(-0.2, len(x_indices) - 0.25)

    ax.grid(True, which="major", linestyle="-", alpha=0.2)
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)

    ax.legend(
        loc="upper left",
        fontsize=10.5,
        framealpha=0.95,
        edgecolor="black"
    )

    plt.tight_layout()

    plt.savefig(f"result/{output_name}.pdf", format="pdf", dpi=300)
    plt.savefig(f"result/{output_name}.png", format="png", dpi=300)

    plt.close(fig)


def draw_granularity_comparison(
    gpu_name="A30",
    y_scale="log",
    output_name="fi_runtime_granularity_no_memory_wall"
):
    setup_matplotlib()
    os.makedirs("result", exist_ok=True)

    gpu_specs = gpus[gpu_name]

    x_labels = [m["name"] for m in models]
    x_indices = np.arange(len(models))

    modes = {
        "Model-level": {
            "mode": "model",
            "marker": "o",
            "ls": "-"
        },
        "Layer-wise": {
            "mode": "layer",
            "marker": "s",
            "ls": "--"
        },
        "Operator-wise": {
            "mode": "operator",
            "marker": "^",
            "ls": "-."
        },
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, cfg in modes.items():
        y = [
            estimate_total_fi_hours(model, gpu_specs, mode=cfg["mode"])
            for model in models
        ]

        ax.plot(
            x_indices,
            y,
            marker=cfg["marker"],
            linestyle=cfg["ls"],
            markersize=8,
            linewidth=2.5,
            label=label
        )

    if y_scale == "log":
        ax.set_yscale("log")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontweight="bold")

    ax.set_xlabel("Model Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Fault Injection Runtime (hours)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Effect of Evaluation Granularity without Memory Wall ({gpu_name})",
        fontsize=15,
        pad=15
    )

    add_time_reference_lines(ax, x_indices)

    ax.set_xlim(-0.2, len(x_indices) - 0.25)

    ax.grid(True, which="major", linestyle="-", alpha=0.2)
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)

    ax.legend(
        loc="upper left",
        fontsize=11,
        framealpha=0.95,
        edgecolor="black"
    )

    plt.tight_layout()

    plt.savefig(f"result/{output_name}.pdf", format="pdf", dpi=300)
    plt.savefig(f"result/{output_name}.png", format="png", dpi=300)

    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)

    df = build_runtime_table(mode="operator")
    df.to_csv("result/fi_runtime_operator_no_memory_wall_hours.csv", index=False)

    print("\nOperator-wise FI runtime without memory wall, in hours:")
    print(df.to_string(index=False))

    draw_figure(
        mode="operator",
        y_scale="log",
        output_name="fi_runtime_operator_no_memory_wall_log"
    )

    draw_figure(
        mode="operator",
        y_scale="linear",
        output_name="fi_runtime_operator_no_memory_wall_linear"
    )

    draw_granularity_comparison(
        gpu_name="A30",
        y_scale="log",
        output_name="fi_runtime_granularity_no_memory_wall_a30"
    )