"""
Plot PE Position Severity Heatmaps for WS Dataflow.

Generates a 3x3 grid: rows = (weight, input, psum), cols = (attention, intermediate, output).
Psum is now row-dependent (injector scales by (K-r)/K).
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
JSON_PATH = os.path.join(CONFIG_DIR, "position_severity_ws.json")
OUT_PATH = os.path.join(CONFIG_DIR, "position_severity_ws_heatmap.png")

MODES = ["weight", "input", "psum"]
MODE_LABELS = {
    "weight": "Weight\n(uniform)",
    "input": "Input\n(rightward, C-c)",
    "psum": "Psum\n(uniform)",
}
LAYER_ORDER = ["attention", "intermediate", "output"]
LAYER_LABELS = {
    "attention": "Attention\nK=256, N=256",
    "intermediate": "Intermediate\nK=256, N=1024",
    "output": "Output\nK=1024, N=256",
}


def load_data():
    with open(JSON_PATH) as f:
        return json.load(f)


def plot():
    data = load_data()
    sev = data["severity"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    for row, mode in enumerate(MODES):
        for col, layer_name in enumerate(LAYER_ORDER):
            ax = axes[row, col]
            mat = np.array(sev[layer_name][mode]["normalized_matrix"])
            info = sev[layer_name][mode]

            if info["uniform"]:
                im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1,
                               aspect="equal", origin="lower")
                ax.text(128, 128, f"Uniform\n{info['raw_min']:.0f}",
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
            else:
                im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1,
                               aspect="equal", origin="lower")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Norm. Severity", fontsize=8)

            if row == 0:
                ax.set_title(LAYER_LABELS[layer_name], fontsize=11, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(MODE_LABELS[mode], fontsize=10, fontweight="bold", rotation=0,
                              labelpad=60, va="center")

            ax.set_xlabel("PE Col (c)" if row == 2 else "", fontsize=9)
            ax.set_ylabel("PE Row (r)" if col == 0 else "", fontsize=9)

            # Tick every 64
            ticks = np.arange(0, 257, 64)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "PE Position Severity — WS Dataflow | 256×256 Systolic Array\n"
        "boltuix/bert-emotion  |  weight / input / psum modes",
        fontsize=14, fontweight="bold", y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=100, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    plot()
