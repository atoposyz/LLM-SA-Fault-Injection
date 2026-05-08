from contextlib import redirect_stdout
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def print_model_layers(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent + f"{name} ({module.__class__.__name__})")
        if list(module.named_children()):
            print_model_layers(module, indent + 4)
        else:
            if hasattr(module, "weight") and module.weight is not None:
                print(" " * (indent + 4) + f"weight: {module.weight.shape}")


def default_project_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-")


def get_operator_count(model) -> int:
    return sum(1 for name, _module in model.named_modules() if name)


def export_model_layers(model_id: str, output_dir: Path) -> None:
    print(f"[INFO] Exporting layer info for {model_id} -> {output_dir}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_layers = getattr(model.config, "num_hidden_layers", "unknown")
    hidden_size = getattr(model.config, "hidden_size", "unknown")
    operator_count = get_operator_count(model)
    print(f"[INFO] num_hidden_layers={hidden_layers}")
    print(f"[INFO] hidden_size={hidden_size}")
    print(f"[INFO] operator_count={operator_count}")

    with open(output_dir / "layerstructure.txt", "w") as f1:
        with redirect_stdout(f1):
            print_model_layers(model)

    with open(output_dir / "layernames.txt", "w") as f2:
        with redirect_stdout(f2):
            for name, _module in model.named_modules():
                print(name)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    model_ids = [
        # "Qwen/Qwen3-0.6B",
        # "Qwen/Qwen3-1.7B",
        # "Qwen/Qwen3-4B",
        # "Qwen/Qwen3-8B",
        # "Qwen/Qwen3-14B",
        # "Qwen/Qwen3-32B",
        # "Qwen/Qwen3-30B-A3B",
        # "Qwen/Qwen3-235B-A22B",
        "deepseek-ai/DeepSeek-V4-Pro"
    ]

    output_project_dir_by_model_id = {
        # "Qwen/Qwen3-0.6B": "qwen3-0.6b",
        # "Qwen/Qwen3-1.7B": "qwen3-1.7b",
        # "Qwen/Qwen3-4B": "qwen3-4b",
        # "Qwen/Qwen3-8B": "qwen3-8b",
        # "Qwen/Qwen3-14B": "qwen3-14b",
        # "Qwen/Qwen3-32B": "qwen3-32b",
        # "Qwen/Qwen3-30B-A3B": "qwen3-30b-a3b",
        # "Qwen/Qwen3-235B-A22B": "qwen3-235-a22b",
        "deepseek-ai/DeepSeek-V4-Pro": "deepseek-v4-pro"
    }

    for model_id in model_ids:
        project_name = output_project_dir_by_model_id.get(model_id, default_project_name(model_id))
        output_dir = repo_root / "config" / f"{project_name}-config"
        export_model_layers(model_id, output_dir)
