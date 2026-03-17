import os
from contextlib import redirect_stdout
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

if __name__ == "__main__":
    model_id = "openai/gpt-oss-20b"
    print(f"[INFO] 正在通过 Meta Device 加载模型结构骨架...")
    
    # 1. 仅下载/加载配置，不加载权重（非常快）
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # 2. 使用 meta 设备初始化模型骨架（0 显存/内存占用）
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # 确保输出目录存在
    os.makedirs("./config", exist_ok=True)

    # 3. 完美输出原生结构
    with open("./config/layerstructure.txt", "w") as f1:
        with redirect_stdout(f1):
            print_model_layers(model)

    with open("./config/layernames.txt", "w") as f2:
        with redirect_stdout(f2):
            for name, module in model.named_modules():
                print(name)
                
    print("[INFO] 模型层级结构已完美保存至 ./config/（全程 0 OOM 风险）")