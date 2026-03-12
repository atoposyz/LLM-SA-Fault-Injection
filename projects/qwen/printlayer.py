from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import redirect_stdout


def print_model_layers(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent + f"{name} ({module.__class__.__name__})")
        if list(module.named_children()):  # 如果还有子层
            print_model_layers(module, indent + 4)
        else:
            if hasattr(module, "weight"):
                print(" " * (indent + 4) + f"weight: {module.weight.shape}")


if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", device_map="auto", trust_remote_code=True)
    model.eval()
    print(next(model.parameters()).dtype)
    print(model.config.num_hidden_layers)

    # 保存 print_model_layers 的输出
    with open("/workplace/home/mayongzhe/faultinject/projects/qwen/layerstructure.txt", "w") as f1:
        with redirect_stdout(f1):
            print_model_layers(model)

    # 保存 named_modules 输出的模块名
    with open("/workplace/home/mayongzhe/faultinject/projects/qwen/layernames.txt", "w") as f2:
        with redirect_stdout(f2):
            for name, module in model.named_modules():
                print(name)
