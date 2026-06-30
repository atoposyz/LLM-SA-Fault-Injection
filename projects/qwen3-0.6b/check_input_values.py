"""Print OS input reg fault diff values at top-left 32x32 crop."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tool", "src"))
from tool.fault_injector_next_nolimit import Fast_SA_FaultInjector
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(42)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto", trust_remote_code=True)
model.eval()
v_proj = model.model.layers[0].self_attn.v_proj

inj = Fast_SA_FaultInjector(sa_rows=8, sa_cols=8, dataflow="OS", fault_type="weight_bitflip_29", precision="fp32")
inj.fault_config["mode"] = "input"
inj.fault_config["op"] = "flip"
inj.set_fault_position(2, 3)
inj.fault_reg[0] = 1
inj.fault_bit[0] = 29

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
p = "Hello world test. " * 80
inp = tok(p, return_tensors="pt").to(model.device)
print(f"tokens: {inp.input_ids.shape[1]}")

class H:
    def __init__(s): s.o = None
    def __call__(s, m, i, o): s.o = o[0].detach().cpu() if isinstance(o, tuple) else o.detach().cpu()

hv = H(); h0 = v_proj.register_forward_hook(hv)
with torch.no_grad(): model(**inp)
base = hv.o.clone(); h0.remove()

inj.reset_fault_pe()
inj.fault_pe_row.append(2); inj.fault_pe_col.append(3)
inj.fault_reg.append(1); inj.fault_bit.append(29)
hv2 = H()
hi = v_proj.register_forward_hook(inj.hook_fn); h0 = v_proj.register_forward_hook(hv2)
with torch.no_grad(): model(**inp)
faulty = hv2.o.clone(); hi.remove(); h0.remove()

diff = (base - faulty).abs()[0]
print(f"overall max={diff.max():.1f} mean={diff.mean():.1f}")

crop = diff[:32, :32]
print("\ntop-left 32x32 (nonzero rows):")
for r in range(32):
    mx = crop[r].max().item()
    if mx > 0.01:
        aff = [str(j) for j in range(32) if crop[r, j] > 0.01]
        vals = [f"{crop[r,j].item():.0f}" for j in range(32) if crop[r,j] > 0.01]
        print(f"  row{r:2d}(%8={r%8}) max={mx:.0f}  cols:{','.join(aff)}  vals:{','.join(vals)}")
