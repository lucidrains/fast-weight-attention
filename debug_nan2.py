import torch
import torch.nn.functional as F
from torch.optim import Adam
from train_toy import MemorizingModel
from einops import rearrange

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)

model = MemorizingModel(num_tokens=8, dim=128, depth=2, dim_value_head=64, causal=True, chunk_size=2)
for m in model.modules():
    if hasattr(m, 'muon_update'):
        m.muon_update = False

half = torch.randint(0, 8, (16, 16))
seq = torch.cat((half, half), dim=-1)
x = seq[:, :-1]

def check_nan(name, val):
    if isinstance(val, dict):
        for k, v in val.items():
            if torch.isnan(v).any():
                print(f"NaN in {name}[{k}]")
    elif isinstance(val, torch.Tensor):
        if torch.isnan(val).any():
            print(f"NaN in {name}")

for m in model.modules():
    m.register_forward_hook(lambda m, inp, out: check_nan(m.__class__.__name__, out))

preds, _ = model(x, return_next_memories=True, ablate_mem=False)
