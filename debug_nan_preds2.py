import torch
from train_toy import MemorizingModel

torch.manual_seed(42)
model = MemorizingModel(num_tokens=8, dim=128, depth=2, dim_value_head=64, causal=True, chunk_size=2)
for m in model.modules():
    if hasattr(m, 'muon_update'): m.muon_update = False

def check_max(name, val):
    if isinstance(val, tuple):
        for i, v in enumerate(val): check_max(f'{name}[{i}]', v)
    elif isinstance(val, dict):
        for k, v in val.items(): check_max(f'{name}[{k}]', v)
    elif hasattr(val, 'max'):
        print(f'{name:40s} max: {val.max().item():.2e} min: {val.min().item():.2e}')

for name, m in model.named_modules():
    m.register_forward_hook(lambda m, inp, out, name=name: check_max(name, out))

half = torch.randint(0, 8, (16, 16))
seq = torch.cat((half, half), dim=-1)

preds, _ = model(seq[:, :-1], return_next_memories=True, ablate_mem=False)
