import torch
from train_toy import MemorizingModel
from einops import rearrange

torch.manual_seed(42)
model = MemorizingModel(num_tokens=8, dim=128, depth=2, dim_value_head=64, causal=True, chunk_size=2)
for m in model.modules():
    if hasattr(m, 'muon_update'): m.muon_update = False

# Force forget gate to be EXACTLY 0
for m in model.modules():
    if hasattr(m, 'to_forget_gate'):
        torch.nn.init.constant_(m.to_forget_gate[-1].bias, -20.0)

half = torch.randint(0, 8, (16, 16))
seq = torch.cat((half, half), dim=-1)

preds, _ = model(seq[:, :-1], return_next_memories=True, ablate_mem=False)
print("Preds max (gate=-20):", preds.max().item())
print("Preds min (gate=-20):", preds.min().item())
print("Preds has NaN:", torch.isnan(preds).any().item())

# Force forget gate to be what it was (-3.0)
for m in model.modules():
    if hasattr(m, 'to_forget_gate'):
        torch.nn.init.constant_(m.to_forget_gate[-1].bias, -3.0)

preds, _ = model(seq[:, :-1], return_next_memories=True, ablate_mem=False)
print("\nPreds max (gate=-3):", preds.max().item())
print("Preds min (gate=-3):", preds.min().item())
print("Preds has NaN:", torch.isnan(preds).any().item())
