import torch
import torch.nn.functional as F
from train_toy import MemorizingModel

torch.manual_seed(42)
model = MemorizingModel(num_tokens=8, dim=128, depth=2, dim_value_head=64, causal=True, chunk_size=2)
for m in model.modules():
    if hasattr(m, 'muon_update'): m.muon_update = False

half = torch.randint(0, 8, (16, 16))
seq = torch.cat((half, half), dim=-1)

preds, _ = model(seq[:, :-1], return_next_memories=True, ablate_mem=False)
print("Preds max:", preds.max().item())
print("Preds min:", preds.min().item())
print("Preds has NaN:", torch.isnan(preds).any().item())
