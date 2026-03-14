import torch
import torch.nn.functional as F
from torch.optim import Adam
from train_toy import MemorizingModel
from einops import rearrange

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)

model = MemorizingModel(num_tokens=8, dim=128, depth=2, dim_value_head=64, causal=True, chunk_size=2)
# Ensure we are using muon_update = False to test if the gate prevents the NaN
for m in model.modules():
    if hasattr(m, 'muon_update'):
        m.muon_update = False

optim = Adam(model.parameters(), lr=1e-4)

try:
    for i in range(100):
        model.train()
        half = torch.randint(0, 8, (16, 16))
        seq = torch.cat((half, half), dim=-1)
        x, labels = seq[:, :-1], seq[:, 1:]
        
        preds, _ = model(x, return_next_memories=True, ablate_mem=False)
        
        loss = F.cross_entropy(
            rearrange(preds, 'b n d -> (b n) d'),
            rearrange(labels, 'b n -> (b n)')
        )
        
        if torch.isnan(loss):
            print(f"NaN loss detected at step {i}")
            break
            
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        if i % 10 == 0:
            print(f'Step {i}, loss: {loss.item():.4f}')
            
except Exception as e:
    print(f"Exception during training: {e}")

