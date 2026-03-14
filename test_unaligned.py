import torch
from fast_weight_attention import FastWeightAttention

model = FastWeightAttention(dim=32, heads=4, dim_head=16, causal=True, chunk_size=16, muon_update=False)
tokens = torch.randn(1, 32, 32)
parallel_out, parallel_state = model(tokens, return_next_memories=True)

pos = 0
curr_state = None
serial_out = []
for split in [7, 10, 5, 10]:
    out, curr_state = model(tokens[:, pos:pos+split], return_next_memories=True, past_mem=curr_state)
    serial_out.append(out)
    pos += split
sequential_out = torch.cat(serial_out, dim=1)

diff = (parallel_out - sequential_out).abs().max()
print("Diff:", diff.item())
