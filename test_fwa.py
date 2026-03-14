import torch
from fast_weight_attention.fast_weight_attention import FastWeightAttentionCore

core = FastWeightAttentionCore(dim=32, dim_head=16, heads=4, muon_update=False)
tokens = torch.randn(1, 16, 32)
out_full, mem_full = core(tokens, return_next_memories=True)

out1, mem1 = core(tokens[:, :1], return_next_memories=True)
chunk2 = torch.cat((tokens[:, 0:1], tokens[:, 1:2]), dim=1) # causal prep
out2, mem2 = core(chunk2, past_mem=mem1, return_next_memories=True)

print("Full:", mem_full['wq'][0,0,0])
print("1st:", mem1['wq'][0,0,0])
print("2nd:", mem2['wq'][0,0,0])
