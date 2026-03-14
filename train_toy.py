# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "adam-atan2-pytorch",
#     "torch-einops-utils",
#     "torch",
#     "tqdm",
#     "x-mlps-pytorch",
#     "accelerate"
# ]
# ///

import fire
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from einops import rearrange

from fast_weight_attention import ChunkedFastWeightAttention
from x_mlps_pytorch import Feedforwards

from accelerate import Accelerator

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# model

class MemorizingModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth = 1,
        dim_value_head = 32,
        causal = True,
        chunk_size = None
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                ChunkedFastWeightAttention(
                    dim = dim,
                    dim_head = 32,
                    dim_value_head = dim_value_head,
                    heads = 4,
                    causal = causal,
                    chunk_size = chunk_size,
                    muon_update = muon_update,
                    max_learning_rate = 1e-2,
                    max_grad_norm = 1.0
                ),
                Feedforwards(dim, depth = 1)
            ]) for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, past_mems = None, return_next_memories = False, ablate_mem = False):
        h = self.embed(x)

        past_mems = default(past_mems, [None] * len(self.layers))
        next_mems = []

        for (attn, ff), past_mem in zip(self.layers, past_mems):
            if return_next_memories:
                attn_out, next_mem = attn(h, past_mem = past_mem, return_next_memories = True, ablate_mem = ablate_mem)
                next_mems.append(next_mem)
            else:
                attn_out = attn(h, past_mem = past_mem, ablate_mem = ablate_mem)

            h = h + attn_out
            h = h + ff(h)

        if return_next_memories:
            return self.head(h), next_mems

        return self.head(h)

# training

def train(
    seed = 42,
    num_tokens = 8,
    dim = 32,
    depth = 2,
    dim_value_head = 32,
    causal = True,
    half_len = 16,
    batch_size = 16,
    num_batches = 1000,
    lr = 1e-1,
    chunk_size = 2,
    eval_batches = 20,
    eval_every = 50,
    memory_only = False,
    muon_update = True
):
    assert chunk_size <= half_len, 'chunk size must be less than or equal to half sequence length'

    results = dict()

    models_to_run = (True,) if memory_only else (False, True)

    accelerator = Accelerator()

    for use_memory in models_to_run:
        torch.manual_seed(seed)

        model = MemorizingModel(
            num_tokens, dim, depth = depth, dim_value_head = dim_value_head,
            causal = causal, chunk_size = chunk_size
        )
        optim = Adam(model.parameters(), lr = lr)

        model, optim = accelerator.prepare(model, optim)

        label = 'Memory' if use_memory else 'Baseline'
        pbar = tqdm(range(num_batches), desc = label)
        last_accs = []

        for i in pbar:
            model.train()

            half = torch.randint(0, num_tokens, (batch_size, half_len), device = accelerator.device)
            seq = torch.cat((half, half), dim = -1)

            x, labels = seq[:, :-1], seq[:, 1:]

            preds, _ = model(x, return_next_memories=True, ablate_mem=not use_memory)

            loss = F.cross_entropy(
                rearrange(preds, 'b n d -> (b n) d'),
                rearrange(labels, 'b n -> (b n)')
            )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

            loss_val = loss.item()

            if i % eval_every == 0 or i >= (num_batches - eval_batches):
                model.eval()
                with torch.no_grad():
                    all_preds, _ = model(x, return_next_memories=True, ablate_mem=not use_memory)

                    preds_class = all_preds.argmax(dim = -1)
                    acc = (preds_class[:, half_len:] == labels[:, half_len:]).float().mean()

                    if i >= (num_batches - eval_batches):
                        last_accs.append(acc.item())

                pbar.set_postfix(loss = f'{loss_val:.3f}', acc = f'{acc.item():.3f}')

        results[label] = sum(last_accs) / len(last_accs)

    # report

    print(f'\n{"-" * 40}')
    for label, acc in results.items():
        print(f'  {label}: {acc:.1%}')
    print(f'{"-" * 40}')

    if not memory_only and 'Baseline' in results:
        memory_acc = results['Memory']
        baseline_acc = results['Baseline']
        advantage = memory_acc - baseline_acc
        print(f'\n  {"✅ Memory works!" if advantage > 0.20 else "❌ No clear advantage."}')

if __name__ == '__main__':
    fire.Fire(train)
