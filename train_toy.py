# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "torch",
#     "tqdm",
#     "x-mlps-pytorch",
#     "torch-einops-utils",
#     "adam-atan2-pytorch",
#     "einx",
#     "termcolor"
# ]
# ///

import fire
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from einops import rearrange
from termcolor import colored

from fast_weight_attention import FastWeightAttention
from x_mlps_pytorch import Feedforwards

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def print_header(char = '-', length = 40):
    print(char * length)

def per_position_acc(model, num_tokens, batch_size, half_len, total_len):
    model.eval()

    with torch.no_grad():
        correct = torch.zeros(total_len - 1, device = next(model.parameters()).device)
        counts = torch.zeros(total_len - 1, device = next(model.parameters()).device)

        for _ in range(200):
            half = torch.randint(0, num_tokens, (batch_size, half_len))
            seq = torch.cat((half, half), dim = -1)

            preds, _ = model(seq, return_next_memories = True)
            preds = preds[:, :-1]

            correct += (preds.argmax(dim = -1) == seq[:, 1:]).float().sum(dim = 0)
            counts += batch_size

        return correct / counts

# model

class MemorizingModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth = 1,
        dim_head = 32,
        dim_value_head = 32,
        heads = 4,
        causal = True,
        muon_update = True,
        use_polar_express = False,
        max_learning_rate = 1e-2,
        max_fast_weight_norm = None,
        chunk_size = 4,
        use_forget_gate = True,
        use_gates = True,
        use_reverse_causal_target = False,
        max_seq_len = 2048
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                FastWeightAttention(
                    dim = dim,
                    dim_head = dim_head,
                    dim_value_head = dim_value_head,
                    heads = heads,
                    causal = causal,
                    muon_update = muon_update,
                    use_polar_express = use_polar_express,
                    max_learning_rate = max_learning_rate,
                    use_gates = use_gates,
                    max_fast_weight_norm = max_fast_weight_norm,
                    use_reverse_causal_target = use_reverse_causal_target,
                    chunk_size = chunk_size,
                    use_forget_gate = use_forget_gate
                ),
                Feedforwards(dim, depth = 1)
            ]) for _ in range(depth)
        ])

        self.head = nn.Linear(dim, num_tokens)

    def forward(
        self,
        x,
        past_mems = None,
        return_next_memories = False,
        ablate_mem = False
    ):
        h = self.embed(x)

        pos = torch.arange(x.shape[-1], device = x.device)
        h = h + self.pos_embed(pos)

        past_mems = default(past_mems, [None] * len(self.layers))
        next_mems = []

        for (attn, ff), past_mem in zip(self.layers, past_mems):
            attn_out = attn(
                h,
                past_mem = past_mem,
                return_next_memories = return_next_memories,
                ablate_mem = ablate_mem
            )

            if return_next_memories:
                attn_out, next_mem = attn_out
                next_mems.append(next_mem)

            h = h + attn_out
            h = h + ff(h)

        logits = self.head(h)

        if not return_next_memories:
            return logits

        return logits, next_mems

# training

def train(
    seed = 42,
    num_tokens = 8,
    dim = 64,
    depth = 1,
    dim_head = 32,
    dim_value_head = 32,
    heads = 4,
    causal = True,
    batch_size = 16,
    num_batches = 500,
    lr = 3e-3,
    chunk_size = 4,
    half_len = 8,
    eval_batches = 50,
    eval_every = 100,
    muon_update = True,
    use_polar_express = True,
    max_learning_rate = 1e-3,
    max_fast_weight_norm = None,
    use_forget_gate = False,
    use_reverse_causal_target = False,
    single_run = False
):
    assert chunk_size <= half_len, 'chunk size must be less than or equal to half sequence length'

    total_len = half_len * 2

    print('')
    print(colored(f'Fast Weight Memory Toy Task', 'cyan', attrs=['bold']))
    print_header()
    print(f'The model must learn an auto-regressive sequence of length {total_len}')
    print(f'consisting of a random chunk of length {half_len} repeated twice.')
    print(f'Since it processes this in chunks of {chunk_size}, it must carry information')
    print(f'across chunks via its fast weight memories to predict the second half.')
    print_header()

    print(colored(f'Hyperparameters:', 'cyan'))
    print(f'  dim={dim}, heads={heads}, depth={depth}, forget_gate={use_forget_gate}')

    if muon_update:
        print(f'  Update Rule: Muon (polar_express={use_polar_express}) | max_lr={max_learning_rate}')
    else:
        print(f'  Update Rule: Plain | lr_base={lr} | max_fast_lr={max_learning_rate}')

    print_header()
    print('')

    results = dict()
    per_pos_results = dict()
    conditions = (True,) if single_run else (True, False)

    for use_gates in conditions:
        torch.manual_seed(seed)

        model = MemorizingModel(
            num_tokens,
            dim,
            depth = depth,
            dim_head = dim_head,
            dim_value_head = dim_value_head,
            heads = heads,
            causal = causal,
            muon_update = muon_update,
            use_polar_express = use_polar_express,
            max_learning_rate = max_learning_rate,
            chunk_size = chunk_size,
            use_forget_gate = use_forget_gate,
            use_gates = use_gates,
            use_reverse_causal_target = use_reverse_causal_target,
            max_fast_weight_norm = max_fast_weight_norm,
            max_seq_len = total_len
        )

        optim = Adam(model.parameters(), lr = lr)

        label = 'Gates' if use_gates else 'No_Gates'
        pbar = tqdm(range(num_batches), desc = label)

        last_accs = []

        for i in pbar:
            model.train()

            half = torch.randint(0, num_tokens, (batch_size, half_len))
            seq = torch.cat((half, half), dim = -1)

            labels = seq[:, 1:]

            preds, _ = model(seq, return_next_memories = True)
            preds = preds[:, :-1]

            loss = F.cross_entropy(
                rearrange(preds, 'b n d -> (b n) d'),
                rearrange(labels, 'b n -> (b n)')
            )

            loss.backward()

            if i % eval_every == 0:
                norms = {}

                for name, param in model.named_parameters():
                    if 'attn_memory' in name and exists(param.grad):
                        key = name.split('.')[-1]
                        norms.setdefault(key, []).append(param.grad.norm().item())

                if len(norms) > 0:
                    avg_norms = {k: sum(v) / len(v) for k, v in norms.items()}
                    norms_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_norms.items())
                    pbar.write(colored(f"  [Step {i:4d}] Grad Norms  |  {norms_str}", 'dark_grey'))

            optim.step()
            optim.zero_grad()

            loss_val = loss.item()

            if i % eval_every == 0 or i >= (num_batches - eval_batches):
                model.eval()

                with torch.no_grad():
                    all_preds, _ = model(seq, return_next_memories = True)
                    all_preds = all_preds[:, :-1]

                    preds_class = all_preds.argmax(dim = -1)
                    acc = (preds_class[:, half_len:] == labels[:, half_len:]).float().mean()

                    if i >= (num_batches - eval_batches):
                        last_accs.append(acc.item())

                pbar.set_postfix(loss = f'{loss_val:.3f}', acc = f'{acc.item():.3f}')

        # per-position accuracy on the second half, to verify the first token of the second half is handled

        per_pos = per_position_acc(model, num_tokens, batch_size, half_len, total_len)

        results[label] = sum(last_accs) / len(last_accs)
        per_pos_results[label] = per_pos[half_len].item()

        pbar.write(colored(f'  Second half mean: {per_pos[half_len:].mean():.1%} | 1st token of 2nd half: {per_pos[half_len]:.1%}', 'dark_grey'))

    # report

    print('')
    print_header()
    for label, acc in results.items():
        print(f'  {label}: {acc:.1%}')
        print(f'     1st token of 2nd half: {per_pos_results[label]:.1%}')
    print_header()

    if not single_run and 'No_Gates' in results:
        advantage = results['Gates'] - results['No_Gates']

        if advantage > 0.10:
            print(colored(f'\n  Gates advantage confirmed.', 'green', attrs=['bold']))
        elif advantage < -0.10:
            print(colored(f'\n  No_Gates was actually better.', 'red', attrs=['bold']))
        else:
            print(colored(f'\n  No significant advantage either way.', 'yellow', attrs=['bold']))

if __name__ == '__main__':
    fire.Fire(train)
