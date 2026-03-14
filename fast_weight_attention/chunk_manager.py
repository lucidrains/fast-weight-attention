import torch
from torch import cat, nn
from torch.nn import Module
from typing import NamedTuple, Any

from torch_einops_utils import safe_cat, tree_map_tensor

from einx import multiply
from einops.layers.torch import Reduce

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# state

class ChunkingState(NamedTuple):
    memory: Any = None
    last_token: torch.Tensor | None = None
    token_count: int = 0

# main class

class ChunkManager(Module):
    def __init__(self, net, chunk_size = None):
        super().__init__()
        self.net = net
        self.chunk_size = chunk_size

        heads, dim, _ = net.attn_memory['wq'].shape

        self.to_forget_gate = nn.Sequential(
            nn.Linear(dim, dim),
            Reduce('b n d -> b d', 'mean'),
            nn.SiLU(),
            nn.Linear(dim, heads)
        )

        nn.init.constant_(self.to_forget_gate[-1].bias, 5.)

    def forward(
        self,
        tokens,
        return_next_memories = False,
        past_mem: ChunkingState | None = None,
        detach_next_memories_every: int | None = None,
        ablate_mem: bool = False,
        **kwargs
    ):
        past_mem = default(past_mem, ChunkingState())

        seq_len = tokens.shape[-2]
        chunk_size = default(self.chunk_size, seq_len)

        count = past_mem.token_count

        to_bound = chunk_size - (count % chunk_size)
        remainder = max(0, seq_len - to_bound)
        num_chunks, chunk_remainder = divmod(remainder, chunk_size)

        split_sizes = (min(seq_len, to_bound), *([chunk_size] * num_chunks), chunk_remainder)
        split_sizes = tuple(filter(lambda n: n > 0, split_sizes))
        segments = tokens.split(split_sizes, dim = -2)

        out_list = []

        for chunk_index, segment in enumerate(segments):
            should_detach = exists(detach_next_memories_every) and divisible_by(chunk_index + 1, detach_next_memories_every)

            segment_len = segment.shape[-2]

            chunk_input = safe_cat((past_mem.last_token, segment), dim = -2)

            past_memory = None
            if exists(past_mem.memory) and not ablate_mem:
                gate = self.to_forget_gate(segment).sigmoid()
                past_memory = {k: multiply('b h, b h ... -> b h ...', gate, v) for k, v in past_mem.memory.items()}

            out, next_mem = self.net(
                chunk_input,
                return_next_memories = True,
                past_mem = past_memory,
                **kwargs
            )

            if exists(past_mem.last_token):
                out = out[:, 1:]

            if should_detach and exists(next_mem):
                next_mem = tree_map_tensor(lambda t: t.detach(), next_mem)

            past_mem = ChunkingState(
                memory = next_mem,
                last_token = segment[:, -1:],
                token_count = past_mem.token_count + segment_len
            )

            out_list.append(out)

        res = cat(out_list, dim = -2)

        if not return_next_memories:
            return res

        return res, past_mem
