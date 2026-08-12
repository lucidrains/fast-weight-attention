from __future__ import annotations
from typing import NamedTuple, Any

import torch
from torch import cat, nn, Tensor
from torch.nn import Module

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
    last_token: Tensor | None = None
    token_count: int = 0
    boundary_state: Any = None
    buffer: Tensor | None = None

# main class

class ChunkManager(Module):
    def __init__(
        self,
        net: Module,
        chunk_size = None,
        use_forget_gate = False
    ):
        super().__init__()
        self.net = net
        self.chunk_size = chunk_size
        self.use_forget_gate = use_forget_gate

        assert not exists(chunk_size) or chunk_size >= 2, 'chunk size must be at least 2'

        if use_forget_gate:
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

        if exists(past_mem.buffer):
            tokens = cat((past_mem.buffer, tokens), dim=-2)

        seq_len = tokens.shape[-2]
        chunk_size = default(self.chunk_size, seq_len)

        num_chunks, chunk_remainder = divmod(seq_len, chunk_size)

        split_sizes = (*([chunk_size] * num_chunks), chunk_remainder)
        split_sizes = tuple(filter(lambda n: n > 0, split_sizes))
        segments = tokens.split(split_sizes, dim = -2)

        out_list = []

        for chunk_index, segment in enumerate(segments):
            should_detach = exists(detach_next_memories_every) and divisible_by(chunk_index + 1, detach_next_memories_every)

            segment_len = segment.shape[-2]

            if segment_len < chunk_size:
                past_mem = ChunkingState(
                    memory = past_mem.memory,
                    last_token = past_mem.last_token,
                    token_count = past_mem.token_count,
                    boundary_state = past_mem.boundary_state,
                    buffer = segment
                )
                continue

            past_memory = None
            if exists(past_mem.memory) and not ablate_mem:
                if self.use_forget_gate:
                    gate = self.to_forget_gate(segment).sigmoid()
                    past_memory = {k: multiply('b h, b h ... -> b h ...', gate, v) for k, v in past_mem.memory.items()}
                else:
                    past_memory = past_mem.memory

            out, next_mem, next_boundary_state = self.net(
                segment,
                return_next_memories = True,
                past_mem = past_memory,
                boundary_state = past_mem.boundary_state,
                return_boundary_state = True,
                **kwargs
            )

            if should_detach and exists(next_mem):
                next_mem = tree_map_tensor(lambda t: t.detach(), next_mem)
                next_boundary_state = tree_map_tensor(lambda t: t.detach(), next_boundary_state)

            past_mem = ChunkingState(
                memory = next_mem,
                last_token = None,
                token_count = past_mem.token_count + segment_len,
                boundary_state = next_boundary_state,
                buffer = None
            )

            out_list.append(out)

        if len(out_list) == 0:
            return (None, past_mem) if return_next_memories else None

        res = cat(out_list, dim = -2)

        if not return_next_memories:
            return res

        return res, past_mem
