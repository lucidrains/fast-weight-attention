from __future__ import annotations
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import cat, nn, randn, roll, tensor
from torch.nn import Module, Linear, ParameterDict, Sequential

from einx import multiply
from einops import einsum, repeat, rearrange, reduce

from adam_atan2_pytorch.muon_adam_atan2 import newtonschulz5
from adam_atan2_pytorch.polar_adam_atan2 import polar_express

# constants

LinearNoBias = partial(Linear, bias = False)

def AttentionMemory(*, wq, wk, wv, wo, wg = None):
    return remove_none_values(dict(wq = wq, wk = wk, wv = wv, wo = wo, wg = wg))

def add_memories(mem1, mem2):
    return {k: mem1[k] + mem2[k] for k in mem1.keys()}

# state

class FastWeightState(NamedTuple):
    memory: dict
    token_count: int = 0

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def remove_none_values(d):
    return {k: v for k, v in d.items() if exists(v)}

def divisible_by(num, den):
    return (num % den) == 0

def is_greater_than_zero(n):
    return n > 0

# differentiable clip weight norm

def softclamp(t, value):
    return (t / value).tanh() * value

def soft_clip_max_norm(weights, max_norm, dim, eps = 1e-5):
    assert max_norm > 1.
    shift, scale = (max_norm + 1.) * 0.5, (max_norm - 1.) * 0.5

    norm = weights.norm(dim = dim, p = 2, keepdim = True)
    softclamped_norm = softclamp(norm - shift, scale) + shift

    return weights / softclamped_norm.clamp_min(eps)

# classes

class ReverseCausalAttention(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qkv = LinearNoBias(dim, dim * 3)

    def forward(self, x):
        n, device = x.shape[-2], x.device

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        score = einsum(q, k, 'b n d, b m d -> b n m') * self.scale

        # reverse causal - each token attends to itself and future tokens

        mask = torch.ones((n, n), device = device, dtype = torch.bool).tril(-1)
        score = score.masked_fill(mask, -torch.finfo(score.dtype).max)

        attn = score.softmax(dim = -1)

        return einsum(attn, v, 'b n m, b m d -> b n d')

# main class

class FastWeightAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_value_head = None,
        heads = 8,
        causal = True,
        chunk_size = None,
        use_forget_gate = False,
        max_learning_rate = 1e-2,
        max_muon_learning_rate = 1e-1,
        muon_update = True,
        use_polar_express = False,
        use_gates = True,
        max_fast_weight_norm = None,
        use_reverse_causal_target = False,
        use_boundary_embed = True
    ):
        super().__init__()

        self.use_gates = use_gates
        self.chunk_size = chunk_size

        assert not exists(chunk_size) or chunk_size >= 2, 'chunk size must be at least 2'

        dim_value_head = default(dim_value_head, dim_head)

        # boundary embedding - added to the hidden state of the last token of each completed chunk, and its query
        # replaced by the embedding itself, so all chunk boundaries route to the same memory region. the store
        # target for the boundary token is the chunk's first token, so the boundary slot comes to hold the previous
        # chunk's first token - resolving the boundary prediction without any lookahead

        self.boundary_embed = nn.Parameter(torch.zeros(dim)) if use_boundary_embed else None

        self.norm = nn.RMSNorm(dim)

        # scale

        self.scale = dim_head ** -0.5

        self.causal = causal

        # memory parameters

        shapes = dict(
            wq = (heads, dim, dim_head),
            wk = (heads, dim, dim_head),
            wv = (heads, dim, dim_value_head),
            wo = (heads, dim_value_head, dim)
        )

        if self.use_gates:
            shapes.update(wg = (heads, dim, dim_value_head))

        self.attn_memory = ParameterDict({
            name: randn(shape) * (shape[-1] ** -0.5)
            for name, shape in shapes.items()
        })

        # forget gate - chunk level, applied to the fast weight memories of the previous chunk
        # gdn update rule - mean of the stored input embeddings, then projected (not projected then averaged)

        self.use_forget_gate = use_forget_gate

        if use_forget_gate:
            self.to_forget_gates = nn.Linear(dim, heads, bias = True)
            self.forget_gate_decay = nn.Parameter(tensor(0.))

        # to optimizer related

        self.to_learning_rate = Sequential(
            LinearNoBias(dim, 2 if muon_update else 1),
            nn.Sigmoid()
        )

        # muon related

        self.muon_update = muon_update

        if muon_update:
            self.muon_update_fn = partial(polar_express if use_polar_express else newtonschulz5, bypass_update_fn = lambda ndim: False)

        lr_scales = tensor([max_learning_rate, max_muon_learning_rate]) if muon_update else tensor([max_learning_rate])
        self.register_buffer('lr_scales', lr_scales, persistent = False)

        # target values

        if use_reverse_causal_target:
            self.to_target_values = ReverseCausalAttention(dim)
        else:
            self.to_target_values = LinearNoBias(dim, dim)

        # using the z-score as well as done for fast-weight PKM proposed by Sakana AI

        self.target_values_norm = nn.LayerNorm(dim, elementwise_affine = False)

        # whether to clip the fast weight norms
        # Volchkov et al. from Clip to Grok

        self.max_fast_weight_norm = max_fast_weight_norm
        self.should_clip_weight_norm = exists(max_fast_weight_norm)

        self.weight_name_to_row_dim = dict(wq = -2, wk = -2, wv = -2, wo = -1)

        if self.use_gates:
            self.weight_name_to_row_dim.update(wg = -2)

    def init_memories(self, batch):
        return {name: repeat(weights, '... -> b ...', b = batch) for name, weights in self.attn_memory.items()}

    def forward(
        self,
        tokens,
        return_next_memories = False,
        return_grads_only = False,
        past_mem: FastWeightState | None = None,
        detach_next_memories_every: int | None = None,
        ablate_mem: bool = False
    ):
        batch = tokens.shape[0]
        seq_len = tokens.shape[-2]
        chunk_size = max(default(self.chunk_size, seq_len), 2)

        past_mem = default(past_mem, FastWeightState(self.init_memories(batch)))

        if seq_len == 0:
            return (tokens, past_mem) if return_next_memories else tokens

        # prenorm

        tokens = self.norm(tokens)

        # calc segments reaching chunk boundaries

        count, memory = past_mem.token_count, past_mem.memory

        to_bound = chunk_size - (count % chunk_size)
        remainder = max(0, seq_len - to_bound)
        num_chunks, chunk_remainder = divmod(remainder, chunk_size)

        split_sizes = (min(seq_len, to_bound), *([chunk_size] * num_chunks), chunk_remainder)
        segments = tokens.split(tuple(filter(is_greater_than_zero, split_sizes)), dim = -2)

        out_list = []

        for chunk_index, segment in enumerate(segments):
            # periodic truncated bptt - detach memories every N chunks

            should_detach = exists(detach_next_memories_every) and divisible_by(chunk_index + 1, detach_next_memories_every)

            segment_len = segment.shape[-2]
            ends_boundary = divisible_by(count + segment_len, chunk_size)

            past_memory = None
            if exists(memory) and not ablate_mem:
                if self.use_forget_gate:
                    chunk_embedding = reduce(segment, 'b n d -> b d', 'mean')
                    forget_logits = self.to_forget_gates(chunk_embedding)
                    forget_gates = (-self.forget_gate_decay * F.softplus(forget_logits)).exp()
                    past_memory = {k: multiply('b h, b h ... -> b h ...', forget_gates, v) for k, v in memory.items()}
                else:
                    past_memory = memory

            out, next_memory = self._forward_chunk(
                segment,
                past_mem = past_memory,
                mark_boundary = ends_boundary,
                return_next_memories = return_next_memories,
                return_grads_only = return_grads_only,
                detach_next_memories = should_detach
            )

            memory = next_memory
            count += segment_len

            out_list.append(out)

        res = cat(out_list, dim = -2)

        if not return_next_memories:
            return res

        return res, FastWeightState(memory = memory, token_count = count)

    def _forward_chunk(
        self,
        tokens,
        past_mem: dict | None = None,
        mark_boundary = False,
        return_next_memories = False,
        return_grads_only = False,
        detach_next_memories = False
    ):
        batch, scale, muon_update, use_gates, should_clip_weight_norm = tokens.shape[0], self.scale, self.muon_update, self.use_gates, self.should_clip_weight_norm

        # mark the last token of each completed chunk with the boundary embedding - added to its hidden state so
        # the model knows it is special, and its query replaced by the embedding itself

        if mark_boundary and exists(self.boundary_embed):
            tokens = cat((tokens[..., :-1, :], tokens[..., -1:, :] + self.boundary_embed), dim = -2)

        # add the fast weight memories

        memory = default(past_mem, self.init_memories(batch))

        # get the memories

        wq, wk, wv, wo = (memory[name] for name in ('wq', 'wk', 'wv', 'wo'))

        gates = None

        if use_gates:
            wg = memory['wg']
            gates = einsum(tokens, wg, 'b n d, b h d dh -> b h n dh')
            gates = gates.sigmoid()

        # attention

        q = einsum(tokens, wq, 'b n d, b h d dh -> b h n dh')
        k = einsum(tokens, wk, 'b n d, b h d dh -> b h n dh')
        v = einsum(tokens, wv, 'b n d, b h d dh -> b h n dh')

        if mark_boundary and exists(self.boundary_embed):
            # query for the boundary token is the boundary embedding itself, so all chunk boundaries
            # route to the same memory region

            boundary_embed = repeat(self.boundary_embed, 'd -> b 1 d', b = batch)
            boundary_q = einsum(boundary_embed, wq, 'b n d, b h d dh -> b h n dh')

            q = cat((q[..., :-1, :], boundary_q), dim = -2)

        score = einsum(q, k, 'b h i dh, b h j dh -> b h i j') * scale

        if self.causal:
            i, j = score.shape[-2:]
            causal_mask = torch.ones((i, j), device = score.device, dtype = torch.bool).triu(j - i + 1)
            score = score.masked_fill(causal_mask, -torch.finfo(score.dtype).max)

        attn = score.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j dh -> b h i dh')

        if use_gates:
            out_pre_gate = out
            out = out * gates

        pred_values = einsum(out, wo, 'b h n dh, b h dh d -> b n d')

        if not return_next_memories:
            return pred_values, memory

        target_values_full = self.to_target_values(tokens)
        target_values_full = self.target_values_norm(target_values_full)

        if mark_boundary:
            # store target for each token is the value of the following token, wrapping around at the chunk end - so
            # the boundary slot comes to hold the value of the chunk's first token, no lookahead needed

            target_values = roll(target_values_full, -1, dims = -2)
            pred_values_for_fast_weight = pred_values
        else:
            target_values = target_values_full[..., 1:, :]

            # base slicing for backwards pass - the last token of an incomplete chunk has no target yet

            tokens = tokens[..., :-1, :]
            pred_values_for_fast_weight = pred_values[..., :-1, :]
            out = out[..., :-1, :]

            if exists(gates):
                gates = gates[..., :-1, :]
                out_pre_gate = out_pre_gate[..., :-1, :]

            attn = attn[..., :-1, :-1]
            q, k, v = q[..., :-1, :], k[..., :-1, :], v[..., :-1, :]

        if tokens.shape[-2] == 0:
            # nothing to update - lone token of an incomplete chunk has no target yet

            return pred_values, memory

        # per token learning rate related

        learning_rates = self.to_learning_rate(tokens) * self.lr_scales

        if muon_update:
            learning_rate, muon_learning_rate = learning_rates.unbind(dim = -1)
        else:
            learning_rate = rearrange(learning_rates, '... 1 -> ...')

        # mse error
        # flipped sign so no need to -grad at end

        error = target_values - pred_values_for_fast_weight

        if not muon_update:
            error = error * rearrange(learning_rate, '... -> ... 1')

        # now go through the backwards pass of attention, using predicted loss to next target value (Sakana AI discovery)

        dout = einsum(error, wo, 'b n d, b h dh d -> b h n dh')

        du = dout * gates if exists(gates) else dout

        delta = reduce(dout * out, '... d -> ... 1', 'sum')

        if exists(gates):
            dgates_pre = dout * out_pre_gate * gates * (1. - gates)

        dv = einsum(attn, du, 'b h i j, b h i dh -> b h j dh')

        dattn = einsum(v, du, 'b h j dh, b h i dh -> b h i j')

        dscore = scale * attn * (dattn - delta)

        dq = einsum(k, dscore, 'b h j dh, b h i j -> b h i dh')
        dk = einsum(q, dscore, 'b h i dh, b h i j -> b h j dh')

        # apply learning rates

        if muon_update:
            tokens_for_dwqk = multiply('b n d, b n', tokens, learning_rate)
            tokens_for_dwv = multiply('b n d, b n', tokens, muon_learning_rate)
            tokens_for_dwg = multiply('b n d, b n', tokens, muon_learning_rate)
            out_for_dwo = multiply('b h n d, b n', out, muon_learning_rate)
        else:
            tokens_for_dwqk = tokens
            tokens_for_dwv = tokens
            tokens_for_dwg = tokens
            out_for_dwo = out

        # get the next memories

        dwq = einsum(dq, tokens_for_dwqk, 'b h i dh, b i d -> b h d dh')
        dwk = einsum(dk, tokens_for_dwqk, 'b h j dh , b j d -> b h d dh')
        dwv = einsum(dv, tokens_for_dwv, 'b h j dh , b j d -> b h d dh')
        dwo = einsum(error, out_for_dwo, 'b n d, b h n dh -> b h dh d')

        dwg = None

        if use_gates:
            dwg = einsum(dgates_pre, tokens_for_dwg, 'b h i dh, b i d -> b h d dh')

        if muon_update:
            dwv = self.muon_update_fn(dwv)
            dwo = self.muon_update_fn(dwo)

            if use_gates:
                dwg = self.muon_update_fn(dwg)

        # prep next memories

        next_mems = AttentionMemory(wq = dwq, wk = dwk, wv = dwv, wo = dwo, wg = dwg)

        if not return_grads_only:
            next_mems = add_memories(memory, next_mems)

        # maybe clip weight norms

        if should_clip_weight_norm:
            for weight_name, row_dim in self.weight_name_to_row_dim.items():
                weight = next_mems[weight_name]
                next_mems[weight_name] = soft_clip_max_norm(weight, self.max_fast_weight_norm, dim = row_dim)

        # maybe detach

        if detach_next_memories:
            next_mems = {k: v.detach() for k, v in next_mems.items()}

        return pred_values, next_mems
