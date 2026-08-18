from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch import cat, nn, randn, tensor
from torch.nn import Module, Linear, ParameterDict, Sequential

from einx import multiply
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

from adam_atan2_pytorch.muon_adam_atan2 import newtonschulz5
from adam_atan2_pytorch.polar_adam_atan2 import polar_express

# constants

LinearNoBias = partial(Linear, bias = False)

def AttentionMemory(*, wq, wk, wv, wo, wg = None):
    return remove_none_values(dict(wq = wq, wk = wk, wv = wv, wo = wo, wg = wg))

def add_memories(mem1, mem2):
    return {k: mem1[k] + mem2[k] for k in mem1.keys()}

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def remove_none_values(d):
    return {k: v for k, v in d.items() if exists(v)}

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
        max_learning_rate = 1e-2,
        max_muon_learning_rate = 1e-1,
        muon_update = True,
        use_polar_express = False,
        use_gates = True,
        max_fast_weight_norm = None,
        use_reverse_causal_target = False
    ):
        super().__init__()

        self.use_gates = use_gates
        self.use_reverse_causal_target = use_reverse_causal_target

        dim_value_head = default(dim_value_head, dim_head)

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

        self.memory_keys = self.attn_memory.keys()

        # to optimizer related

        self.to_learning_rate = Sequential(
            LinearNoBias(dim, 2 if muon_update else 1),
            nn.Sigmoid()
        )

        # muon related

        self.muon_update = muon_update
        self.use_polar_express = use_polar_express

        if muon_update:
            self.muon_update_fn = partial(polar_express if self.use_polar_express else newtonschulz5, bypass_update_fn = lambda ndim: False)

        lr_scales = tensor([max_learning_rate, max_muon_learning_rate]) if muon_update else tensor([max_learning_rate])
        self.register_buffer('lr_scales', lr_scales, persistent = False)

        # target values

        if self.use_reverse_causal_target:
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
        past_mem: AttentionMemory | None = None,
        detach_next_memories = False,
        boundary_state: tuple | None = None,
        return_boundary_state = False
    ):
        batch, scale, muon_update, use_gates, should_clip_weight_norm = tokens.shape[0], self.scale, self.muon_update, self.use_gates, self.should_clip_weight_norm

        # prenorm

        tokens = self.norm(tokens)

        # add the fast weight memories

        if exists(past_mem):
            memory = past_mem
        else:
            memory = self.init_memories(batch)

        # get the memories

        wq, wk, wv, wo = tuple(memory[name] for name in ('wq', 'wk', 'wv', 'wo'))

        gates = None

        if use_gates:
            wg = memory['wg']
            gates = einsum(tokens, wg, 'b n d, b h d dh -> b h n dh')
            gates = gates.sigmoid()

        # attention

        q = einsum(tokens, wq, 'b n d, b h d dh -> b h n dh')
        k = einsum(tokens, wk, 'b n d, b h d dh -> b h n dh')
        v = einsum(tokens, wv, 'b n d, b h d dh -> b h n dh')

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
            return pred_values

        target_values_full = self.to_target_values(tokens)
        target_values_full = self.target_values_norm(target_values_full)
        target_values = target_values_full[..., 1:, :]

        # extract boundary state for the next chunk before slicing

        next_boundary_state = (
            tokens[..., -1:, :],
            pred_values[..., -1:, :],
            out[..., -1:, :],
            gates[..., -1:, :] if use_gates else None,
            out_pre_gate[..., -1:, :] if use_gates else None,
            q[..., -1:, :],
            k[..., -1:, :],
            v[..., -1:, :]
        )

        # base slicing for backwards pass

        tokens = tokens[..., :-1, :]
        pred_values_for_fast_weight = pred_values[..., :-1, :]
        out = out[..., :-1, :]

        if exists(gates):
            gates = gates[..., :-1, :]
            out_pre_gate = out_pre_gate[..., :-1, :]

        attn_sliced = attn[..., :-1, :-1]
        q, k, v = q[..., :-1, :], k[..., :-1, :], v[..., :-1, :]

        if exists(boundary_state):
            b_tokens, b_pred_values, b_out, b_gates, b_out_pre_gate, b_q, b_k, b_v = boundary_state

            boundary_target = target_values_full[..., :1, :]
            target_values = cat((boundary_target, target_values), dim = -2)

            # cleanly concat boundary state to the rest of the tensors

            tokens, pred_values_for_fast_weight, out, q, k, v = tuple(
                cat((b_t, t), dim = -2) for b_t, t in zip(
                    (b_tokens, b_pred_values, b_out, b_q, b_k, b_v),
                    (tokens, pred_values_for_fast_weight, out, q, k, v)
                )
            )

            if exists(gates) and exists(b_gates):
                gates = cat((b_gates, gates), dim = -2)
                out_pre_gate = cat((b_out_pre_gate, out_pre_gate), dim = -2)

            # pad attention matrix dynamically for the boundary token

            attn = F.pad(attn_sliced, (1, 0, 1, 0), value = 0.)
            attn[..., 0, 0] = 1.
        else:
            attn = attn_sliced

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

        if not return_boundary_state:
            return pred_values, next_mems

        return pred_values, next_mems, next_boundary_state
