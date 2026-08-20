import torch
import random

import pytest
param = pytest.mark.parametrize

from fast_weight_attention import FastWeightAttention

# tests

@param('causal', (False, True))
@param('use_gates', (False, True))
@param('max_fast_weight_norm', (None, 2.))
@param('muon_update,use_polar_express', [
    (False, False),
    (True, False),
    (True, True)
])
def test_mem(
    causal,
    use_gates,
    muon_update,
    max_fast_weight_norm,
    use_polar_express
):
    mem = FastWeightAttention(512, causal = causal, muon_update = muon_update, use_polar_express = use_polar_express, use_gates = use_gates, max_fast_weight_norm = max_fast_weight_norm)

    tokens = torch.randn(1, 64, 512)

    past_mem = None

    retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)
    retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)
    retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)

    assert retrieved.shape == tokens.shape

@param('causal', (False, True))
@param('use_gates', (False, True))
def test_chunked(causal, use_gates):
    seq_len = 32
    chunk_size = 8

    net = FastWeightAttention(
        dim = 16,
        dim_head = 8,
        heads = 2,
        causal = causal,
        use_gates = use_gates,
        chunk_size = chunk_size,
        use_forget_gate = use_gates
    )
    tokens = torch.randn(1, seq_len, 16)

    # all at once

    out_all, state_all = net(tokens, return_next_memories = True)

    # streaming with call boundaries at multiples of the chunk size

    outs = []
    past_mem = None
    curr_idx = 0

    while curr_idx < seq_len:
        step_size = chunk_size * random.randint(1, 3)
        next_idx = min(curr_idx + step_size, seq_len)

        chunk = tokens[:, curr_idx:next_idx, :]
        out_chunk, past_mem = net(chunk, return_next_memories = True, past_mem = past_mem)

        outs.append(out_chunk)

        curr_idx = next_idx

    out_chunked = torch.cat(outs, dim = 1)

    # validate output parity

    assert torch.allclose(out_all, out_chunked, atol = 1e-4)

    # validate memory parity

    for k in state_all.memory:
        assert torch.allclose(state_all.memory[k], past_mem.memory[k], atol = 1e-4)

    # validate token count

    assert state_all.token_count == past_mem.token_count == seq_len

# sequential (chunk-by-chunk streaming) vs parallel (one-shot) must agree
# on outputs, next memories, and gradients, for any streaming that respects chunk boundaries

@param('use_reverse_causal_target', (False, True))
@param('max_fast_weight_norm', (None, 2.))
@param('muon_update,use_polar_express,use_forget_gate', [
    (False, False, False),
    (True, False, True),
    (True, True, True)
])
def test_sequential_parallel_parity(
    use_reverse_causal_target,
    muon_update,
    use_polar_express,
    max_fast_weight_norm,
    use_forget_gate
):
    seq_len = 30
    chunk_size = 4

    def build():
        return FastWeightAttention(
            dim = 16,
            dim_head = 8,
            heads = 2,
            chunk_size = chunk_size,
            causal = True,
            use_gates = True,
            use_forget_gate = use_forget_gate,
            muon_update = muon_update,
            use_polar_express = use_polar_express,
            max_fast_weight_norm = max_fast_weight_norm,
            use_reverse_causal_target = use_reverse_causal_target
        )

    torch.manual_seed(0)
    tokens = torch.randn(2, seq_len, 16)

    # parallel - whole sequence in one shot

    torch.manual_seed(1)
    net_parallel = build()
    out_parallel, state_parallel = net_parallel(tokens, return_next_memories = True)
    out_parallel.pow(2).mean().backward()
    grads_parallel = {k: v.grad.clone() for k, v in net_parallel.named_parameters() if v.grad is not None}

    # sequential - stream chunk by chunk, respecting chunk boundaries

    torch.manual_seed(1)
    net_sequential = build()

    outs, past_mem, curr_idx = [], None, 0

    while curr_idx < seq_len:
        step_size = chunk_size * random.randint(1, 3)
        next_idx = min(curr_idx + step_size, seq_len)

        chunk = tokens[:, curr_idx:next_idx, :]
        out_chunk, past_mem = net_sequential(chunk, past_mem = past_mem, return_next_memories = True)

        outs.append(out_chunk)
        curr_idx = next_idx

    out_sequential = torch.cat(outs, dim = 1)
    out_sequential.pow(2).mean().backward()
    grads_sequential = {k: v.grad.clone() for k, v in net_sequential.named_parameters() if v.grad is not None}

    # forward parity

    assert torch.allclose(out_parallel, out_sequential, atol = 1e-4)

    # memory parity

    for k in state_parallel.memory:
        assert torch.allclose(state_parallel.memory[k], past_mem.memory[k], atol = 1e-4)

    assert state_parallel.token_count == past_mem.token_count == seq_len

    # gradient parity

    assert set(grads_parallel) == set(grads_sequential)

    for name, grad in grads_parallel.items():
        assert torch.allclose(grad, grads_sequential[name], atol = 1e-4), name

def test_chunked_streaming_partial():
    net = FastWeightAttention(dim = 16, dim_head = 8, heads = 2, chunk_size = 8)

    seq_len = 23
    tokens = torch.randn(1, seq_len, 16)

    # stream with arbitrary step sizes - partial chunks are processed immediately,
    # with the last token of each incomplete chunk excluded from the fast weight update

    outs = []
    past_mem = None
    curr_idx = 0

    while curr_idx < seq_len:
        step_size = random.randint(1, 5)
        next_idx = min(curr_idx + step_size, seq_len)

        chunk = tokens[:, curr_idx:next_idx, :]
        out_chunk, past_mem = net(chunk, return_next_memories = True, past_mem = past_mem)

        assert out_chunk.shape[-2] == chunk.shape[-2]

        outs.append(out_chunk)

        curr_idx = next_idx

    assert torch.cat(outs, dim = 1).shape[-2] == seq_len
    assert past_mem.token_count == seq_len
