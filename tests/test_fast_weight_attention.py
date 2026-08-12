import torch
import random

import pytest
param = pytest.mark.parametrize

from fast_weight_attention import FastWeightAttention, ChunkManager

# helpers

def exists(val):
    return val is not None

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
def test_chunk_manager(causal, use_gates):
    seq_len = 32
    chunk_size = 8

    net = FastWeightAttention(
        dim = 16,
        dim_head = 8,
        heads = 2,
        causal = causal,
        use_gates = use_gates
    )

    manager = ChunkManager(net, chunk_size = chunk_size, use_forget_gate = use_gates)
    tokens = torch.randn(1, seq_len, 16)

    # all at once

    out_all, mem_all = manager(tokens, return_next_memories = True)

    # streaming variable sized chunks

    outs = []
    past_mem = None
    curr_idx = 0

    while curr_idx < seq_len:
        step_size = random.randint(1, 4)
        next_idx = min(curr_idx + step_size, seq_len)

        chunk = tokens[:, curr_idx:next_idx, :]
        out_chunk, past_mem = manager(chunk, return_next_memories = True, past_mem = past_mem)

        if exists(out_chunk):
            outs.append(out_chunk)

        curr_idx = next_idx

    out_chunked = torch.cat(outs, dim = 1)

    # validate output parity

    assert torch.allclose(out_all, out_chunked, atol = 1e-4)

    # validate memory parity

    for k in mem_all.memory:
        assert torch.allclose(mem_all.memory[k], past_mem.memory[k], atol = 1e-4)

def test_chunk_manager_partial_chunk_return_type():
    net = FastWeightAttention(dim = 16, dim_head = 8, heads = 2)
    manager = ChunkManager(net, chunk_size = 8)

    partial = torch.randn(1, 3, 16)

    # a partial chunk is buffered, so nothing is emitted

    res, past_mem = manager(partial, return_next_memories = True)
    assert res is None
    assert exists(past_mem.buffer)

    # without return_next_memories, a plain None is returned (not a tuple)

    res = manager(partial, return_next_memories = False)
    assert res is None
