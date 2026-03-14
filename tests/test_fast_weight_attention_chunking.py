import pytest
import torch
from fast_weight_attention import ChunkedFastWeightAttention

param = pytest.mark.parametrize

# tests

@param('heads', [1, 4])
@param('muon_update', [True, False])
@param('causal', [True, False])
def test_fast_weight_attention_basic(heads, muon_update, causal):
    model = ChunkedFastWeightAttention(dim = 32, heads = heads, dim_head = 16, muon_update = muon_update, causal = causal)
    tokens = torch.randn(2, 64, 32)

    out, memories = model(
        tokens,
        return_next_memories = True
    )

    retrieved = model(
        tokens,
        past_mem = memories
    )

    assert tokens.shape == retrieved.shape

@param('causal', [True, False])
@param('muon_update', [True, False])
def test_fast_weight_attention_basic_parity(causal, muon_update):
    dim = 32
    seq_len = 32
    chunk_size = 16

    model = ChunkedFastWeightAttention(
        dim = dim,
        heads = 4,
        dim_head = 16,
        chunk_size = chunk_size,
        muon_update = muon_update,
        causal = causal
    )

    tokens = torch.randn(1, seq_len, dim)

    # parallel processing

    parallel_out, parallel_state = model(
        tokens,
        return_next_memories = True
    )

    # sequential manual chunked processing

    seq_a = tokens[:, :16]
    seq_b = tokens[:, 16:]

    out_a, state_a = model(
        seq_a,
        return_next_memories = True
    )
    out_b, state_b = model(
        seq_b,
        return_next_memories = True,
        past_mem = state_a
    )

    sequential_out = torch.cat((out_a, out_b), dim = 1)

    # verify parity

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-5)

    for key in parallel_state.memory:
        assert torch.allclose(parallel_state.memory[key], state_b.memory[key], atol = 1e-5)

@param('causal', [True, False])
@param('muon_update', [True, False])
@param('seq_len, chunk_size', [
    (16, 8),
    (32, 16),
    (48, 16),
    (17, 16) # edge case: remainder of 1 correctly prepends last_token to pass seq_len > 1 assert
])
def test_fast_weight_attention_parity_multi(causal, muon_update, seq_len, chunk_size):
    dim = 32
    model = ChunkedFastWeightAttention(
        dim = dim,
        heads = 4,
        dim_head = 16,
        chunk_size = chunk_size,
        muon_update = muon_update,
        causal = causal
    )

    tokens = torch.randn(1, seq_len, dim)

    # parallel

    parallel_out, parallel_state = model(
        tokens,
        return_next_memories = True
    )

    # sequential

    state = None
    sequential_outputs = []

    for chunk in tokens.split(chunk_size, dim = 1):
        out, state = model(
            chunk,
            return_next_memories = True,
            past_mem = state
        )
        sequential_outputs.append(out)

    sequential_out = torch.cat(sequential_outputs, dim = 1)

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-5)

    for key in parallel_state.memory:
        assert torch.allclose(parallel_state.memory[key], state.memory[key], atol = 1e-5)

@param('causal', [True, False])
def test_fast_weight_attention_causality(causal):
    dim = 32
    seq_len = 8
    model = ChunkedFastWeightAttention(
        dim = dim,
        heads = 4,
        dim_head = 16,
        causal = causal
    )

    tokens = torch.randn(1, seq_len, dim, requires_grad = True)

    output = model(tokens)

@param('causal', [True])
@param('muon_update', [True])
def test_fast_weight_attention_unaligned_chunks(causal, muon_update):
    torch.manual_seed(42)
    dim = 32
    seq_len = 32

    model = ChunkedFastWeightAttention(
        dim = dim,
        heads = 4,
        dim_head = 16,
        causal = causal,
        chunk_size = 16,
        muon_update = muon_update
    )

    tokens = torch.randn(1, seq_len, dim)

    # split into unaligned random chunks (must be > 1 length per chunk)
    splits = [7, 10, 5, 10]
    assert sum(splits) == seq_len

    curr_state = None
    serial_outputs = []

    pos = 0
    for split in splits:
        chunk = tokens[:, pos:pos+split]
        out, curr_state = model(
            chunk,
            return_next_memories = True,
            past_mem = curr_state
        )
        serial_outputs.append(out)
        pos += split

    sequential_out = torch.cat(serial_outputs, dim = 1)

    assert sequential_out.shape == tokens.shape

    for key in curr_state.memory:
        assert not curr_state.memory[key].isnan().any()

def test_fast_weight_attention_detach_memories():
    dim = 32
    seq_len = 32

    model = ChunkedFastWeightAttention(
        dim = dim,
        heads = 4,
        dim_head = 16,
        chunk_size = 16
    )

    # with detach
    tokens = torch.randn(1, seq_len, dim, requires_grad = True)
    out, _ = model(tokens, return_next_memories = True, detach_next_memories_every = 1)
    out.sum().backward()
    grad_detached = tokens.grad.clone()

    # without detach
    tokens2 = tokens.detach().clone().requires_grad_(True)
    out2, _ = model(tokens2, return_next_memories = True)
    out2.sum().backward()
    grad_full = tokens2.grad.clone()

    # detaching memory should change the gradient profile
    assert not torch.allclose(grad_detached, grad_full, atol = 1e-6), 'detach_next_memories_every had no effect on gradients'
