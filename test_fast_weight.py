import torch
from fast_weight_attention import FastWeightAttention

def test_fast_weight_attention():
    # Test with newtonschulz5 (muon_update=True, use_polar_express=False)
    attn_newton = FastWeightAttention(
        dim=64,
        dim_head=16,
        heads=4,
        muon_update=True,
        use_polar_express=False
    )
    tokens = torch.randn(2, 10, 64)
    pred_values, next_memories = attn_newton(tokens, return_next_memories=True)
    assert pred_values.shape == (2, 10, 64)
    assert 'wq' in next_memories.memory

    # Test with polar express (muon_update=True, use_polar_express=True)
    attn_polar = FastWeightAttention(
        dim=64,
        dim_head=16,
        heads=4,
        muon_update=True,
        use_polar_express=True
    )
    pred_values_polar, next_memories_polar = attn_polar(tokens, return_next_memories=True)
    assert pred_values_polar.shape == (2, 10, 64)
    assert 'wq' in next_memories_polar.memory

    print("All tests passed successfully.")

if __name__ == "__main__":
    test_fast_weight_attention()
