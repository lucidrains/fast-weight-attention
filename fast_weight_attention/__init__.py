
from fast_weight_attention.fast_weight_attention import (
    FastWeightAttention
)

from fast_weight_attention.chunk_manager import (
    ChunkManager,
    ChunkingState
)

def ChunkedFastWeightAttention(*args, chunk_size = None, **kwargs):
    return ChunkManager(FastWeightAttention(*args, **kwargs), chunk_size = chunk_size)
