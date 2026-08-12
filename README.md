## Fast Weight Attention

An attention based fast weight episodic memory, in the same vein as the memory MLP from TTT / [Titans](https://codeberg.org/lucidrains/titans-pytorch) and [fast weight PKM](https://codeberg.org/lucidrains/fast-weight-product-key-memory) from Sakana AI

## Install

```bash
$ pip install fast-weight-attention
```

## Usage

```python
import torch
from fast_weight_attention import FastWeightAttention

mem = FastWeightAttention(512, causal = True)

tokens = torch.randn(1, 64, 512)

past_mem = None

retrieved, next_mem = mem(tokens, past_mem = past_mem, return_next_memories = True)
retrieved, next_mem = mem(tokens, past_mem = next_mem, return_next_memories = True)
retrieved, next_mem = mem(tokens, past_mem = next_mem, return_next_memories = True)

assert retrieved.shape == tokens.shape

# you can then retrieve without fast weight updating

retrieved = mem(tokens, return_next_memories = False)
```

With chunked processing (automatically segments the sequence and carries memory across chunks):

```python
import torch
from fast_weight_attention import ChunkedFastWeightAttention

mem = ChunkedFastWeightAttention(
    512,
    causal = True,
    chunk_size = 64   # process 64 tokens at a time, carrying fast weight memories across chunks
)

tokens = torch.randn(1, 512, 512)

retrieved, next_mem = mem(tokens, return_next_memories = True)

assert retrieved.shape == tokens.shape
```

> Note: a trailing segment shorter than `chunk_size` is buffered (not processed) so that
> chunk boundaries stay consistent across streaming calls — feed it the remaining tokens
> (or the next tokens) in a subsequent call to have it emitted. If you need the full
> output from a single call, make the sequence length a multiple of `chunk_size`.

## Citations

```bibtex
@article{zhang2026loger,
    title   = {LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory},
    author  = {Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Sun, Chen and Yang, Ming-Hsuan and Cole, Forrester and Darrell, Trevor and Sun, Deqing},
    journal = {arXiv preprint arXiv:2603.03269},
    year    = {2026}
}
```

```bibtex
@misc{zhao2026fastweightproductkeymemory,
    title   = {Fast-weight Product Key Memory},
    author  = {Tianyu Zhao and Llion Jones},
    year    = {2026},
    eprint  = {2601.00671},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2601.00671},
}
```

```bibtex
@misc{jordan2024muon,
    author  = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein},
    title   = {Muon: An optimizer for hidden layers in neural networks},
    year    = {2024},
    url     = {https://kellerjordan.github.io/posts/muon/}
}
```

```bibtex
@article{Yaghoubietal2026,
    author  = {Yaghoubi, Mohammad and Nieto-Posadas, Andres and Mosser, Coralie-Anne and Gisiger, Thomas and Wilson, Émmanuel and Williams, Sylvain and Brandon, Mark P.},
    title   = {Predictive coding of reward in the hippocampus},
    journal = {Nature},
    year    = {2026},
    doi     = {10.1038/s41586-025-09958-0}
}
```

```bibtex
@article{volchkov2026cliptogrok,
    title   = {Clip to Grok: Weight Norm Clipping for Accelerated Generalization},
    author  = {Volchkov, Vladimir and Rivlin, Aviad},
    year    = {2026},
    journal = {arXiv preprint},
    note    = {Implementation available at \url{https://github.com/NiftyliuS/cliptogrok}}
}
```
