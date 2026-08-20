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

> Sequences are segmented internally at `chunk_size` boundaries (defaults to the full sequence), carrying
> the fast weight memories across chunks. The last token of each completed chunk is marked with a learned
> boundary embedding, and its store target wraps around to the chunk's first token - no lookahead needed.
> Chunks shorter than `chunk_size` are processed immediately, with the last token of an incomplete chunk
> excluded from the fast weight update.

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
