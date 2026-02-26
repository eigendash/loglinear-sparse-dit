# loglinear-sparse-dit

Log-linear sparse attention in plain MLX, plus a small pixel-space diffusion transformer to try it on. It runs on Apple silicon with no custom kernels.

The method is from *Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers* (CVPR 2026, arXiv:2512.16615). I wrote this from the paper's algorithm and complexity analysis. It is a small independent implementation, not a reproduction of the paper's FID or throughput numbers, which come from a CUDA kernel on an H200.

## Method as implemented

Block-sparse top-k attention picks, for each block of queries, the k most similar key blocks and attends only to those. The expensive part is the selection: scoring every query block against every key block is quadratic in sequence length. LLSA removes that by selecting coarse to fine.

Tokens are mean-pooled by a factor `block` repeatedly, giving levels 0 to L where level l has `N / block^l` tokens. At the coarsest level every token is a candidate. At each level, every group of `block` consecutive queries scores only its candidate keys and each query keeps its top-k. A selected token at level l is a block at level l-1, so its children become the candidates one level down. The candidate list therefore never grows past `topk * block`, and the selection cost summed over levels is a geometric series, O(N k).

Attention then runs over the fine keys that survived the descent plus the candidate tokens at every coarser level (the paper's KV enrichment). These coarse tokens stand in for whatever the selection skipped. Since a level-l token summarises `block^l` fine tokens, I add `l * log(block)` to its attention logit, so it counts as that many tokens in the softmax. Each query sees on the order of `topk * block * L` keys, which gives O(N log N) in total.

Two readings of the paper are mine and could differ from the original. The paper describes the reweighting as multiplying coarse keys and values by `block^l`. I implemented the log-logit form because it is the one that makes a coarse token count as `block^l` tokens in a softmax. And the pseudocode's indexing is loose about whether top-k is taken per query token or per query block; I take it per query token among the candidates its block shares.

Selection is not differentiated through. Gradients flow through the gathered keys and values, and MLX's autodiff handles the gather, so there is no hand-written backward pass like the paper's index-transposition kernel.

For image data, `llsa/layout.py` reorders pixels into Z-order so that every aligned group of 4 tokens is a 2x2 patch and every group of 16 a 4x4 patch. Without this, consecutive row-major tokens pool pixels that are not neighbours. The diffusion model uses the paper's noise rescaling for images above 64x64, `x_t = (1-t) x0 + s t eps` with `s = n/64`.

## Checks

`pytest` runs 16 tests. The ones that matter most: with a single level and every block selected, LLSA equals full attention to 1e-4; with constant values the output equals those values for any weights; a planted distant matching block is found and attended to; gradients are finite for q, k and v; and the number of keys per query grows slowly (under 2x going from 4k to 1M tokens at block 16 and top-k 8).

## Toy diffusion experiment

`scripts/train_toy.py` trains the same 4-layer, 64-wide pixel DiT on procedurally generated 32x32 images (a few Gaussian blobs and sometimes a bar), once with full attention and once with LLSA (block 4, top-k 4, so 68 keys per query against 1024). Both train for 2000 steps with the same seed and are evaluated on the same held-out images with the same noise and timesteps.

| attention | held-out flow-matching loss |
|---|---:|
| full | 0.4698 |
| LLSA | 0.4870 |

LLSA is about 4% worse here. I did not run more seeds, so I cannot say how much of that gap is noise. The paper's claim is that LLSA matches or beats full attention in FID, and a toy at this scale is a weak test of it. The samples from both models are speckled blob-like noise: 2000 steps on a tiny model is not enough to generate clean images, and I make no claim about sample quality.

## Speed

`scripts/benchmark.py` times forward and forward-plus-backward for both attentions at several lengths. MLX's full attention is a fused kernel and LLSA here is a sequence of gathers and matmuls, so I expected full attention to win at short lengths.

Measured on an M4 Pro with the GPU otherwise idle: batch 1, 4 heads, head dimension 64, block 16, top-k 8, mean of 5 calls after a warm-up call, one run. Times are milliseconds.

| N | keys per query | full fwd | LLSA fwd | full fwd+bwd | LLSA fwd+bwd |
|---:|---:|---:|---:|---:|---:|
| 1024 | 192 | 1.5 | 2.2 | 1.8 | 1.9 |
| 4096 | 272 | 4.3 | 4.1 | 21.1 | 8.6 |
| 16384 | 320 | 64.5 | 17.2 | 342.0 | 39.3 |
| 32768 | 384 | 272.3 | 40.5 | out of memory | 92.3 |

Full attention is faster at 1k tokens and the two are even on the forward pass at 4k. From there the gap grows with length. The backward pass of MLX's full attention fails at 32k tokens: it asks for a 17.2 GB buffer, which is the size of the 4-head float32 score matrix and over the 12.9 GB single-buffer limit on this machine. LLSA's backward runs fine at that length.

This benchmark caught a real bug. My first gather broadcast the key tensor across all query rows, and its gradient scattered into a dense `rows x N x d` buffer, so the backward pass was quadratic in memory and ran out of it at 16k tokens. It now gathers from a flat table, and `tests/test_llsa.py` has a peak-memory test that fails on the old version.

## Limits

No fused kernel, so the gathered keys and values are materialised. That is O(N log N) in memory but with a large constant. Block size and top-k are fixed per call, there is no support for N that is not a multiple of `block^(L+1)` (it raises), attention is bidirectional only, and positions are a learned embedding instead of the RoPE the paper uses. The diffusion model is a toy, with no VAE-free high-resolution training, no low-resolution pretraining, and no FID.

## Running it

```
pip install -e ".[dev]"
pytest
python scripts/train_toy.py
python scripts/benchmark.py
```
