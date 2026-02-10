"""Log-linear sparse attention: hierarchical top-k block selection in plain MLX.

Tokens are mean-pooled by a factor `block` repeatedly, giving levels 0..L where
level l has N / block^l tokens. Selection runs coarse to fine:

  * At the coarsest level every token is a candidate.
  * At level l, every group of `block` consecutive level-l queries looks only at its
    candidate keys, and each query keeps its `topk` best. A selected level-l token
    is a level-(l-1) block, so its `block` children become the candidates one level
    down. Candidate lists therefore never exceed topk * block, and the total
    selection cost is O(N * topk) instead of O(N^2).
  * At level 0 a query block attends to the fine keys chosen at the end of the
    descent, plus (enrichment) the candidate tokens at every coarser level, which
    stand in for everything the selection skipped.

A coarse token at level l summarises block^l fine tokens, so its attention logit is
raised by l * log(block): it then counts as block^l tokens in the softmax. Total keys
per query are O(topk * block * levels), so attention is O(N log N) overall.

Selection is not differentiated through; gradients flow through the gathered
keys and values.
"""

import math

import mlx.core as mx


def num_levels(n, block):
    """Largest L with block^(L+1) <= n, i.e. floor(log_block(n)) - 1."""
    levels, power = 0, block * block
    while power <= n:
        levels, power = levels + 1, power * block
    return levels


def keys_per_query(n, block, topk, levels=None, enrich=None):
    """Number of keys each query attends to under LLSA."""
    levels = num_levels(n, block) if levels is None else levels
    enrich = levels if enrich is None else enrich
    total = 0
    for level in range(enrich + 1):
        total += n // block**levels if level == levels else min(topk * block, n // block**level)
    return total


def _gather(x, idx):
    """x (G, N, d), idx (G, T, C) -> (G, T, C, d).

    Rows are fetched from a flattened (G*N, d) table. Broadcasting x across T and
    using take_along_axis looks equivalent, but its gradient scatters into a dense
    (G, T, N, d) buffer, which brings back the quadratic memory this method avoids.
    """
    G, N, d = x.shape
    flat = (idx + (mx.arange(G) * N)[:, None, None]).reshape(-1)
    return x.reshape(G * N, d)[flat].reshape(*idx.shape, d)


def _pyramid(x, block, levels):
    G, N, d = x.shape
    out = [x]
    for _ in range(levels):
        x = x.reshape(G, -1, block, d).mean(axis=2)
        out.append(x)
    return out


def hierarchical_select(qs, ks, block, topk):
    """Return {level: candidate token indices (G, T_level, C_level)}.

    cands[L] lists every coarsest token; cands[l < L] are the children of what the
    level above selected; cands[0] are the fine keys each level-0 block attends to.
    """
    levels = len(qs) - 1
    G, n_top, d = ks[levels].shape
    cand = mx.broadcast_to(mx.arange(n_top)[None, None], (G, n_top // block, n_top))
    cands = {levels: cand}
    for level in range(levels, 0, -1):
        q = qs[level].reshape(G, -1, block, d)
        scores = q @ _gather(ks[level], cand).swapaxes(-1, -2)
        k = min(topk, cand.shape[-1])
        best = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        shared = mx.broadcast_to(cand[:, :, None], (*scores.shape[:3], cand.shape[-1]))
        picked = mx.take_along_axis(shared, best, axis=-1).reshape(G, -1, k)
        cand = (picked[..., None] * block + mx.arange(block)).reshape(G, picked.shape[1], k * block)
        cands[level - 1] = cand
    return cands


def llsa(q, k, v, block=16, topk=8, levels=None, enrich=None, reweight=True):
    """q, k, v: (batch, heads, N, d). Bidirectional, as in a diffusion transformer.

    levels:   number of coarse levels L (default floor(log_block N) - 1)
    enrich:   how many coarse levels to append as extra keys (default all L, 0 = none)
    reweight: weight coarse tokens by block^level in the softmax
    """
    Bt, H, N, d = q.shape
    levels = num_levels(N, block) if levels is None else levels
    enrich = levels if enrich is None else enrich
    if levels < 1 or N % block ** (levels + 1):
        raise ValueError(f"N={N} must be a multiple of block^(levels+1)={block ** (levels + 1)} with levels>=1")
    q, k, v = (x.reshape(Bt * H, N, d) for x in (q, k, v))
    qs, ks, vs = (_pyramid(x, block, levels) for x in (q, k, v))
    cands = hierarchical_select(
        [mx.stop_gradient(x) for x in qs], [mx.stop_gradient(x) for x in ks], block, topk
    )

    n_blocks = N // block
    keys, values, bias = [], [], []
    for level in range(enrich + 1):
        idx = mx.repeat(cands[level], block**level, axis=1)  # one row per level-0 block
        keys.append(_gather(ks[level], idx))
        values.append(_gather(vs[level], idx))
        weight = level * math.log(block) if reweight else 0.0
        bias.append(mx.full((idx.shape[-1],), weight))
    keys, values, bias = mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2), mx.concatenate(bias)

    qb = q.reshape(Bt * H, n_blocks, block, d)
    logits = (qb @ keys.swapaxes(-1, -2)) * d**-0.5 + bias
    out = mx.softmax(logits, axis=-1) @ values
    return out.reshape(Bt, H, N, d)


def full_attention(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
