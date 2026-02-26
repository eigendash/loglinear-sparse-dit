"""Time attention forward and backward for full attention vs LLSA across lengths.

    python scripts/benchmark.py
    python scripts/benchmark.py --lengths 1024 4096 16384 --block 16 --topk 8
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llsa import full_attention, keys_per_query, llsa  # noqa: E402


def timed(fn, iters):
    """Mean milliseconds per call, or None if Metal cannot allocate the buffers."""
    try:
        mx.eval(fn())
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        return (time.perf_counter() - start) / iters * 1000
    except RuntimeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32768])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    print("| N | keys/query | full fwd ms | LLSA fwd ms | full fwd+bwd ms | LLSA fwd+bwd ms |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n in args.lengths:
        mx.random.seed(0)
        q, k, v = (mx.random.normal((1, args.heads, n, args.dim)) for _ in range(3))
        mx.eval(q, k, v)
        sparse = lambda q, k, v: llsa(q, k, v, block=args.block, topk=args.topk)  # noqa: E731
        try:
            sparse(q, k, v)
        except ValueError:
            print(f"| {n} | n/a: not a multiple of block^(levels+1) | | | | |")
            continue
        grad_full = mx.grad(lambda q, k, v: full_attention(q, k, v).sum(), argnums=(0, 1, 2))
        grad_sparse = mx.grad(lambda q, k, v: sparse(q, k, v).sum(), argnums=(0, 1, 2))
        row = [
            timed(lambda: full_attention(q, k, v), args.iters),
            timed(lambda: sparse(q, k, v), args.iters),
            timed(lambda: grad_full(q, k, v), args.iters),
            timed(lambda: grad_sparse(q, k, v), args.iters),
        ]
        keys = keys_per_query(n, args.block, args.topk)
        cells = ["out of memory" if t is None else f"{t:.1f}" for t in row]
        print(f"| {n} | {keys} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
