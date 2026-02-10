"""Train the same tiny pixel DiT with full attention and with LLSA on procedural images.

    python scripts/train_toy.py
    python scripts/train_toy.py --steps 3000 --size 32

Images are a few soft Gaussian blobs plus an occasional bright bar, so global layout
matters a little. This checks that LLSA trains like full attention on a toy; it is
not a benchmark of image quality.
"""

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llsa.dit import DiTConfig, PixelDiT, flow_loss, sample  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"


def make_images(rng, n, size):
    ys, xs = np.mgrid[0:size, 0:size] / size
    out = np.zeros((n, size, size, 1), dtype=np.float32)
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            cy, cx, w = rng.random(), rng.random(), 0.04 + 0.1 * rng.random()
            out[i, :, :, 0] += np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * w**2))
        if rng.random() < 0.5:
            row = rng.integers(0, size - 2)
            out[i, row : row + 2, :, 0] += 0.8
    return np.clip(out, 0, 1) * 2 - 1


def write_png(path, grid):
    """Minimal grayscale PNG writer so the script needs no imaging library."""
    img = ((np.clip(grid, -1, 1) + 1) * 127.5).astype(np.uint8)
    raw = b"".join(b"\x00" + row.tobytes() for row in img)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    head = struct.pack(">IIBBBBB", img.shape[1], img.shape[0], 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def run(attention, args, data, eval_set):
    mx.random.seed(args.seed)
    cfg = DiTConfig(size=args.size, attention=attention, block=args.block, topk=args.topk)
    model = PixelDiT(cfg)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=optim.cosine_decay(args.lr, args.steps), weight_decay=0.0)
    step = nn.value_and_grad(model, lambda m, x: flow_loss(m, x))
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    for i in range(1, args.steps + 1):
        batch = mx.array(data[rng.integers(0, len(data), args.batch)])
        loss, grads = step(model, batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if i % 500 == 0:
            print(f"  [{attention}] step {i:5d}  loss {float(loss):.4f}", flush=True)
    seconds = time.perf_counter() - start

    mx.random.seed(1234)  # the same noise and timesteps for every model
    losses = [float(flow_loss(model, mx.array(eval_set[j : j + 64]))) for j in range(0, len(eval_set), 64)]
    mx.random.seed(99)
    imgs = np.array(sample(model, 8))[..., 0]
    ART.mkdir(exist_ok=True)
    write_png(ART / f"samples_{attention}.png", np.concatenate(list(imgs), axis=1))
    return float(np.mean(losses)), seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data, eval_set = make_images(rng, 4096, args.size), make_images(rng, 512, args.size)
    rows = [(a, *run(a, args, data, eval_set)) for a in ("full", "llsa")]

    print("\n| attention | held-out flow loss | train time |")
    print("|---|---:|---:|")
    for name, loss, seconds in rows:
        print(f"| {name} | {loss:.4f} | {seconds:.0f}s |")
    print(f"\nsamples written to {ART}/samples_*.png")


if __name__ == "__main__":
    main()
