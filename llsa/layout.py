"""Token ordering for 2D data.

Row-major order puts vertical neighbours a full row apart, so pooling consecutive
tokens mixes unrelated pixels. Z-order (Morton) order interleaves row and column
bits, so every aligned group of 4 tokens is a 2x2 patch, every group of 16 a 4x4
patch, and so on. Mean-pooling consecutive tokens then pools spatial neighbourhoods.
"""

import numpy as np


def zorder_permutation(size):
    """perm such that tokens_z = tokens_rowmajor[perm]; size must be a power of two."""
    if size & (size - 1):
        raise ValueError("size must be a power of two")
    bits = size.bit_length() - 1
    p = np.arange(size * size)
    row = np.zeros_like(p)
    col = np.zeros_like(p)
    for i in range(bits):
        col |= ((p >> (2 * i)) & 1) << i
        row |= ((p >> (2 * i + 1)) & 1) << i
    return row * size + col
