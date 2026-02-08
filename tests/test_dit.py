import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from llsa.dit import DiTConfig, PixelDiT, flow_loss, noise_scale, sample


def tiny(attention):
    return PixelDiT(DiTConfig(size=16, dim=32, heads=2, depth=2, attention=attention, block=4, topk=2))


@pytest.mark.parametrize("attention", ["full", "llsa"])
def test_output_shape_and_zero_init(attention):
    mx.random.seed(0)
    model = tiny(attention)
    x = mx.random.normal((3, 16, 16, 1))
    out = model(x, mx.array([0.1, 0.5, 0.9]))
    assert out.shape == x.shape
    # adaLN-zero: every block starts as the identity
    h = mx.random.normal((3, 256, 32))
    for block in model.blocks:
        np.testing.assert_allclose(np.array(block(h, mx.zeros((3, 32)))), np.array(h), atol=1e-6)


@pytest.mark.parametrize("attention", ["full", "llsa"])
def test_loss_has_finite_nonzero_gradients(attention):
    mx.random.seed(0)
    model = tiny(attention)
    x0 = mx.random.normal((4, 16, 16, 1))
    loss, grads = nn.value_and_grad(model, lambda m: flow_loss(m, x0))(model)
    flat = [g for _, g in tree_flatten(grads)]
    assert all(np.isfinite(np.array(g)).all() for g in flat)
    assert any(float(mx.abs(g).max()) > 0 for g in flat)


def test_noise_scale_follows_the_paper_rule():
    assert noise_scale(64) == 1.0 and noise_scale(32) == 1.0
    assert noise_scale(128) == 2.0 and noise_scale(256) == 4.0


def test_sampler_runs_and_is_finite():
    mx.random.seed(0)
    out = sample(tiny("llsa"), 2, steps=3)
    assert out.shape == (2, 16, 16, 1) and np.isfinite(np.array(out)).all()
