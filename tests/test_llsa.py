import mlx.core as mx
import numpy as np
import pytest

from llsa import full_attention, keys_per_query, llsa, num_levels, zorder_permutation


def qkv(N=256, d=16, seed=0, B=1, H=2):
    mx.random.seed(seed)
    return tuple(mx.random.normal((B, H, N, d)) for _ in range(3))


def test_num_levels():
    assert num_levels(16384, 16) == 2
    assert num_levels(256, 4) == 3
    assert num_levels(1024, 4) == 4


def test_single_level_selecting_everything_equals_full_attention():
    q, k, v = qkv(N=256)
    out = llsa(q, k, v, block=4, topk=64, levels=1, enrich=0)
    np.testing.assert_allclose(np.array(out), np.array(full_attention(q, k, v)), atol=1e-4)


def test_constant_values_pass_through_whatever_the_weights():
    q, k, _ = qkv(N=256)
    v = mx.ones_like(q) * 3.0
    out = llsa(q, k, v, block=4, topk=4)
    np.testing.assert_allclose(np.array(out), 3.0, atol=1e-5)


def test_reweighting_changes_the_output_and_enrichment_does_too():
    q, k, v = qkv(N=256)
    base = llsa(q, k, v, block=4, topk=4, enrich=0)
    enriched = llsa(q, k, v, block=4, topk=4, reweight=False)
    weighted = llsa(q, k, v, block=4, topk=4, reweight=True)
    assert not np.allclose(np.array(base), np.array(enriched), atol=1e-4)
    assert not np.allclose(np.array(enriched), np.array(weighted), atol=1e-4)


def test_selection_finds_a_distant_matching_block():
    # A query block whose keys match one far-away block should pull that block's values.
    N, d, B = 1024, 16, 4
    rng = np.random.default_rng(0)
    k = rng.normal(size=(1, 1, N, d)).astype(np.float32) * 0.1
    v = rng.normal(size=(1, 1, N, d)).astype(np.float32)
    target = slice(700, 700 + B)
    direction = rng.normal(size=d).astype(np.float32)
    direction /= np.linalg.norm(direction)
    k[0, 0, target] = direction * 6
    q = np.zeros_like(k)
    q[0, 0, 8 : 8 + B] = direction * 6
    v_np = v
    q, k, v = map(mx.array, (q, k, v))
    approx = np.array(llsa(q, k, v, block=B, topk=4))[0, 0, 8:12]
    exact = np.array(full_attention(q, k, v))[0, 0, 8:12]
    np.testing.assert_allclose(approx, exact, atol=0.05)
    np.testing.assert_allclose(approx.mean(0), v_np[0, 0, target].mean(0), atol=0.1)


def test_gradients_are_finite_for_all_inputs():
    q, k, v = qkv(N=256)
    grads = mx.grad(lambda q, k, v: llsa(q, k, v, block=4, topk=4).sum(), argnums=(0, 1, 2))(q, k, v)
    for g in grads:
        assert np.isfinite(np.array(g)).all() and float(mx.abs(g).max()) > 0


def test_rejects_lengths_that_do_not_tile():
    q, k, v = qkv(N=100)
    with pytest.raises(ValueError):
        llsa(q, k, v, block=4, topk=4)


def test_keys_per_query_matches_the_actual_gather_and_grows_slowly():
    N, B, K = 1024, 4, 4
    assert keys_per_query(N, B, K) == K * B * num_levels(N, B) + N // B ** num_levels(N, B)
    assert keys_per_query(1 << 20, 16, 8) < 2000
    assert keys_per_query(1 << 20, 16, 8) / keys_per_query(1 << 12, 16, 8) < 2


def test_zorder_is_a_permutation_with_local_groups():
    perm = zorder_permutation(8)
    assert sorted(perm.tolist()) == list(range(64))
    for g in range(16):  # every aligned group of 4 tokens is a 2x2 patch
        rows, cols = perm[4 * g : 4 * g + 4] // 8, perm[4 * g : 4 * g + 4] % 8
        assert rows.max() - rows.min() == 1 and cols.max() - cols.min() == 1
        assert rows.min() % 2 == 0 and cols.min() % 2 == 0
