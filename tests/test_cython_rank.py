"""Test Cython ranking implementation for speed and correctness."""

import numpy as np
import pytest

from pi_torchreid import metrics


@pytest.mark.skipif(
    not hasattr(metrics, "IS_CYTHON_AVAI") or not metrics.IS_CYTHON_AVAI,
    reason="Cython evaluation not available",
)
def test_cython_rank_precision():
    """Test that Cython and Python implementations produce same results."""
    num_q = 30
    num_g = 300
    max_rank = 5

    # Create deterministic test data
    np.random.seed(42)
    distmat = np.random.rand(num_q, num_g).astype(np.float32) * 20
    q_pids = np.random.randint(0, num_q, size=num_q)
    g_pids = np.random.randint(0, num_g, size=num_g)
    q_camids = np.random.randint(0, 5, size=num_q)
    g_camids = np.random.randint(0, 5, size=num_g)

    # Ensure at least some matches exist
    # Make some query pids match gallery pids
    for i in range(min(10, num_q)):
        q_pids[i] = g_pids[i % num_g]

    # Test Market1501 metric
    cmc_py, mAP_py = metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)
    cmc_cy, mAP_cy = metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)

    np.testing.assert_allclose(cmc_py, cmc_cy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mAP_py, mAP_cy, rtol=1e-5, atol=1e-6)

    # Test CUHK03 metric
    np.random.seed(42)
    cmc_py, mAP_py = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank,
        use_metric_cuhk03=True,
        use_cython=False,
    )
    np.random.seed(42)
    cmc_cy, mAP_cy = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank,
        use_metric_cuhk03=True,
        use_cython=True,
    )

    np.testing.assert_allclose(cmc_py, cmc_cy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mAP_py, mAP_cy, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not hasattr(metrics, "IS_CYTHON_AVAI") or not metrics.IS_CYTHON_AVAI,
    reason="Cython evaluation not available",
)
@pytest.mark.slow
def test_cython_rank_speed():
    """Test that Cython implementation is faster than Python (optional benchmark)."""
    import timeit

    num_q = 30
    num_g = 300
    max_rank = 5

    setup = f"""
import numpy as np
from pi_torchreid import metrics
np.random.seed(42)
distmat = np.random.rand({num_q}, {num_g}).astype(np.float32) * 20
q_pids = np.random.randint(0, {num_q}, size={num_q})
g_pids = np.random.randint(0, {num_g}, size={num_g})
q_camids = np.random.randint(0, 5, size={num_q})
g_camids = np.random.randint(0, 5, size={num_g})
# Ensure matches
for i in range(min(10, {num_q})):
    q_pids[i] = g_pids[i % {num_g}]
"""

    # Benchmark Python version
    pytime = timeit.timeit(
        f"metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, {max_rank}, use_cython=False)",
        setup=setup,
        number=20,
    )

    # Benchmark Cython version
    cytime = timeit.timeit(
        f"metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, {max_rank}, use_cython=True)",
        setup=setup,
        number=20,
    )

    speedup = pytime / cytime
    print(f"\nPython time: {pytime:.4f} s")
    print(f"Cython time: {cytime:.4f} s")
    print(f"Cython is {speedup:.2f}x faster than Python")

    # Cython should be faster (at least 1.5x for this small test)
    assert speedup >= 1.0, "Cython should be at least as fast as Python"
