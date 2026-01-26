"""Integration tests for Cython vs Python equivalence."""

import pytest
import numpy as np
from torchreid import metrics


@pytest.mark.skipif(
    not hasattr(metrics, "IS_CYTHON_AVAI") or not metrics.IS_CYTHON_AVAI,
    reason="Cython evaluation not available",
)
class TestCythonPythonEquivalence:
    """Test that Cython and Python implementations produce identical results."""

    def test_evaluate_rank_equivalence_market1501(self):
        """Test Market1501 metric equivalence."""
        np.random.seed(42)
        num_q = 30
        num_g = 300
        max_rank = 10

        distmat = np.random.rand(num_q, num_g).astype(np.float32) * 20
        q_pids = np.random.randint(0, num_q, size=num_q)
        g_pids = np.random.randint(0, num_g, size=num_g)
        q_camids = np.random.randint(0, 5, size=num_q)
        g_camids = np.random.randint(0, 5, size=num_g)

        # Ensure at least some matches exist
        for i in range(min(10, num_q)):
            q_pids[i] = g_pids[i % num_g]

        cmc_py, mAP_py = metrics.evaluate_rank(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False
        )
        cmc_cy, mAP_cy = metrics.evaluate_rank(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True
        )

        np.testing.assert_allclose(cmc_py, cmc_cy, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(mAP_py, mAP_cy, rtol=1e-5, atol=1e-6)

    def test_evaluate_rank_equivalence_cuhk03(self):
        """Test CUHK03 metric equivalence."""
        np.random.seed(42)
        num_q = 30
        num_g = 300
        max_rank = 10

        distmat = np.random.rand(num_q, num_g).astype(np.float32) * 20
        q_pids = np.random.randint(0, num_q, size=num_q)
        g_pids = np.random.randint(0, num_g, size=num_g)
        q_camids = np.random.randint(0, 5, size=num_q)
        g_camids = np.random.randint(0, 5, size=num_g)

        # Ensure at least some matches exist
        for i in range(min(10, num_q)):
            q_pids[i] = g_pids[i % num_g]

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

    def test_evaluate_rank_equivalence_various_sizes(self):
        """Test equivalence with various dataset sizes."""
        for num_q, num_g in [(5, 20), (10, 50), (20, 100), (50, 200)]:
            np.random.seed(42)
            distmat = np.random.rand(num_q, num_g).astype(np.float32) * 20
            q_pids = np.random.randint(0, min(num_q, 10), size=num_q)
            g_pids = np.random.randint(0, min(num_g, 10), size=num_g)
            q_camids = np.random.randint(0, 5, size=num_q)
            g_camids = np.random.randint(0, 5, size=num_g)

            # Ensure matches
            for i in range(min(5, num_q)):
                q_pids[i] = g_pids[i % num_g]

            cmc_py, mAP_py = metrics.evaluate_rank(
                distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10, use_cython=False
            )
            cmc_cy, mAP_cy = metrics.evaluate_rank(
                distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10, use_cython=True
            )

            np.testing.assert_allclose(cmc_py, cmc_cy, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(mAP_py, mAP_cy, rtol=1e-5, atol=1e-6)
