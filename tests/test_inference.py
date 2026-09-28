import unittest
import numpy as np

from src.inference import (
    total_time_on_test,
    mle,
    naive_estimator,
    exact_confidence_interval,
    wald_confidence_interval,
    log_wald_confidence_interval,
)


class TestInference(unittest.TestCase):

    def test_total_time_on_test(self):
        times = [3, 8, 5, 12, 20]

        self.assertEqual(
            total_time_on_test(times, 3),
            32,
        )

    def test_mle(self):
        self.assertAlmostEqual(
            mle(32, 3),
            32 / 3,
        )

    def test_naive_estimator(self):
        times = [3, 8, 5, 12, 20]

        self.assertEqual(
            naive_estimator(times, 3),
            np.mean([3, 5, 8]),
        )
    def test_no_censoring_ttt(self):
        times = [3, 5, 8, 12, 20]
        self.assertEqual(total_time_on_test(times, 5), sum(times))

    def test_exact_interval_positive(self):
        lower, upper = exact_confidence_interval(32, 3)
        self.assertGreater(lower, 0)
        self.assertGreater(upper, lower)

    def test_wald_interval_centered_at_mle(self):
        lower, upper = wald_confidence_interval(32, 3)
        theta_hat = mle(32, 3)

        self.assertAlmostEqual((lower + upper) / 2, theta_hat)

    def test_log_wald_interval_positive(self):
        lower, upper = log_wald_confidence_interval(32, 3)

        self.assertGreater(lower, 0)
        self.assertGreater(upper, lower)


if __name__ == "__main__":
    unittest.main()