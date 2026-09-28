import unittest
import numpy as np

from src.inference import (
    total_time_on_test,
    mle,
    naive_estimator,
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


if __name__ == "__main__":
    unittest.main()