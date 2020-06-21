import math
import unittest

import numpy as np

from autograd import Tensor
from autograd.loss import LogitBinaryCrossEntropy, SquaredErrorLoss


class TestLoss(unittest.TestCase):
    def test_binary_cross_entropy_with_logits_correct(self) -> None:

        targets = np.asarray([1.0, 1, 1, 1, 0, 0, 0])
        # Good logits
        logits = Tensor(
            [1000.0, 1000, 1000, 1000, -1000, -1000, -1000], requires_grad=True
        )

        bce_loss = LogitBinaryCrossEntropy()

        loss = bce_loss.loss(targets, logits)
        assert math.isclose(loss.data.tolist(), 0.0, rel_tol=1e-5)
        loss.backward()
        np.testing.assert_array_almost_equal(logits.grad, np.zeros(7))

    def test_binary_cross_entropy_with_logits_wrong(self) -> None:

        targets = np.asarray([1.0, 1, 1, 1, 0, 0, 0])
        # Bad logits
        logits = Tensor(
            [-1000.0, -1000, -1000, -1000, 1000, 1000, 1000],
            requires_grad=True,
        )

        bce_loss = LogitBinaryCrossEntropy()

        loss = bce_loss.loss(targets, logits)
        assert math.isclose(loss.data.tolist(), 1000.0, rel_tol=1e-5)
        loss.backward()
        np.testing.assert_array_almost_equal(
            logits.grad, [-1, -1, -1, -1, 1, 1, 1]
        )

    def test_squared_error_loss_correct(self) -> None:

        targets = np.arange(5)
        # Perfect predictions
        predictions = Tensor(np.arange(5), requires_grad=True)

        se_loss = SquaredErrorLoss()

        loss = se_loss.loss(targets, predictions)
        assert math.isclose(loss.data.tolist(), 0.0, rel_tol=1e-5)
        loss.backward()
        np.testing.assert_array_almost_equal(predictions.grad, np.zeros(5))

    def test_squared_error_loss_wrong(self) -> None:

        targets = np.arange(5)
        # Perfect predictions
        predictions = Tensor(np.zeros(5), requires_grad=True)

        se_loss = SquaredErrorLoss()

        loss = se_loss.loss(targets, predictions)
        assert math.isclose(loss.data.tolist(), 1.0 + 4 + 9 + 16, rel_tol=1e-5)
        loss.backward()
        np.testing.assert_array_almost_equal(
            predictions.grad, [0, -2, -4, -6, -8]
        )


if __name__ == "__main__":
    TestLoss().test_binary_cross_entropy_with_logits_correct()
