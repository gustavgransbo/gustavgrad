import numpy as np

from gustavgrad import Tensor
from gustavgrad.loss import LogitBinaryCrossEntropy, SquaredErrorLoss


class TestLoss:
    def test_binary_cross_entropy_with_logits_correct(self) -> None:

        targets = np.asarray([1.0, 1, 1, 1, 0, 0, 0])
        # Good logits
        logits = Tensor(
            [1000.0, 1000, 1000, 1000, -1000, -1000, -1000], requires_grad=True
        )

        bce_loss = LogitBinaryCrossEntropy()

        loss = bce_loss.loss(targets, logits)
        np.testing.assert_allclose(loss.data, 0)
        loss.backward()
        np.testing.assert_allclose(logits.grad, 0)

    def test_binary_cross_entropy_with_logits_wrong(self) -> None:

        targets = np.asarray([1.0, 1, 1, 1, 0, 0, 0])
        # Bad logits
        logits = Tensor(
            [-1000.0, -1000, -1000, -1000, 1000, 1000, 1000],
            requires_grad=True,
        )

        bce_loss = LogitBinaryCrossEntropy()

        loss = bce_loss.loss(targets, logits)
        np.testing.assert_allclose(loss.data, 1000)
        loss.backward()
        np.testing.assert_allclose(logits.grad, [-1, -1, -1, -1, 1, 1, 1])

    def test_squared_error_loss_correct(self) -> None:

        targets = np.arange(5)
        # Perfect predictions
        predictions = Tensor(np.arange(5), requires_grad=True)

        se_loss = SquaredErrorLoss()

        loss = se_loss.loss(targets, predictions)
        np.testing.assert_allclose(loss.data, 0)
        loss.backward()
        np.testing.assert_allclose(predictions.grad, np.zeros(5))

    def test_squared_error_loss_wrong(self) -> None:

        targets = np.arange(5)
        # Perfect predictions
        predictions = Tensor(np.zeros(5), requires_grad=True)

        se_loss = SquaredErrorLoss()

        loss = se_loss.loss(targets, predictions)
        np.testing.assert_allclose(loss.data, 1.0 + 4 + 9 + 16)
        loss.backward()
        np.testing.assert_allclose(predictions.grad, [0, -2, -4, -6, -8])
