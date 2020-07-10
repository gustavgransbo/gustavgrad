from typing import List, Tuple

import numpy as np
import pytest

from gustavgrad import Tensor
from gustavgrad.loss import LogitBinaryCrossEntropy, SquaredErrorLoss


class TestLogitBinaryCrossEntropy:
    def targets_and_correct_logits(
        self, requires_grad: bool = True
    ) -> Tuple[List[float], Tensor]:
        targets = np.asarray([1.0, 1, 1, 1, 0, 0, 0])
        correct_logits = Tensor(
            [1000.0, 1000, 1000, 1000, -1000, -1000, -1000],
            requires_grad=requires_grad,
        )
        return targets, correct_logits

    @pytest.fixture
    def targets_and_correct_logits_requires_grad(
        self,
    ) -> Tuple[List[float], Tensor]:
        return self.targets_and_correct_logits(requires_grad=True)

    @pytest.fixture
    def targets_and_correct_logits_requires_no_grad(
        self,
    ) -> Tuple[List[float], Tensor]:
        return self.targets_and_correct_logits(requires_grad=False)

    def test_binary_cross_entropy_with_logits_correct(
        self,
        targets_and_correct_logits_requires_grad: Tuple[List[float], Tensor],
    ) -> None:
        """ Assert loss behaves well when logits align with targets"""
        targets, correct_logits = targets_and_correct_logits_requires_grad

        bce_loss = LogitBinaryCrossEntropy()
        loss = bce_loss.loss(targets, correct_logits)

        # Assert zero loss
        np.testing.assert_allclose(loss.data, 0)

        loss.backward()

        # Assert zero gradient
        np.testing.assert_allclose(correct_logits.grad, 0)

    def test_binary_cross_entropy_no_grad(
        self,
        targets_and_correct_logits_requires_no_grad: Tuple[
            List[float], Tensor
        ],
    ) -> None:
        """ Assert loss behaves well when logits align with targets"""
        targets, logits = targets_and_correct_logits_requires_no_grad

        bce_loss = LogitBinaryCrossEntropy()
        loss = bce_loss.loss(targets, logits)

        assert not loss.requires_grad

    def test_binary_cross_entropy_with_logits_wrong(self) -> None:
        """ Assert loss behaves well when logits are the opposite of targets"""

        targets = np.asarray([1.0, 1, 1, 1, 0, 0, 0])
        wrong_logits = Tensor(
            [-1000.0, -1000, -1000, -1000, 1000, 1000, 1000],
            requires_grad=True,
        )

        bce_loss = LogitBinaryCrossEntropy()
        loss = bce_loss.loss(targets, wrong_logits)

        # Assert large loss (1000)
        np.testing.assert_allclose(loss.data, 1000)

        loss.backward()

        # Assert correct gradient
        np.testing.assert_allclose(
            wrong_logits.grad, [-1, -1, -1, -1, 1, 1, 1]
        )


class TestSquaredErrorLoss:
    def test_squared_error_loss_correct(self) -> None:
        """ Assert loss behaves correctly for correct predictions"""

        targets = np.arange(5)
        perfect_predictions = Tensor(np.arange(5), requires_grad=True)

        se_loss = SquaredErrorLoss()

        loss = se_loss.loss(targets, perfect_predictions)

        # Assert zero loss
        np.testing.assert_allclose(loss.data, 0)

        loss.backward()

        # Assert zero grad
        np.testing.assert_allclose(perfect_predictions.grad, 0)

    def test_squared_error_loss_wrong(self) -> None:
        """ Assert loss behvases correctly for incorrect predictions"""

        targets = np.arange(5)
        incorrect_predictions = Tensor(np.zeros(5), requires_grad=True)

        se_loss = SquaredErrorLoss()

        loss = se_loss.loss(targets, incorrect_predictions)

        # Assert correct loss
        np.testing.assert_allclose(loss.data, 1.0 + 4 + 9 + 16)

        loss.backward()

        # Assert correct gradient
        np.testing.assert_allclose(
            incorrect_predictions.grad, [0, -2, -4, -6, -8]
        )
