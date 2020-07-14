from unittest import mock

import numpy as np
import pytest

from gustavgrad import Tensor
from gustavgrad.module import Module, Parameter
from gustavgrad.optim import SGD


class TestSGD:
    class Model(Module):
        def __init__(self) -> None:
            self.layer1 = Parameter(2, 10)
            self.layer2 = Parameter(10, 5)

        def predict(self, x: Tensor) -> Tensor:
            # x has shape(batch_size, 2)
            x = x @ self.layer1  # (batch_size, 10)
            x = x @ self.layer2  # (batch_size, 5)
            return x

    @pytest.fixture
    def module_with_grad(self) -> Model:
        model = self.Model()
        y = model.predict(Tensor([[1, 2], [1, 3]]))
        y.backward(np.ones(shape=(2, 5)))

        for param in model.parameters():
            assert not np.allclose(param.grad, 0)

        return model

    @pytest.fixture
    def module_with_invalidated_grad(self) -> Model:
        model = self.Model()
        model.layer1.grad = None
        model.layer2.grad = None

        return model

    @pytest.mark.parametrize("lr", [0.001, 0.01, 0.1, 1])
    def test_gradient_descent_step(self, lr, module_with_grad):

        layer1_expected_value = (
            module_with_grad.layer1.data - module_with_grad.layer1.grad * lr
        )
        layer2_expected_value = (
            module_with_grad.layer2.data - module_with_grad.layer2.grad * lr
        )

        sgd = SGD(lr)
        sgd.step(module_with_grad)

        np.testing.assert_allclose(
            module_with_grad.layer1.data, layer1_expected_value
        )
        np.testing.assert_allclose(
            module_with_grad.layer2.data, layer2_expected_value
        )

    @mock.patch("gustavgrad.module.Parameter.__isub__")
    def test_updates_all_parameters(self, mocked_function, module_with_grad):
        sgd = SGD()
        sgd.step(module_with_grad)

        assert Parameter.__isub__.call_count == 2

    @mock.patch("gustavgrad.module.Parameter.__isub__")
    def test_ignores_invalidated_grad(
        self, mocked_function, module_with_invalidated_grad
    ):
        sgd = SGD()
        sgd.step(module_with_invalidated_grad)

        assert Parameter.__isub__.call_count == 0

    def test_invaldates_gradients(self, module_with_grad):
        """ Assert that updating the parameters of a Module in a SGD step
        invalidates their gradients"""

        sgd = SGD()
        sgd.step(module_with_grad)

        for parameter in module_with_grad.parameters():
            assert parameter.grad is None
