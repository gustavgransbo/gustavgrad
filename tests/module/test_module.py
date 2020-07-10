import numpy as np
import pytest

from gustavgrad import Tensor
from gustavgrad.module import Module, Parameter


class TestModule:
    class Model(Module):
        def __init__(self) -> None:
            self.layer1 = Parameter(2, 10)
            self.layer2 = Parameter(10, 5)

        def predict(self, x: Tensor) -> Tensor:
            # x has shape(batch_size, 2)
            x = x @ self.layer1  # (batch_size, 10)
            x = x @ self.layer2  # (batch_size, 5)
            return x

    class ModelWithSubModel(Module):
        def __init__(self) -> None:
            self.sub_model = TestModule.Model()
            self.layer2 = Parameter(5, 1)

        def predict(self, x: Tensor) -> Tensor:
            # x has shape(batch_size, 2)
            x = self.sub_model.predict(x)  # (batch_size, 5)
            x = x @ self.layer2  # (batch_size, 1)
            return x

    @pytest.fixture
    def model(self) -> Model:
        return TestModule.Model()

    @pytest.fixture
    def model_with_non_zero_grad(self, model: Model) -> Model:
        y = model.predict(Tensor([[1, 2], [1, 3]]))
        y.backward(np.ones(shape=(2, 5)))

        # Assert all gradients are non-zero after backprop
        assert not np.allclose(model.layer1.grad, 0)
        assert not np.allclose(model.layer2.grad, 0)

        return model

    @pytest.fixture
    def model_with_sub_model(self) -> Model:
        return TestModule.ModelWithSubModel()

    @pytest.fixture
    def model_with_sub_model_and_non_zero_grad(
        self, model_with_sub_model: ModelWithSubModel
    ) -> Model:
        y = model_with_sub_model.predict(Tensor([[1, 2], [1, 3]]))
        y.backward(np.ones(shape=(2, 1)))

        # Assert all gradients are non-zero after backprop
        assert not np.allclose(model_with_sub_model.sub_model.layer1.grad, 0)
        assert not np.allclose(model_with_sub_model.sub_model.layer2.grad, 0)
        assert not np.allclose(model_with_sub_model.layer2.grad, 0)

        return model_with_sub_model

    def test_parameters_simple_module(self, model: Model) -> None:
        assert len(list(model.parameters())) == 2

    def test_parameters_module_with_sub_module(
        self, model_with_sub_model: ModelWithSubModel
    ) -> None:
        assert len(list(model_with_sub_model.parameters())) == 3

    def test_zero_grad_simple_module(
        self, model_with_non_zero_grad: Model
    ) -> None:
        model_with_non_zero_grad.zero_grad()

        # Assert all gradients are zero after zero_grad
        np.testing.assert_allclose(model_with_non_zero_grad.layer1.grad, 0)
        np.testing.assert_allclose(model_with_non_zero_grad.layer2.grad, 0)

    def test_zero_grad_simple_module_with_sub_module(
        self, model_with_sub_model_and_non_zero_grad: ModelWithSubModel
    ) -> None:
        model_with_sub_model_and_non_zero_grad.zero_grad()

        # Assert all gradients are zero after zero_grad
        np.testing.assert_allclose(
            model_with_sub_model_and_non_zero_grad.sub_model.layer1.grad, 0
        )
        np.testing.assert_allclose(
            model_with_sub_model_and_non_zero_grad.sub_model.layer2.grad, 0
        )
        np.testing.assert_allclose(
            model_with_sub_model_and_non_zero_grad.layer2.grad, 0
        )
